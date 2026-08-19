'''
Geometry-only same-scan object fusion.

Replaces the appearance-influenced association path that used to live in
rerun_realtime_mapping.py
(spatial_sim + CLIP visual_sim -> sim_sum -> argmax -> merge_obj_matches, plus the
periodic CLIP-gated merge_objects()) with a pure-geometry cascade:

    New detection D
        -> 3D AABB(+margin) overlap gate          (candidate filtering only)
        -> association gate            max(strong, weak) >= thresh
        -> merge EVERY strong match, plus the single closest weak one

(A surface-normal-consistency gate used to sit after the association gate here too.
Dropped from THIS cascade specifically: it was blind to two flush, same-facing surfaces
anyway -- see the abs()-because-unoriented note on _normal_consistency -- and it was
actively rejecting genuine matches between different faces of the same large object
growing within one continuous scan, e.g. a cabinet's front and side panel, whose local
normals legitimately differ. That fragmented single large objects into one node per
face. It's still used in evaluate_object_pair_gates' post-scan consolidation sweep,
where that reasoning doesn't apply -- see that function's own docstring.)

Both directions come from ONE screen-space intersection of the detection's 2D mask with
the object's projected footprint, differing only in the denominator (ASSOCIATION_MODE_
PROJECTION, the default -- see _association_by_projection):

    strong = |mask(D) & footprint(O_vis)| / |mask(D)|            "how much of D is O?"
    weak   = |mask(D) & footprint(O_vis)| / |footprint(O_vis)|   "how much of O is in D?"

footprint(O_vis) projects the object's points that this camera should have seen (frustum
+ z-buffer against the frame's own depth map, _visible_point_mask) and closes the scatter
into an area (projected_footprint); mask(D) needs no projection at all, being this
frame's actual segmentation.

Neither direction suffices alone. strong fails whenever a detection reveals surface the
object has not accumulated yet (walking around a table), spawning a duplicate node per
viewpoint. weak fails whenever a 2D mask covers only part of the object -- which is why
the footprint is restricted to what was visible from here.

A pure-3D nearest-neighbour variant of both ratios (ASSOCIATION_MODE_POINT_3D --
counting a point as "covered" whenever some point of the other cloud lies within tau,
with no camera/projection involved at all) was tried in between and reverted back to
this projection-based default. It has a real advantage for a "before"-scan-seeded object
(see load_prior_scene_objects_as_seeds in slam/utils.py): projection's occlusion test
is tolerance-based and can't always tell "the same surface" apart from "a new thin
object now sitting flush against it" (e.g. a book on a previously-bare desk seed), so a
seed's stale footprint can swallow the new object's own detection where a pure 3D check
wouldn't. But 3D reopened a worse, more common problem: "some point of the other cloud
lies within tau" is weak evidence, and on screen a point a centimetre to the side lands
on a different pixel/mask, so sideways leakage across genuinely different-but-touching
objects (a blanket draped on a sofa, a pillow resting on a cushion) was impossible under
projection and came back under 3D -- see the leak discussion below, which is what this
reverts to fixing. The seed/thin-object-on-seed risk 3D was covering is back as a known
gap in the SPATIAL gate specifically; _eroded_mask_subset (still used by the 3D fallback
paths described below) does not help it, since that trims the DETECTION's own mask
boundary, not the SEED's stale footprint. What does help: evaluate_detection_gates now
backstops a seed-descended object (has 'confirmed_pcd', see load_prior_scene_objects_as_seeds)
with a CLIP+DINO similarity check whenever that object's this-scan-only geometry can't yet
independently confirm the spatial match on its own -- a new book on a desk seed looks
nothing like the desk it's spatially close to, so it fails that check even though the
desk's stale footprint still spatially swallows it. See evaluate_detection_gates' own
docstring for the exact two-stage rule.

Sideways leakage on screen is impossible in principle -- a point a centimetre to the
side of a genuinely different object lands on a different pixel, which belongs to a
different mask -- but the two remaining fallback paths still use the 3D ratio and so
still carry its leak risk:

  * No `view` at all, or too few of the object's points fall in this frame's frustum to
    mean anything (has_view is False) -- the projection route is unavailable and the 3D
    D->O ratio alone decides.
  * `_association_by_projection` itself fails to form a footprint (object not visible
    this frame, or a shape mismatch) even though has_view was True.

For these fallback cases, _eroded_mask_subset restricts the detection's own points to
this frame's SAM mask eroded inward by tau (this frame's own depth/camera only, no
cross-scan risk) before the 3D ratio is computed, trimming boundary points -- the same
mitigation the 3D-only design relied on, kept here as a strictly-improving safety net
for a path that's now secondary rather than the primary decision-maker.

The two directions also carry different authority, which is why the merge policy is
asymmetric. A strong match asserts the detection IS substantially that object, so several
at once mean an earlier frame split one object into several nodes and this observation
should bridge them. A weak match only asserts the detection's silhouette encloses the
node -- something a host detection does for every object lying on it -- so merging all of
them would collapse a sofa and everything on it into one node. Only the closest weak
match merges; the rest are logged with reason "weak_not_closest".

When a detector misses a small object sitting on a large one, SAM's mask for the large
object swallows the small one's pixels, those pixels' depth IS the small object's
surface, and the small node is genuinely covered, so it still merges away. That is a
detection failure and is handled where it happens; the debug log records enough
(gate_class plus a near-zero strong ratio) to find the frames where it occurred.

clip_ft / dino_ft / text_ft are never read here -- features are still extracted and
stored by the mapping loop for the later before/after graph matching stage.
'''

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
from tqdm import trange

from conceptgraph.slam.slam_classes import DetectionList, MapObjectList
from conceptgraph.slam.utils import from_intrinsics_matrix, get_bounding_box, merge_obj2_into_obj1

# How BOTH association directions are measured (see the module docstring).
ASSOCIATION_MODE_PROJECTION = "projection"  # 2D: one screen-space intersection, two denominators (default)
ASSOCIATION_MODE_POINT_3D = "point_3d"      # 3D: nearest-neighbour ratios, kept for comparison/fallback

# reason codes written into the debug jsonl
REASON_TOO_FEW_POINTS = "too_few_points"
REASON_BBOX_REJECT = "bbox_reject"
REASON_POINT_OVERLAP_REJECT = "point_overlap_reject"
# Passed the spatial gate against a seed-descended object's FULL geometry (seed +
# confirmed), but that object's confirmed-only share (see slam/utils.py's
# load_prior_scene_objects_as_seeds) didn't independently pass, and CLIP+DINO
# similarity to the detection fell short too -- see evaluate_detection_gates.
REASON_APPEARANCE_REJECT = "appearance_reject"
# evaluate_object_pair_gates only (see its docstring for why online per-frame fusion
# doesn't use this): the tau-inlier correspondence points' surface normals disagree.
REASON_NORMAL_REJECT = "normal_reject"
REASON_MERGE = "merge"
REASON_DEFERRED = "deferred_to_next_sweep"
# Passed both gates on the weak (containment) direction, but another weak candidate
# matched this detection better and only the closest one is allowed to merge.
REASON_WEAK_NOT_CLOSEST = "weak_not_closest"


@dataclass
class GeometryFusionParams:
    '''
    Everything the geometry gates need, resolved from cfg once per scene so the
    per-frame call sites don't carry a 15-argument signature around.
    '''
    # gates
    bbox_margin: float                  # AABB slack [m]
    point_distance_thresh: float        # tau = downsample_voxel_size * point_distance_factor
    point_overlap_thresh: float
    min_points_for_gates: int
    visibility_depth_tolerance: float   # z-buffer slack [m] for the occlusion test
    min_visible_points: int             # below this, the containment direction carries no evidence
    association_gate_mode: str          # ASSOCIATION_MODE_PROJECTION (default) | ASSOCIATION_MODE_POINT_3D
    projection_close_factor: float      # closing radius = factor * fx * voxel / z  [px]
    seed_semantic_sim_threshold: float  # CLIP+DINO similarity floor for a seed-descended
                                         # object whose confirmed-only geometry doesn't yet
                                         # independently pass the spatial gate on its own
    # evaluate_object_pair_gates only -- see its docstring for why online per-frame
    # fusion (evaluate_detection_gates) doesn't use these.
    normal_radius: float                 # downsample_voxel_size * normal_radius_factor
    normal_max_nn: int
    normal_cos_thresh: float             # cos(normal_angle_thresh_deg)
    normal_consistency_thresh: float
    max_sweeps: int
    # recognition-trust thresholds, applied by annotate_recognition_trust() after
    # compute_recognition_confidence() has populated the fields it reads. Both are
    # minimums an object must MEET to be trusted -- nothing is removed either way.
    recognition_min_recognized_frames: int
    recognition_min_confidence: float
    # merge_obj2_into_obj1 passthrough
    downsample_voxel_size: float
    dbscan_remove_noise: bool
    dbscan_eps: float
    dbscan_min_points: int
    spatial_sim_type: str
    device: str


def geometry_fusion_params_from_cfg(cfg) -> GeometryFusionParams:
    voxel = float(cfg['downsample_voxel_size'])
    point_distance_thresh = voxel * float(cfg['fusion_point_distance_factor'])
    return GeometryFusionParams(
        bbox_margin=float(cfg['fusion_bbox_margin']),
        point_distance_thresh=point_distance_thresh,
        point_overlap_thresh=float(cfg['fusion_point_overlap_thresh']),
        min_points_for_gates=int(cfg['fusion_min_points_for_gates']),
        visibility_depth_tolerance=float(cfg['fusion_visibility_depth_tolerance']),
        min_visible_points=int(cfg['fusion_min_visible_points']),
        association_gate_mode=str(cfg['fusion_association_gate_mode']),
        projection_close_factor=float(cfg['fusion_projection_close_factor']),
        seed_semantic_sim_threshold=float(cfg['fusion_seed_semantic_sim_threshold']),
        normal_radius=voxel * float(cfg['fusion_normal_radius_factor']),
        normal_max_nn=int(cfg['fusion_normal_max_nn']),
        normal_cos_thresh=math.cos(math.radians(float(cfg['fusion_normal_angle_thresh_deg']))),
        normal_consistency_thresh=float(cfg['fusion_normal_consistency_thresh']),
        max_sweeps=int(cfg['fusion_global_consolidation_max_sweeps']),
        recognition_min_recognized_frames=int(cfg['recognition_min_recognized_frames']),
        recognition_min_confidence=float(cfg['recognition_min_confidence']),
        downsample_voxel_size=voxel,
        dbscan_remove_noise=bool(cfg['dbscan_remove_noise']),
        dbscan_eps=float(cfg['dbscan_eps']),
        dbscan_min_points=int(cfg['dbscan_min_points']),
        spatial_sim_type=str(cfg['spatial_sim_type']),
        device=str(cfg['device']),
    )


class FusionDebugWriter:
    '''
    Appends one json object per line to <out_dir>/{online_fusion,global_consolidation}.jsonl.

    Only candidates that pass the bbox gate get a detailed record; bbox-rejected pairs
    would be frames x detections x objects lines (millions), so they're counted in the
    per-detection summary record instead.
    '''

    def __init__(self, out_dir, enabled: bool = True):
        self.enabled = bool(enabled)
        self._files = {}
        if self.enabled:
            self.out_dir = Path(out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _file(self, name):
        if name not in self._files:
            # "w": each scene's log belongs to one run, so a re-run replaces it rather
            # than appending to the previous run's records. Opened once per writer, so
            # everything within a scene still accumulates.
            self._files[name] = open(self.out_dir / f"{name}.jsonl", "w")
        return self._files[name]

    def log(self, name, record):
        if not self.enabled:
            return
        f = self._file(name)
        f.write(json.dumps(record, default=_json_default) + "\n")

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------- geometry helpers

def _points(pcd) -> np.ndarray:
    return np.asarray(pcd.points, dtype=np.float64)


def aabb_from_points(points):
    '''
    Axis-aligned (lo, hi) of a raw (N, 3) point array, each shape (3,).

    Split out from _aabb so callers holding plain numpy -- the before/after benchmark
    comparison reads 'pcd_np' straight out of the saved pickle, with no open3d cloud
    to hand -- can reuse the same box the fusion gates use, and feed it to the equally
    numpy-native _bbox_gate_vector.
    '''
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        nan = np.full(3, np.nan)
        return nan, nan
    return pts.min(axis=0), pts.max(axis=0)


def _aabb(pcd):
    '''Axis-aligned (lo, hi) straight off the point cloud, each shape (3,).'''
    return aabb_from_points(_points(pcd))


def _stack_aabbs(objects):
    '''(N,3) lo and (N,3) hi for a whole object list.'''
    if len(objects) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    los, his = zip(*(_aabb(o['pcd']) for o in objects))
    return np.stack(los), np.stack(his)


def _bbox_gate_vector(lo_a, hi_a, los_b, his_b, margin):
    '''
    Boxes count as overlapping when they are within `margin` of each other, i.e.
    (lo_a - margin <= hi_b) & (hi_a + margin >= lo_b) on all three axes.
    Deliberately more permissive than the oriented-box IoU used elsewhere -- this
    is candidate filtering only, and an AABB never raises the "Plane vertices are
    not coplanar" error that pytorch3d's box3d_overlap can.
    '''
    if len(los_b) == 0:
        return np.zeros(0, dtype=bool)
    return (((lo_a - margin) <= his_b) & ((hi_a + margin) >= los_b)).all(axis=1)


def _normals(pcd, params: GeometryFusionParams) -> np.ndarray:
    '''
    Lazily estimated per-point normals, cached on the point cloud itself. Used by
    evaluate_object_pair_gates only (see its docstring for why online per-frame fusion
    doesn't use this).

    Stored in pcd.normals rather than a new object-dict key on purpose:
    merge_obj2_into_obj1 raises ValueError on unhandled object keys, and open3d
    invalidates normals for us -- PointCloud.__iadd__ clears them when either side
    lacks them, and pcd_denoise_dbscan rebuilds the cloud from points/colors only.
    So a length mismatch is a reliable "stale" signal.
    '''
    if len(pcd.normals) != len(pcd.points):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=params.normal_radius, max_nn=params.normal_max_nn
            )
        )
    return np.asarray(pcd.normals, dtype=np.float64)


def _invalidate_normals(pcd):
    '''
    Force recomputation on next use (called right after any merge changes an object's
    geometry, both online per-frame merges and global-consolidation ones -- even though
    only the latter reads normals, a later consolidation sweep still needs them fresh
    for whatever an online merge changed in between).
    '''
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))


@dataclass
class FrameView:
    '''
    Everything needed to ask "which of these world points should this frame's camera
    have seen?" -- the pose inverse, the intrinsics, and the frame's own depth map,
    all already available in the mapping loop.
    '''
    world_to_cam: np.ndarray    # (4, 4)
    fx: float
    fy: float
    cx: float
    cy: float
    depth: np.ndarray           # (H, W), same map the detections were backprojected from

    @classmethod
    def from_frame(cls, cam_to_world, cam_K, depth_array) -> "FrameView":
        fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
        return cls(
            world_to_cam=np.linalg.inv(np.asarray(cam_to_world, dtype=np.float64)),
            fx=fx, fy=fy, cx=cx, cy=cy,
            depth=np.asarray(depth_array),
        )


def _visible_point_mask(points, view: FrameView, params: GeometryFusionParams) -> np.ndarray:
    '''
    Boolean mask over world-frame `points`: True where the point is in front of the
    camera, projects inside the image, and is not occluded according to the frame's
    own depth map (standard z-buffer test, z <= depth + tolerance).

    Pixels with invalid depth (0) count as NOT visible: the detections were
    backprojected from this same depth map, so a point with no depth reading could
    never have been detected here and must not be held against the ratio.
    '''
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    cam = points @ view.world_to_cam[:3, :3].T + view.world_to_cam[:3, 3]
    z = cam[:, 2]
    visible = z > 0
    if not visible.any():
        return visible

    h, w = view.depth.shape[:2]
    zs = np.where(visible, z, 1.0)                       # avoid divide-by-zero on culled points
    u = np.rint(view.fx * cam[:, 0] / zs + view.cx).astype(np.int64)
    v = np.rint(view.fy * cam[:, 1] / zs + view.cy).astype(np.int64)
    visible &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not visible.any():
        return visible

    idx = np.nonzero(visible)[0]
    measured = view.depth[v[idx], u[idx]]
    occluded_or_unmeasured = (measured <= 0) | (z[idx] > measured + params.visibility_depth_tolerance)
    visible[idx[occluded_or_unmeasured]] = False
    return visible


def _project_pixels(points, visible, view: FrameView):
    '''
    (u, v) of the visible points, in visible-point order. The clip is a no-op for points
    that came through _visible_point_mask (it already required them to be in frame) and
    a guard for anything else.
    '''
    if view is None or not visible.any():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cam = points[visible] @ view.world_to_cam[:3, :3].T + view.world_to_cam[:3, 3]
    z = cam[:, 2]
    h, w = view.depth.shape[:2]
    return (np.clip(np.rint(view.fx * cam[:, 0] / z + view.cx).astype(np.int64), 0, w - 1),
            np.clip(np.rint(view.fy * cam[:, 1] / z + view.cy).astype(np.int64), 0, h - 1))


def projected_footprint(points, view: FrameView, params: GeometryFusionParams, visible=None):
    '''
    Rasterize the object's visible points into this frame's image plane and return
    (footprint, visible_mask, radius) -- the area the object occupies on screen, the
    boolean mask selecting which input points contributed, and the closing radius used.

    Why an area and not just the hit pixels: point clouds are voxel-downsampled, so a
    projection is a scatter of dots, not a silhouette -- a detection's own cloud covers
    barely 3% of its own mask (a sofa mask of 34k pixels holds 986 points). Morphological
    CLOSING (dilate then erode) fills the gaps that sparsity opened while leaving the
    outer boundary essentially where it was. Plain dilation is not a substitute: it pushes
    the boundary outward onto whatever the object is resting on, which is exactly the
    confusion this whole direction exists to avoid.

    The radius is derived, not tuned: fx * downsample_voxel_size / z is the pixel footprint
    of one voxel at that depth. A radius fixed in pixels would be meaningless here anyway,
    since the reconstruction's scale is normalized rather than metric.
    '''
    if view is None or len(points) == 0:
        return None, np.zeros(len(points), dtype=bool), 0

    if visible is None:
        visible = _visible_point_mask(points, view, params)
    if not visible.any():
        return None, visible, 0

    u, v = _project_pixels(points, visible, view)
    h, w = view.depth.shape[:2]
    hits = np.zeros((h, w), dtype=np.uint8)
    hits[v, u] = 1

    z = (points[visible] @ view.world_to_cam[:3, :3].T + view.world_to_cam[:3, 3])[:, 2]
    radius = int(round(params.projection_close_factor
                       * float(np.median(view.fx * params.downsample_voxel_size / z))))
    if radius < 1:
        return hits.astype(bool), visible, 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(hits, cv2.MORPH_CLOSE, kernel).astype(bool), visible, radius


def trimmed_surface_distance(points_a, points_b, keep_fraction=0.7):
    """
    How far apart two point clouds sit, ignoring the worst-matching tail: take every
    point's nearest-neighbour distance in BOTH directions, sort the pooled distances,
    and average the smallest `keep_fraction` of them.

    Pooling both directions is what makes this an identity test rather than a
    containment test. A cushion resting on a sofa has every cushion->sofa distance
    near zero, so the one-directional average would call them the same object; most
    sofa->cushion distances are large, and pooling lets them dominate (the pool is
    sized |A| + |B|, so the larger cloud contributes proportionally more). Trimming
    the tail is what tolerates the partial overlap two scans of the same object
    almost always have.

    Used by the benchmark comparison to settle "this object simply did not move"
    before any appearance similarity is consulted -- the same question, and the same
    kind of answer, as the position-only merging done between frames during a scan.
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    d_a2b, _ = cKDTree(b).query(a)
    d_b2a, _ = cKDTree(a).query(b)
    pooled = np.sort(np.concatenate([d_a2b, d_b2a]))
    keep = max(1, int(round(keep_fraction * len(pooled))))
    return float(pooled[:keep].mean())


def _directional_overlap(pcd_src, pcd_dst, params: GeometryFusionParams, src_subset=None, dst_subset=None):
    '''
    Overlap(src -> dst) = #{p in src : min_{q in dst} ||p - q|| < tau} / |src|,
    along with the nearest-neighbor correspondence that produced it.

    `src_subset`, when given, restricts the source points (and hence the denominator)
    to those indices -- that's how the visibility-normalized O_vis->D direction is
    computed. `dst_subset` restricts which points of the destination cloud can be
    matched against at all -- that's how a detection's own mask-eroded points
    (_eroded_mask_subset) keep boundary points out of the search when the detection is
    the destination (the weak direction). Both returned index arrays are always in
    their respective FULL cloud's index space.

    Returns (overlap, src_inlier_idx, dst_nn_idx) -- the two index arrays are the
    tau-inlier correspondence pairs (currently unused by callers, kept for whatever
    debugging/analysis wants to know exactly which points matched).
    '''
    src_all = _points(pcd_src)
    src_pts = src_all if src_subset is None else src_all[src_subset]
    dst_all = _points(pcd_dst)
    dst_pts = dst_all if dst_subset is None else dst_all[dst_subset]
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return 0.0, np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    dists, nn_idx = cKDTree(dst_pts).query(src_pts)
    local_inlier_idx = np.nonzero(dists < params.point_distance_thresh)[0]
    src_inlier_idx = local_inlier_idx if src_subset is None else np.asarray(src_subset)[local_inlier_idx]
    dst_nn_local = nn_idx[local_inlier_idx]
    dst_nn_idx = dst_nn_local if dst_subset is None else np.asarray(dst_subset)[dst_nn_local]
    return float(len(local_inlier_idx)) / len(src_pts), src_inlier_idx, dst_nn_idx


def _normal_consistency(pcd_src, pcd_dst, src_idx, dst_idx, params: GeometryFusionParams):
    '''
    Fraction of the tau-inlier correspondences whose surface normals agree:
    |n_src . n_dst| >= cos(angle_thresh). Absolute value because estimate_normals
    leaves normals unoriented (a surface seen from opposite sides flips sign).

    evaluate_object_pair_gates only -- see its docstring for why online per-frame fusion
    doesn't use this.

    The denominator is the number of tau-inlier correspondences, not |src| -- points
    with no neighbor inside tau aren't correspondences at all, so including them
    would just re-apply the point-overlap gate a second time.
    '''
    if len(src_idx) == 0:
        return 0.0
    n_src = _normals(pcd_src, params)[src_idx]
    n_dst = _normals(pcd_dst, params)[dst_idx]
    cos_sim = np.abs(np.einsum('ij,ij->i', n_src, n_dst))
    return float((cos_sim >= params.normal_cos_thresh).mean())


def _eroded_mask_subset(det, det_pcd, view: "FrameView", params: GeometryFusionParams) -> np.ndarray:
    '''
    Indices into det_pcd's points that survive eroding this frame's OWN detection mask
    inward by tau (point_distance_thresh), converted from metres to pixels via this
    detection's own depth/camera -- no cross-frame or cross-scan geometry involved, only
    this frame's fresh SAM mask and this frame's own depth map.

    Points nearest a detection's mask boundary are exactly where a
    touching-but-different object's points are most likely to land within tau (the
    contact-line/band leak -- see the module docstring): trimming them here, before the
    overlap gates ever see them, removes the leak at its source instead of trying to
    detect it after the fact. The trimmed set is used only for the association-gate
    computation, never for what actually gets merged into the map -- a confirmed match
    still folds in the detection's full, un-eroded geometry.

    Falls back to ALL of det_pcd's indices (no trimming) when there's no usable
    mask/view, or when erosion would remove every point -- a detection too small to
    survive erosion should fall through to the ordinary point-overlap gate rather than
    being silently blinded here.
    '''
    all_idx = np.arange(len(_points(det_pcd)))
    masks = det.get('mask')
    if not masks or view is None:
        return all_idx
    mask = np.asarray(masks[0]).astype(bool)
    if mask.shape != view.depth.shape[:2]:
        return all_idx

    pts = _points(det_pcd)
    cam = pts @ view.world_to_cam[:3, :3].T + view.world_to_cam[:3, 3]
    z = cam[:, 2]
    in_front = z > 0
    if not in_front.any():
        return all_idx

    radius_px = int(round(params.point_distance_thresh * view.fx / float(np.median(z[in_front]))))
    if radius_px < 1:
        return all_idx
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)

    h, w = eroded.shape
    zs = np.where(in_front, z, 1.0)
    u = np.rint(view.fx * cam[:, 0] / zs + view.cx).astype(np.int64)
    v = np.rint(view.fy * cam[:, 1] / zs + view.cy).astype(np.int64)
    inside = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    keep = np.zeros(len(pts), dtype=bool)
    idx_inside = np.nonzero(inside)[0]
    keep[idx_inside] = eroded[v[idx_inside], u[idx_inside]]
    kept_idx = np.nonzero(keep)[0]
    return kept_idx if len(kept_idx) > 0 else all_idx


# ---------------------------------------------------------------- gate evaluation

class _VisibilityCache:
    '''
    Per-frame memo of "which points of this object should this camera have seen" and
    "where does it land on screen". Both depend only on the object's geometry and the
    camera, so they are the same for every detection in the frame; without this they
    would be recomputed once per (detection, candidate) pair.

    Keyed by id(pcd) while holding a reference to that pcd, so the id cannot be
    recycled onto a different cloud. A merge replaces obj['pcd'] with the new cloud
    process_pcd() returns, so merged geometry simply misses the cache. The reference is
    load-bearing, not incidental: without it nothing here keeps the point cloud alive,
    a freed pcd's memory can get reused by an unrelated later allocation with the same
    id(), and mask_for()/footprint_for() would then silently hand back arrays sized for
    the wrong cloud (an IndexError deep inside _directional_overlap's src_subset
    indexing, not here, is what that looks like from the caller's side).
    '''

    def __init__(self, view: FrameView, params: GeometryFusionParams):
        self.view = view
        self.params = params
        self._entries = {}

    def _entry(self, pcd):
        entry = self._entries.get(id(pcd))
        if entry is None:
            pts = _points(pcd)
            visible = _visible_point_mask(pts, self.view, self.params)
            footprint, _, radius = projected_footprint(pts, self.view, self.params, visible)
            entry = (pcd, visible, footprint, radius, _project_pixels(pts, visible, self.view))
            self._entries[id(pcd)] = entry
        return entry

    def mask_for(self, pcd):
        return None if self.view is None else self._entry(pcd)[1]

    def footprint_for(self, pcd):
        '''(footprint, closing radius) -- the on-screen area this object occupies.'''
        return (None, 0) if self.view is None else self._entry(pcd)[2:4]

    def pixels_for(self, pcd):
        '''(u, v) of the visible points, in visible-point order. Same for every detection.'''
        return None if self.view is None else self._entry(pcd)[4]


def _association_by_projection(det, obj_pcd, params: GeometryFusionParams, view: FrameView,
                               visibility: "_VisibilityCache", visible, record):
    '''
    Both directions from ONE screen-space intersection, differing only in the denominator:

        strong = |mask(D) & footprint(O_vis)| / |mask(D)|          "how much of D does O explain?"
        weak   = |mask(D) & footprint(O_vis)| / |footprint(O_vis)| "how much of O does D cover?"

    Returns (strong_ratio, weak_ratio, strong_normal_args, weak_normal_args); any of them
    is None when the projection could not be formed. The last two elements are legacy
    (pcd_src, pcd_dst, src_idx, dst_idx) correspondence tuples, unused now that the
    normal-consistency gate is gone, but kept as the return shape evaluate_detection_gates
    still unpacks.

    mask(D) is this frame's actual SAM mask, so the strong direction needs no reprojection
    of the detection at all -- it is already the ground truth for "where D is on screen".
    det['mask'] is a list because merging concatenates it; entry 0 is this frame's mask,
    the only one that shares a camera with `view`. It comes from gobs after resize_gobs,
    so it already matches the depth map's dimensions.
    '''
    masks = det.get('mask')
    if not masks:
        return None, None, None, None
    det_mask = np.asarray(masks[0]).astype(bool)

    det_pcd = det['pcd']
    obj_pts = _points(obj_pcd)
    if visibility is not None:
        footprint, radius = visibility.footprint_for(obj_pcd)
        u, v = visibility.pixels_for(obj_pcd)
    else:
        footprint, _, radius = projected_footprint(obj_pts, view, params, visible)
        u, v = _project_pixels(obj_pts, visible, view)
    if footprint is None or det_mask.shape != footprint.shape:
        return None, None, None, None

    footprint_area = int(footprint.sum())
    mask_area = int(det_mask.sum())
    record["n_footprint_pixels"] = footprint_area
    record["n_det_mask_pixels"] = mask_area
    record["projection_close_radius"] = int(radius)
    if footprint_area == 0 or mask_area == 0:
        return None, None, None, None

    intersection = int((footprint & det_mask).sum())
    weak_ratio = float(intersection) / footprint_area
    strong_ratio = float(intersection) / mask_area
    record["weak_projection_ratio"] = weak_ratio
    record["strong_projection_ratio"] = strong_ratio

    return strong_ratio, weak_ratio, None, None


def _association_overlap(det, det_eroded_idx, target_pcd, params: GeometryFusionParams,
                          view: FrameView, visibility: "_VisibilityCache", record: dict):
    '''
    Core of the association gate, generalized to any target point cloud -- normally
    obj['pcd'], but evaluate_detection_gates also calls this against
    obj['confirmed_pcd'] (a seed-descended object's this-scan-only share, see
    slam/utils.py's load_prior_scene_objects_as_seeds) to decide whether that object
    still needs an appearance check. Writes its diagnostic fields into `record`
    (n_visible_obj_points, projection ratios, footprint pixel counts, overlap_*) --
    pass a scratch dict for a call whose diagnostics shouldn't clobber the primary
    call's fields in the caller's real record.

    Returns (strong_overlap, weak_overlap, used_projection).
    '''
    det_pcd = det['pcd']
    overlap_d2o = _directional_overlap(det_pcd, target_pcd, params, src_subset=det_eroded_idx)[0]
    record["overlap_det_to_obj"] = overlap_d2o
    record["overlap_obj_to_det"] = _directional_overlap(target_pcd, det_pcd, params, dst_subset=det_eroded_idx)[0]
    strong_overlap = overlap_d2o
    weak_overlap = None
    used_projection = False

    visible = visibility.mask_for(target_pcd) if visibility is not None else (
        _visible_point_mask(_points(target_pcd), view, params) if view is not None else None)
    if visible is not None:
        record["n_visible_obj_points"] = int(visible.sum())
    has_view = visible is not None and record["n_visible_obj_points"] >= params.min_visible_points

    if has_view and params.association_gate_mode == ASSOCIATION_MODE_PROJECTION:
        proj_strong, proj_weak, _, _ = _association_by_projection(det, target_pcd, params, view, visibility, visible, record)
        if proj_strong is not None:
            strong_overlap = proj_strong
            weak_overlap = proj_weak
            used_projection = True
    elif has_view:
        weak_overlap = _directional_overlap(
            target_pcd, det_pcd, params, src_subset=np.nonzero(visible)[0], dst_subset=det_eroded_idx)[0]
        record["overlap_visible_obj_to_det"] = weak_overlap

    return strong_overlap, weak_overlap, used_projection


def _clip_dino_similarity(det, obj, clip_weight: float = 0.5, dino_weight: float = 0.5) -> float:
    '''
    How alike a detection and an object look, via the same weighted-dot-product formula
    convert_concept_graphs_to_scene_diff_benchmark_data.py's semantic_similarity() uses
    for the offline before/after comparison (clip_ft_mean/dino_ft_mean are running means
    of L2-normalized per-frame features, never renormalized, so their dot product IS the
    average cosine similarity over every frame pair -- see that function's docstring).
    Cast to float32 here since det's and obj's features aren't guaranteed to share dtype
    (e.g. an fp16 detection feature against a seed's carried-over fp32 mean).
    '''
    clip_sim = float(torch.dot(det['clip_ft_mean'].float(), obj['clip_ft_mean'].float()))
    dino_sim = float(torch.dot(det['dino_ft_mean'].float(), obj['dino_ft_mean'].float()))
    return clip_weight * clip_sim + dino_weight * dino_sim


def evaluate_detection_gates(det, obj, params: GeometryFusionParams, view: FrameView = None,
                             visibility: "_VisibilityCache" = None, det_eroded_idx=None) -> dict:
    """
    Run the point-overlap gate for one (detection, existing object) pair that already
    passed the bbox gate. Returns the debug record; `merged` is filled in by the caller
    once the fusion decision is actually carried out.

    The gate takes the better of two directions -- see the module docstring for why
    neither suffices alone:

      strong  "how much of this detection does that object explain?"
      weak    "how much of that object does this detection cover?"

    Under ASSOCIATION_MODE_PROJECTION (the default) both come from one screen-space
    intersection of the detection's 2D mask with the object's projected footprint,
    differing only in the denominator. Under ASSOCIATION_MODE_POINT_3D, or whenever
    projection can't be formed (no `view`, or too few of the object's points fall in
    this frame's frustum to mean anything), both are 3D nearest-neighbour point ratios
    (_directional_overlap) instead -- see the module docstring for the risk that mode
    carries. For that 3D fallback, det_pcd's points are first restricted to
    _eroded_mask_subset(det, ...) -- this frame's own detection mask, eroded inward by
    tau -- so points nearest the detection's own boundary don't enter the search. This
    depends only on `det`/`view`/`params`, not `obj`, so a caller evaluating the same
    detection against many candidate objects should compute it once and pass it in as
    `det_eroded_idx` rather than paying for cv2.erode + reprojection again per
    candidate; left as None here only for standalone/direct callers.

    `gate_class` on a passing record says which direction earned the pass; the caller uses
    it to apply "merge every strong match, but only the closest weak one".

    For a seed-descended `obj` (has a 'confirmed_pcd' key -- see slam/utils.py's
    load_prior_scene_objects_as_seeds), passing the above is not enough on its own. A
    second spatial check runs against confirmed_pcd alone (this scan's own re-confirmed
    share, excluding whatever geometry the object still carries straight from the
    "before" scan): if THAT independently passes the same gate, the object has already
    earned enough of this scan's own evidence and the match proceeds exactly as above.
    If not -- confirmed_pcd is still empty or too sparse, or it just doesn't spatially
    match this detection -- a CLIP+DINO similarity check (_clip_dino_similarity) against
    params.seed_semantic_sim_threshold decides instead, since spatial proximity to a
    stale seed footprint alone can't tell "the same object, still there" from "something
    new placed flush against it" (see the module docstring).
    """
    record = {
        "bbox_pass": True,
        "overlap_det_to_obj": 0.0,
        "overlap_obj_to_det": 0.0,
        "overlap_visible_obj_to_det": None,
        "strong_projection_ratio": None,
        "weak_projection_ratio": None,
        "n_visible_obj_points": 0,
        "n_footprint_pixels": 0,
        "n_det_mask_pixels": 0,
        "projection_close_radius": 0,
        "strong_overlap": 0.0,
        "weak_overlap": None,
        "primary_overlap": 0.0,
        "point_gate_direction": None,
        "point_gate_pass": False,
        "confirmed_primary_overlap": None,
        "confirmed_gate_pass": None,
        "appearance_gate_required": False,
        "appearance_gate_pass": None,
        "semantic_similarity": None,
        "merged": False,
        "reason": REASON_POINT_OVERLAP_REJECT,
    }

    det_pcd, obj_pcd = det['pcd'], obj['pcd']
    if len(det_pcd.points) < params.min_points_for_gates or len(obj_pcd.points) < params.min_points_for_gates:
        record["reason"] = REASON_TOO_FEW_POINTS
        return record

    if det_eroded_idx is None:
        det_eroded_idx = _eroded_mask_subset(det, det_pcd, view, params)

    strong_overlap, weak_overlap, used_projection = _association_overlap(
        det, det_eroded_idx, obj_pcd, params, view, visibility, record)
    strong_direction, weak_direction = (
        ("projection_det_to_obj", "projection_obj_to_det") if used_projection
        else ("det_to_obj", "visible_obj_to_det"))

    record["strong_overlap"] = strong_overlap
    record["weak_overlap"] = weak_overlap

    if weak_overlap is not None and weak_overlap > strong_overlap:
        record["primary_overlap"] = weak_overlap
        record["point_gate_direction"] = weak_direction
    else:
        record["primary_overlap"] = strong_overlap
        record["point_gate_direction"] = strong_direction

    if record["primary_overlap"] < params.point_overlap_thresh:
        return record
    record["point_gate_pass"] = True

    confirmed_pcd = obj.get('confirmed_pcd')
    if confirmed_pcd is not None:
        confirmed_primary = 0.0
        if len(confirmed_pcd.points) >= params.min_points_for_gates:
            c_strong, c_weak, _ = _association_overlap(det, det_eroded_idx, confirmed_pcd, params, view, visibility, {})
            confirmed_primary = c_weak if (c_weak is not None and c_weak > c_strong) else c_strong
        record["confirmed_primary_overlap"] = confirmed_primary
        record["confirmed_gate_pass"] = confirmed_primary >= params.point_overlap_thresh

        if not record["confirmed_gate_pass"]:
            record["appearance_gate_required"] = True
            similarity = _clip_dino_similarity(det, obj)
            record["semantic_similarity"] = similarity
            record["appearance_gate_pass"] = similarity >= params.seed_semantic_sim_threshold
            if not record["appearance_gate_pass"]:
                record["reason"] = REASON_APPEARANCE_REJECT
                return record

    record["reason"] = REASON_MERGE
    # A strong claim stands on its own -- the detection is substantially this object, so
    # several of them at once means the node was split and should be rejoined. A weak
    # claim only says the detection encloses the node, which a host mask does for every
    # object lying on it, so the caller keeps just the closest one.
    record["gate_class"] = "strong" if strong_overlap >= params.point_overlap_thresh else "weak"
    return record


def evaluate_object_pair_gates(obj_a, obj_b, params: GeometryFusionParams) -> dict:
    '''
    Same cascade for an (object, object) pair in the final consolidation, except the
    primary criterion is max(overlap_A->B, overlap_B->A): both sides can be partial
    views here, so requiring one specific direction would leave real splits unmerged.

    Unlike evaluate_detection_gates, this ALSO runs a surface-normal-consistency gate
    after the point-overlap one. Online per-frame fusion dropped that gate (see the
    module docstring): it rejected genuine matches between different faces of the same
    large object as it grew face by face within one continuous, already-anchored scan.
    That reasoning doesn't transfer here -- this is a one-time, post-scan sweep over
    otherwise-unrelated surviving objects, and max(A->B, B->A) alone is specifically
    vulnerable to a small object entirely resting against a larger, differently-angled
    surface (its whole point cloud can land within tau of the big object's surface,
    scoring 1.0 in one direction, even though the two surfaces face very different
    ways) -- normal consistency is the one signal here that can still catch that,
    despite the known blind spot for two flush, SAME-facing surfaces (abs() treats
    opposite-facing normals as agreeing, since estimate_normals leaves them unoriented).
    '''
    record = {
        "bbox_pass": True,
        "overlap_a_to_b": 0.0,
        "overlap_b_to_a": 0.0,
        "primary_overlap": 0.0,
        "normal_consistency_ratio": 0.0,
        "point_gate_pass": False,
        "normal_gate_pass": False,
        "merged": False,
        "reason": REASON_POINT_OVERLAP_REJECT,
    }

    pcd_a, pcd_b = obj_a['pcd'], obj_b['pcd']
    if len(pcd_a.points) < params.min_points_for_gates or len(pcd_b.points) < params.min_points_for_gates:
        record["reason"] = REASON_TOO_FEW_POINTS
        return record

    overlap_a2b, a_idx, b_idx = _directional_overlap(pcd_a, pcd_b, params)
    overlap_b2a, b_idx_r, a_idx_r = _directional_overlap(pcd_b, pcd_a, params)
    record["overlap_a_to_b"] = overlap_a2b
    record["overlap_b_to_a"] = overlap_b2a
    record["primary_overlap"] = max(overlap_a2b, overlap_b2a)

    if record["primary_overlap"] < params.point_overlap_thresh:
        return record
    record["point_gate_pass"] = True

    if overlap_a2b >= overlap_b2a:
        ratio = _normal_consistency(pcd_a, pcd_b, a_idx, b_idx, params)
    else:
        ratio = _normal_consistency(pcd_b, pcd_a, b_idx_r, a_idx_r, params)
    record["normal_consistency_ratio"] = ratio
    if ratio < params.normal_consistency_thresh:
        record["reason"] = REASON_NORMAL_REJECT
        return record

    record["normal_gate_pass"] = True
    record["reason"] = REASON_MERGE
    return record


def _merge_into(obj1, obj2, params: GeometryFusionParams, run_dbscan: bool):
    '''
    merge_obj2_into_obj1 (which already handles pcd union + reprocess, n_points, and
    bbox recomputation) plus normal invalidation, since obj1's geometry just changed --
    needed even for an online merge that itself never reads normals, since this same
    object's current pcd may still be compared in evaluate_object_pair_gates later.
    '''
    merged = merge_obj2_into_obj1(
        obj1=obj1,
        obj2=obj2,
        downsample_voxel_size=params.downsample_voxel_size,
        dbscan_remove_noise=params.dbscan_remove_noise,
        dbscan_eps=params.dbscan_eps,
        dbscan_min_points=params.dbscan_min_points,
        spatial_sim_type=params.spatial_sim_type,
        device=params.device,
        run_dbscan=run_dbscan,
    )
    _invalidate_normals(merged['pcd'])
    return merged


# ---------------------------------------------------------------- online fusion

def fuse_detections_geometry_only(
    detection_list: DetectionList,
    objects: MapObjectList,
    params: GeometryFusionParams,
    frame_idx: int,
    view: FrameView = None,
    debug: FusionDebugWriter = None,
) -> MapObjectList:
    '''
    Geometry-only replacement for
    compute_spatial_similarities -> compute_visual_similarities -> aggregate_similarities
    -> match_detections_to_objects -> merge_obj_matches.

    Detections are handled one at a time against the live object list, so a merge
    performed for detection k is already visible to detection k+1. A detection that
    matches several nodes strongly fuses with all of them at once, collapsing them into a
    single object; among nodes it only matched weakly it takes at most the closest one.
    '''
    obj_list = list(objects)
    visibility = _VisibilityCache(view, params)

    for det_idx, det in enumerate(detection_list):
        # The index this detection's mask carries in the frame's mask array -- the same
        # "#N" drawn on the detected/final mask overlays, so a debug row names a mask
        # that can be found by eye.
        mask_idx = det['mask_idx'][0] if det.get('mask_idx') else None
        los, his = _stack_aabbs(obj_list)
        lo_d, hi_d = _aabb(det['pcd'])
        bbox_pass = _bbox_gate_vector(lo_d, hi_d, los, his, params.bbox_margin)

        # Depends only on this detection's own mask/depth, not on any candidate object,
        # so it's computed once per detection rather than once per (detection, candidate)
        # pair -- see evaluate_detection_gates' docstring.
        det_eroded_idx = _eroded_mask_subset(det, det['pcd'], view, params)

        records = []
        strong_ids, weak_ids = [], []
        for obj_idx in np.nonzero(bbox_pass)[0]:
            record = evaluate_detection_gates(
                det, obj_list[obj_idx], params, view, visibility, det_eroded_idx=det_eroded_idx)
            # obj_idx is this object's position in the live list, which merges and
            # deletions keep shifting -- useless for following one node across frames.
            # curr_obj_num is stable and is what fused_masks/ prints on its badges, so a
            # log row names a specific object in a specific overlay image.
            record["obj_num"] = obj_list[obj_idx].get('curr_obj_num')
            records.append((int(obj_idx), record))
            # point_gate_pass alone isn't enough any more: a seed-descended candidate
            # (see evaluate_detection_gates) can pass the spatial gate and still get
            # rejected by the appearance gate that follows it, in which case gate_class
            # is never set. Only a record that made it all the way through has one.
            if "gate_class" not in record:
                continue
            (strong_ids if record["gate_class"] == "strong" else weak_ids).append(int(obj_idx))

        # Every strong match merges: each one says the detection is substantially that
        # object, so finding several means an earlier frame split one object into
        # several nodes and this observation bridges them back.
        #
        # At most ONE weak match merges, the closest. A weak match only says the
        # detection's silhouette encloses the node, which a host detection does for
        # every object resting on it -- merging all of them would collapse a sofa and
        # everything on it into one node.
        by_overlap = {i: r["primary_overlap"] for i, r in records}
        weak_keep = max(weak_ids, key=lambda i: by_overlap[i]) if weak_ids else None
        merging = strong_ids + ([weak_keep] if weak_keep is not None else [])
        weak_dropped = set(weak_ids) - {weak_keep}

        if merging:
            anchor = merging[0]
            obj_list[anchor] = _merge_into(obj_list[anchor], det, params, run_dbscan=False)
            for other in merging[1:]:
                obj_list[anchor] = _merge_into(obj_list[anchor], obj_list[other], params, run_dbscan=False)
            for other in sorted(merging[1:], reverse=True):
                del obj_list[other]
            action = "fused"
        else:
            obj_list.append(det)
            action = "new_object"

        if debug is not None and debug.enabled:
            merged_ids = set(merging)
            for obj_idx, record in records:
                record["merged"] = obj_idx in merged_ids
                if obj_idx in weak_dropped:
                    # Distinguish "lost the weak contest" from "never passed the gates".
                    record["reason"] = REASON_WEAK_NOT_CLOSEST
                debug.log("online_fusion", {
                    "frame_idx": frame_idx,
                    "det_idx": det_idx,
                    "mask_idx": mask_idx,
                    "obj_idx": obj_idx,
                    **record,
                })
            if merging:
                summary_reason = REASON_MERGE
            elif records:
                # why the most promising candidate still failed
                summary_reason = max(records, key=lambda r: r[1]["primary_overlap"])[1]["reason"]
            else:
                summary_reason = REASON_BBOX_REJECT
            debug.log("online_fusion", {
                "frame_idx": frame_idx,
                "det_idx": det_idx,
                "mask_idx": mask_idx,
                "summary": True,
                "n_objects": int(len(bbox_pass)),
                "n_bbox_pass": int(bbox_pass.sum()),
                # n_gate_pass counts everything that cleared both gates; n_merged is what
                # the weak-closest-only rule actually let through.
                "n_gate_pass": len(strong_ids) + len(weak_ids),
                "n_merged": len(merging),
                "n_strong": len(strong_ids),
                "n_weak": len(weak_ids),
                "n_weak_dropped": len(weak_dropped),
                "n_bbox_reject": int(len(bbox_pass) - bbox_pass.sum()),
                "action": action,
                "reason": summary_reason,
            })

    return MapObjectList(obj_list)


# ---------------------------------------------------------------- global consolidation

def global_geometry_consolidation(
    objects: MapObjectList,
    params: GeometryFusionParams,
    debug: FusionDebugWriter = None,
) -> MapObjectList:
    '''
    One geometry-only object-object consolidation pass over the surviving objects
    after the last frame, repeated until a full sweep produces no merge.

    Unlike merge_overlap_objects(), which computes one overlap matrix up front and
    then keeps consuming it even for objects whose geometry a merge has since changed,
    every sweep recomputes bbox candidates and overlaps from scratch, and any object
    touched by a merge is retired from the current sweep (reason="deferred_to_next_sweep")
    so it is only re-examined against fresh geometry.
    '''
    obj_list = list(objects)
    if len(obj_list) < 2:
        return MapObjectList(obj_list)

    print(f"Before global geometry consolidation: {len(obj_list)}")

    for sweep in range(params.max_sweeps):
        los, his = _stack_aabbs(obj_list)

        # Fresh candidates for this sweep's geometry -- never a carried-over matrix.
        candidates = []
        for i in range(len(obj_list) - 1):
            bbox_pass = _bbox_gate_vector(los[i], his[i], los[i + 1:], his[i + 1:], params.bbox_margin)
            for offset in np.nonzero(bbox_pass)[0]:
                candidates.append((i, i + 1 + int(offset)))

        evaluated = []
        for (i, j) in candidates:
            record = evaluate_object_pair_gates(obj_list[i], obj_list[j], params)
            evaluated.append((i, j, record))

        evaluated.sort(key=lambda t: t[2]["primary_overlap"], reverse=True)

        frozen = set()
        dead = set()
        merged_any = False
        for (i, j, record) in evaluated:
            # point_gate_pass alone isn't enough: evaluate_object_pair_gates can pass
            # the spatial gate and still fail the normal-consistency one that follows
            # it, in which case normal_gate_pass stays False.
            if record["normal_gate_pass"]:
                if i in frozen or j in frozen:
                    # One side's geometry already changed this sweep; re-evaluate it
                    # next sweep instead of trusting this now-stale measurement.
                    record["reason"] = REASON_DEFERRED
                else:
                    obj_list[i] = _merge_into(obj_list[i], obj_list[j], params, run_dbscan=True)
                    record["merged"] = True
                    frozen.update((i, j))
                    dead.add(j)
                    merged_any = True

            if debug is not None and debug.enabled:
                debug.log("global_consolidation", {"sweep": sweep, "obj_a": i, "obj_b": j, **record})

        if dead:
            obj_list = [o for k, o in enumerate(obj_list) if k not in dead]

        print(f"  consolidation sweep {sweep}: {len(candidates)} candidates, "
              f"{len(dead)} merged, {len(obj_list)} objects left")

        if not merged_any:
            break
    else:
        print(f"  consolidation hit max_sweeps={params.max_sweeps} without converging")

    print(f"After global geometry consolidation: {len(obj_list)}")
    return MapObjectList(obj_list)


# ---------------------------------------------------------------- recognition confidence

def count_visible_frames(object_points, dataset, params: GeometryFusionParams, desc=None):
    """
    Walk a whole sequence and ask, per object, from which frames its geometry should
    have been visible -- frustum plus the z-buffer test against each frame's own depth
    map, the same visibility the association gates use, gated on min_visible_points.

    Returns (n_visible, visible_frames): a count per object and the matching set of
    frame indices, so a caller can intersect those frames with something else.

    `object_points` is a list of (N, 3) arrays rather than objects, because the
    before/after benchmark comparison runs this with one scan's objects against the
    OTHER scan's dataset -- there is no "this object's own frames" relationship there,
    only geometry and a camera trajectory.
    """
    n_visible = [0] * len(object_points)
    visible_frames = [set() for _ in object_points]
    if not object_points:
        return n_visible, visible_frames

    frames = trange(len(dataset), desc=desc) if desc else range(len(dataset))
    for frame_idx in frames:
        _, depth_tensor, intrinsics, *_ = dataset[frame_idx]
        depth_array = depth_tensor[..., 0].cpu().numpy()
        pose = dataset.poses[frame_idx].cpu().numpy()
        view = FrameView.from_frame(
            cam_to_world=pose, cam_K=intrinsics.cpu().numpy()[:3, :3], depth_array=depth_array)

        for i, pts in enumerate(object_points):
            if len(pts) == 0:
                continue
            visible = _visible_point_mask(pts, view, params)
            if int(visible.sum()) < params.min_visible_points:
                continue
            n_visible[i] += 1
            visible_frames[i].add(frame_idx)

    return n_visible, visible_frames


def compute_recognition_confidence(objects, dataset, params: GeometryFusionParams,
                                   debug: FusionDebugWriter = None) -> None:
    '''
    For each object, using its FINAL merged geometry: over every frame in the sequence,
    was this object in a position this camera should have seen it (frustum + z-buffer
    against that frame's own depth map, the same visibility test the association gates
    use), and if so, did a detection actually get merged into this object from that
    frame? confidence = recognized / should-have-been-visible.

    "Recognized" is deliberately loose -- a single point of a merged detection's mask
    landing on this object in a frame counts, matching how the gates themselves decide
    to merge on partial evidence. This metric is not about how COMPLETE any one
    detection was; it is about which of the angles this object geometrically exposed
    the pipeline actually got a look from, versus which are backed by nothing but the
    3D merge that stitched this node together. A pillow assembled from three viewpoints
    out of fifteen it was visible from is a real object with weak evidence, and that
    gap is exactly what downstream (e.g. deciding whether an apparent removal in the
    "after" scan is real or just never-observed) needs to know.

    Runs once, after global consolidation, so every object's geometry is final --
    running it earlier would score partially-merged fragments against a visibility test
    their own future growth hasn't happened yet, moving the ground under the metric.

    `image_idx` already carries exactly what "recognized in frame f" means: it is
    extended (never overwritten) by merge_obj2_into_obj1 for every detection folded
    into this object, whether that happened during online fusion or a later
    consolidation merge, so it already reflects the object's full merge history.

    Mutates each object in place with `recognition_confidence` (float in [0, 1], or
    None if the object was never geometrically visible from any frame -- which should
    not happen for anything that was actually detected, but is not fabricated into a
    number if it does) plus the raw counts behind it, `recognition_n_visible` and
    `recognition_n_recognized`.
    '''
    if len(objects) == 0:
        return

    recognized_frames = [set(o['image_idx']) for o in objects]
    n_visible, visible_frames = count_visible_frames(
        [np.asarray(o['pcd'].points) for o in objects], dataset, params,
        desc="Recognition confidence")
    n_recognized = [len(visible_frames[i] & recognized_frames[i]) for i in range(len(objects))]

    for i, obj in enumerate(objects):
        vis, rec = n_visible[i], n_recognized[i]
        obj['recognition_n_visible'] = vis
        obj['recognition_n_recognized'] = rec
        obj['recognition_confidence'] = (rec / vis) if vis > 0 else None
        if debug is not None and debug.enabled:
            debug.log("recognition_confidence", {
                "obj_num": obj.get('curr_obj_num'),
                "class_name": obj.get('class_name'),
                "n_frames_visible": vis,
                "n_frames_recognized": rec,
                "confidence": obj['recognition_confidence'],
            })


def annotate_recognition_trust(objects, params: GeometryFusionParams,
                               debug: FusionDebugWriter = None) -> None:
    """
    Marks each object `recognition_trusted` -- recognized in at least
    `recognition_min_recognized_frames` frames AND with recognition_confidence at or
    above `recognition_min_confidence`.

    Nothing is removed. Weakly-evidenced objects stay in the graph, in obj_json, and in
    the scene-graph render (drawn with a pale border there); the flag is what the
    before/after benchmark comparison uses to decide which objects are solid enough to
    *assert* a change from. An object too weak to claim "this was removed" is still
    perfectly good as something the other scan's objects can match against, which is
    why dropping it here would lose information the comparison needs.

    Must run after compute_recognition_confidence() has populated the fields this reads.
    If that never ran (cfg.compute_recognition_confidence is False), the fields are
    absent and everything is trusted -- no metric was computed, so there is no basis to
    doubt anything.

    confidence=None (never geometrically visible from any frame -- see
    compute_recognition_confidence) fails the frame-count test anyway, since zero
    visible frames implies zero recognized ones.
    """
    n_trusted = 0
    for obj in objects:
        n_rec = obj.get('recognition_n_recognized')
        if n_rec is None:
            obj['recognition_trusted'] = True
            n_trusted += 1
            continue
        conf = obj.get('recognition_confidence')
        trusted = (n_rec >= params.recognition_min_recognized_frames
                   and conf is not None
                   and conf >= params.recognition_min_confidence)
        obj['recognition_trusted'] = trusted
        n_trusted += int(trusted)
        if not trusted and debug is not None and debug.enabled:
            debug.log("recognition_trust", {
                "obj_num": obj.get('curr_obj_num'),
                "class_name": obj.get('class_name'),
                "n_recognized": n_rec,
                "confidence": conf,
                "failed_frame_count": n_rec < params.recognition_min_recognized_frames,
                "failed_confidence": conf is None or conf < params.recognition_min_confidence,
            })

    print(f"Recognition trust: {n_trusted}/{len(objects)} objects trusted "
          f"(min_recognized_frames={params.recognition_min_recognized_frames}, "
          f"min_confidence={params.recognition_min_confidence}); none removed")
