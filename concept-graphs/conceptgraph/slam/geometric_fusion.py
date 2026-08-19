'''
Geometry-only same-scan object fusion (cfg.object_fusion_mode == "geometry_only").

Replaces the appearance-influenced association path in rerun_realtime_mapping.py
(spatial_sim + CLIP visual_sim -> sim_sum -> argmax -> merge_obj_matches, plus the
periodic CLIP-gated merge_objects()) with a pure-geometry cascade:

    New detection D
        -> 3D AABB(+margin) overlap gate          (candidate filtering only)
        -> association gate            max(strong, weak) >= thresh
        -> surface normal consistency gate         |n_D . n_O| over tau-inliers
        -> merge EVERY strong match, plus the single closest weak one

Both directions are measured in the camera, from ONE intersection of the detection's 2D
mask with the object's projected footprint -- they differ only in the denominator:

    strong = |mask(D) & footprint(O_vis)| / |mask(D)|            "how much of D is O?"
    weak   = |mask(D) & footprint(O_vis)| / |footprint(O_vis)|   "how much of O is in D?"

footprint(O_vis) projects the object's points that this camera should have seen (frustum
+ z-buffer against the frame's own depth map) and closes the scatter into an area;
mask(D) needs no projection at all, being this frame's actual segmentation.

Neither direction suffices alone. strong fails whenever a detection reveals surface the
object has not accumulated yet (walking around a table), spawning a duplicate node per
viewpoint. weak fails whenever a 2D mask covers only part of the object -- which is why
the footprint is restricted to what was visible from here.

Both were once measured as 3D nearest-neighbour ratios, and both leaked, because "some
point of the other cloud lies within tau" is far weaker evidence than pixel containment:

  * weak leaked catastrophically. For an object resting on a surface, its rim and the
    host surface immediately beside it both sit inside tau, so a host detection scored
    0.3-0.5 against every small object on it -- and 0.90 once the frame border cropped a
    handbag down to its contact band, which destroyed that node. On screen the same
    comparisons give 0.00-0.01 while the bag's own mask still scores 0.89-0.99.
  * strong leaked more mildly, in the same direction. A large node's cloud is dense
    enough that a small detection clipped by the frame border almost always finds
    something within tau: 20 such merges scored D->O = 1.000 in 3D, and re-scoring a
    sample of 15 on screen dropped two of them below the threshold outright and most of
    the rest well below 1.0.

Sideways leakage is impossible on screen -- a point a centimetre to the side lands on a
different pixel, which belongs to a different mask.

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
from scipy.spatial import cKDTree
from tqdm import trange

from conceptgraph.slam.slam_classes import DetectionList, MapObjectList
from conceptgraph.slam.utils import from_intrinsics_matrix, get_bounding_box, merge_obj2_into_obj1

GEOMETRY_ONLY_MODE = "geometry_only"

# How BOTH association directions are measured (see the module docstring).
ASSOCIATION_MODE_PROJECTION = "projection"  # 2D: one screen-space intersection, two denominators
ASSOCIATION_MODE_POINT_3D = "point_3d"      # 3D: legacy nearest-neighbour ratios

# reason codes written into the debug jsonl
REASON_TOO_FEW_POINTS = "too_few_points"
REASON_BBOX_REJECT = "bbox_reject"
REASON_POINT_OVERLAP_REJECT = "point_overlap_reject"
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
    normal_radius: float                # downsample_voxel_size * normal_radius_factor
    normal_max_nn: int
    normal_cos_thresh: float            # cos(normal_angle_thresh_deg)
    normal_consistency_thresh: float
    min_points_for_gates: int
    visibility_depth_tolerance: float   # z-buffer slack [m] for the occlusion test
    min_visible_points: int             # below this, the containment direction carries no evidence
    association_gate_mode: str          # ASSOCIATION_MODE_PROJECTION | ASSOCIATION_MODE_POINT_3D
    projection_close_factor: float      # closing radius = factor * fx * voxel / z  [px]
    max_sweeps: int
    # final-graph filter, applied by filter_by_recognition_confidence() after
    # compute_recognition_confidence() has populated the fields it reads
    min_recognized_frames: int          # drop objects recognized in this many frames or fewer
    min_recognition_confidence: float   # drop objects whose confidence falls below this
    # merge_obj2_into_obj1 passthrough
    downsample_voxel_size: float
    dbscan_remove_noise: bool
    dbscan_eps: float
    dbscan_min_points: int
    spatial_sim_type: str
    device: str


def geometry_fusion_params_from_cfg(cfg) -> GeometryFusionParams:
    voxel = float(cfg['downsample_voxel_size'])
    return GeometryFusionParams(
        bbox_margin=float(cfg['fusion_bbox_margin']),
        point_distance_thresh=voxel * float(cfg['fusion_point_distance_factor']),
        point_overlap_thresh=float(cfg['fusion_point_overlap_thresh']),
        normal_radius=voxel * float(cfg['fusion_normal_radius_factor']),
        normal_max_nn=int(cfg['fusion_normal_max_nn']),
        normal_cos_thresh=math.cos(math.radians(float(cfg['fusion_normal_angle_thresh_deg']))),
        normal_consistency_thresh=float(cfg['fusion_normal_consistency_thresh']),
        min_points_for_gates=int(cfg['fusion_min_points_for_gates']),
        visibility_depth_tolerance=float(cfg['fusion_visibility_depth_tolerance']),
        min_visible_points=int(cfg['fusion_min_visible_points']),
        association_gate_mode=str(cfg['fusion_association_gate_mode']),
        projection_close_factor=float(cfg['fusion_projection_close_factor']),
        max_sweeps=int(cfg['fusion_global_consolidation_max_sweeps']),
        min_recognized_frames=int(cfg['fusion_min_recognized_frames']),
        min_recognition_confidence=float(cfg['fusion_min_recognition_confidence']),
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


def _aabb(pcd):
    '''Axis-aligned (lo, hi) straight off the point cloud, each shape (3,).'''
    pts = _points(pcd)
    if len(pts) == 0:
        nan = np.full(3, np.nan)
        return nan, nan
    return pts.min(axis=0), pts.max(axis=0)


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
    Lazily estimated per-point normals, cached on the point cloud itself.

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
    '''Force recomputation on next use (called right after a merge changes geometry).'''
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


def _directional_overlap(pcd_src, pcd_dst, params: GeometryFusionParams, src_subset=None):
    '''
    Overlap(src -> dst) = #{p in src : min_{q in dst} ||p - q|| < tau} / |src|,
    along with the nearest-neighbor correspondence that produced it.

    `src_subset`, when given, restricts the source points (and hence the denominator)
    to those indices -- that's how the visibility-normalized O_vis->D direction is
    computed. Returned indices are always in the FULL source cloud's index space, so
    the normal gate can index _normals(pcd_src) with them directly.

    Returns (overlap, src_inlier_idx, dst_nn_idx) -- the two index arrays are the
    tau-inlier correspondence pairs, reused by the normal gate so no second NN search
    is needed.
    '''
    src_all = _points(pcd_src)
    src_pts = src_all if src_subset is None else src_all[src_subset]
    dst_pts = _points(pcd_dst)
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return 0.0, np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    dists, nn_idx = cKDTree(dst_pts).query(src_pts)
    local_inlier_idx = np.nonzero(dists < params.point_distance_thresh)[0]
    src_inlier_idx = local_inlier_idx if src_subset is None else np.asarray(src_subset)[local_inlier_idx]
    return float(len(local_inlier_idx)) / len(src_pts), src_inlier_idx, nn_idx[local_inlier_idx]


def _normal_consistency(pcd_src, pcd_dst, src_idx, dst_idx, params: GeometryFusionParams):
    '''
    Fraction of the tau-inlier correspondences whose surface normals agree:
    |n_src . n_dst| >= cos(angle_thresh). Absolute value because estimate_normals
    leaves normals unoriented (a surface seen from opposite sides flips sign).

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


# ---------------------------------------------------------------- gate evaluation

class _VisibilityCache:
    '''
    Per-frame memo of "which points of this object should this camera have seen" and
    "where does it land on screen". Both depend only on the object's geometry and the
    camera, so they are the same for every detection in the frame; without this they
    would be recomputed once per (detection, candidate) pair.

    Keyed by id(pcd) while holding a reference to that pcd, so the id cannot be
    recycled onto a different cloud. A merge replaces obj['pcd'] with the new cloud
    process_pcd() returns, so merged geometry simply misses the cache.
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
    is None when the projection could not be formed.

    mask(D) is this frame's actual SAM mask, so the strong direction needs no reprojection
    of the detection at all -- it is already the ground truth for "where D is on screen".
    det['mask'] is a list because merging concatenates it; entry 0 is this frame's mask,
    the only one that shares a camera with `view`. It comes from gobs after resize_gobs,
    so it already matches the depth map's dimensions.

    Normals still need 3D correspondence pairs and a projection produces none, so each
    direction gets its own, restricted to exactly the points its ratio counted:
      * weak   -- O's visible points whose projection landed inside mask(D)
      * strong -- D's own points whose projection landed inside footprint(O)
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

    # Correspondences for whichever direction ends up winning. Closing can carry a
    # footprint across a boundary without any point landing there, so an empty subset is
    # possible; it means no normal evidence, and the normal gate then fails -- the right
    # call for a match nothing actually touches.
    empty = np.zeros(0, dtype=int)
    obj_inside = np.nonzero(visible)[0][det_mask[v, u]]
    if len(obj_inside):
        _, s_idx, d_idx = _directional_overlap(obj_pcd, det_pcd, params, src_subset=obj_inside)
        weak_args = (obj_pcd, det_pcd, s_idx, d_idx)
    else:
        weak_args = (obj_pcd, det_pcd, empty, empty)

    # D belongs to this frame, so every one of its points projects into it; the z-buffer
    # pass still runs so a point the depth map contradicts is not counted.
    det_pts = _points(det_pcd)
    det_visible = (visibility.mask_for(det_pcd) if visibility is not None
                   else _visible_point_mask(det_pts, view, params))
    du, dv = (visibility.pixels_for(det_pcd) if visibility is not None
              else _project_pixels(det_pts, det_visible, view))
    det_inside = np.nonzero(det_visible)[0][footprint[dv, du]] if len(du) else empty
    if len(det_inside):
        _, s_idx, d_idx = _directional_overlap(det_pcd, obj_pcd, params, src_subset=det_inside)
        strong_args = (det_pcd, obj_pcd, s_idx, d_idx)
    else:
        strong_args = (det_pcd, obj_pcd, empty, empty)

    return strong_ratio, weak_ratio, strong_args, weak_args


def evaluate_detection_gates(det, obj, params: GeometryFusionParams, view: FrameView = None,
                             visibility: "_VisibilityCache" = None) -> dict:
    """
    Run the point-overlap and normal gates for one (detection, existing object) pair
    that already passed the bbox gate. Returns the debug record; `merged` is filled in
    by the caller once the fusion decision is actually carried out.

    The gate takes the better of two directions -- see the module docstring for why
    neither suffices alone:

      strong  "how much of this detection does that object explain?"
      weak    "how much of that object does this detection cover?"

    Under ASSOCIATION_MODE_PROJECTION both come from one screen-space intersection of the
    detection's 2D mask with the object's projected footprint, differing only in the
    denominator. Under ASSOCIATION_MODE_POINT_3D both are nearest-neighbour point ratios
    in 3D, which is what this used to do throughout.

    `gate_class` on a passing record says which direction earned the pass; the caller uses
    it to apply "merge every strong match, but only the closest weak one".

    With no `view`, or too few visible points to mean anything, the projection route is
    unavailable and the 3D D->O ratio alone decides.
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
        "normal_consistency_ratio": 0.0,
        "point_gate_pass": False,
        "normal_gate_pass": False,
        "merged": False,
        "reason": REASON_POINT_OVERLAP_REJECT,
    }

    det_pcd, obj_pcd = det['pcd'], obj['pcd']
    if len(det_pcd.points) < params.min_points_for_gates or len(obj_pcd.points) < params.min_points_for_gates:
        record["reason"] = REASON_TOO_FEW_POINTS
        return record

    # The 3D ratios are always computed: they are the point_3d mode's answer and, under
    # projection, the diagnostic that lets one log show both ways of scoring the pair.
    overlap_d2o, d_idx, o_nn_idx = _directional_overlap(det_pcd, obj_pcd, params)
    record["overlap_det_to_obj"] = overlap_d2o
    record["overlap_obj_to_det"] = _directional_overlap(obj_pcd, det_pcd, params)[0]

    visible = visibility.mask_for(obj_pcd) if visibility is not None else (
        _visible_point_mask(_points(obj_pcd), view, params) if view is not None else None)
    if visible is not None:
        record["n_visible_obj_points"] = int(visible.sum())
    has_view = visible is not None and record["n_visible_obj_points"] >= params.min_visible_points

    # 3D fallback, used verbatim by point_3d mode and whenever no usable view exists.
    strong_overlap, strong_args = overlap_d2o, (det_pcd, obj_pcd, d_idx, o_nn_idx)
    weak_overlap, weak_args = None, None
    strong_direction, weak_direction = "det_to_obj", "visible_obj_to_det"

    if has_view and params.association_gate_mode == ASSOCIATION_MODE_PROJECTION:
        proj = _association_by_projection(det, obj_pcd, params, view, visibility, visible, record)
        if proj[0] is not None:
            strong_overlap, weak_overlap, strong_args, weak_args = proj
            strong_direction, weak_direction = "projection_det_to_obj", "projection_obj_to_det"
    elif has_view:
        weak_overlap, vis_idx, vis_nn_idx = _directional_overlap(
            obj_pcd, det_pcd, params, src_subset=np.nonzero(visible)[0])
        record["overlap_visible_obj_to_det"] = weak_overlap
        weak_args = (obj_pcd, det_pcd, vis_idx, vis_nn_idx)

    record["strong_overlap"] = strong_overlap
    record["weak_overlap"] = weak_overlap

    if weak_overlap is not None and weak_overlap > strong_overlap:
        record["primary_overlap"] = weak_overlap
        record["point_gate_direction"] = weak_direction
        normal_args = weak_args
    else:
        record["primary_overlap"] = strong_overlap
        record["point_gate_direction"] = strong_direction
        normal_args = strong_args

    if record["primary_overlap"] < params.point_overlap_thresh:
        return record
    record["point_gate_pass"] = True

    # Normals are checked on the correspondence of whichever direction won the gate.
    ratio = _normal_consistency(*normal_args, params)
    record["normal_consistency_ratio"] = ratio
    if ratio < params.normal_consistency_thresh:
        record["reason"] = REASON_NORMAL_REJECT
        return record

    record["normal_gate_pass"] = True
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
    The normal gate reuses the correspondence of whichever direction won.
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
    bbox recomputation) plus normal invalidation, since obj1's geometry just changed.
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

        records = []
        strong_ids, weak_ids = [], []
        for obj_idx in np.nonzero(bbox_pass)[0]:
            record = evaluate_detection_gates(det, obj_list[obj_idx], params, view, visibility)
            # obj_idx is this object's position in the live list, which merges and
            # deletions keep shifting -- useless for following one node across frames.
            # curr_obj_num is stable and is what fused_masks/ prints on its badges, so a
            # log row names a specific object in a specific overlay image.
            record["obj_num"] = obj_list[obj_idx].get('curr_obj_num')
            records.append((int(obj_idx), record))
            if not record["normal_gate_pass"]:
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

    obj_points = [np.asarray(o['pcd'].points) for o in objects]
    recognized_frames = [set(o['image_idx']) for o in objects]
    n_visible = [0] * len(objects)
    n_recognized = [0] * len(objects)

    for frame_idx in trange(len(dataset), desc="Recognition confidence"):
        _, depth_tensor, intrinsics, *_ = dataset[frame_idx]
        depth_array = depth_tensor[..., 0].cpu().numpy()
        pose = dataset.poses[frame_idx].cpu().numpy()
        view = FrameView.from_frame(
            cam_to_world=pose, cam_K=intrinsics.cpu().numpy()[:3, :3], depth_array=depth_array)

        for i, pts in enumerate(obj_points):
            if len(pts) == 0:
                continue
            visible = _visible_point_mask(pts, view, params)
            if int(visible.sum()) < params.min_visible_points:
                continue
            n_visible[i] += 1
            if frame_idx in recognized_frames[i]:
                n_recognized[i] += 1

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


def filter_by_recognition_confidence(objects, params: GeometryFusionParams,
                                     debug: FusionDebugWriter = None) -> MapObjectList:
    '''
    Drops objects from the final graph whose recognition evidence is too thin to trust:
    recognized in `min_recognized_frames` frames or fewer -- by default 1, meaning a
    single detection that only geometric merging (never a second independent look)
    stitched into whatever it is now -- or with recognition_confidence below
    `min_recognition_confidence`, meaning most of the angles that should have shown it
    never did.

    Must run after compute_recognition_confidence() has populated the fields this reads.
    If it never ran (cfg.compute_recognition_confidence is False), those fields are
    absent and every object is kept unfiltered -- this function is a no-op rather than
    an error, since filtering on a metric that was never computed isn't meaningful.

    confidence=None (never geometrically visible from any frame -- see
    compute_recognition_confidence) is not evaluated by the confidence check; it is
    already caught by the recognized-frames check, since zero visible frames implies
    zero recognized ones.
    '''
    kept, dropped = [], []
    for obj in objects:
        n_rec = obj.get('recognition_n_recognized')
        if n_rec is None:
            kept.append(obj)
            continue
        conf = obj.get('recognition_confidence')
        few_frames = n_rec <= params.min_recognized_frames
        low_confidence = conf is not None and conf < params.min_recognition_confidence
        if few_frames or low_confidence:
            dropped.append((obj, few_frames, low_confidence))
        else:
            kept.append(obj)

    if dropped and debug is not None and debug.enabled:
        for obj, few_frames, low_confidence in dropped:
            debug.log("recognition_filter", {
                "obj_num": obj.get('curr_obj_num'),
                "class_name": obj.get('class_name'),
                "n_recognized": obj.get('recognition_n_recognized'),
                "confidence": obj.get('recognition_confidence'),
                "dropped_few_frames": few_frames,
                "dropped_low_confidence": low_confidence,
            })

    print(f"Recognition-confidence filter: kept {len(kept)}/{len(objects)} objects "
          f"(min_recognized_frames={params.min_recognized_frames}, "
          f"min_recognition_confidence={params.min_recognition_confidence})")
    return MapObjectList(kept)
