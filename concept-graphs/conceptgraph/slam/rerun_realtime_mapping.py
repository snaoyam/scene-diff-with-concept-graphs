'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import os
import copy
import logging
import uuid
from pathlib import Path
import pickle
import gzip

# Third-party imports
import cv2
import numpy as np
import torch
from tqdm import trange
import hydra
from omegaconf import DictConfig, OmegaConf
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from collections import Counter

# Local application/library specific imports
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import consolidate_captions, get_openai_client
from conceptgraph.utils.ious import mask_subtract_contained, remove_small_mask_components
from conceptgraph.utils.general_utils import (
    DETECTIONS_EXP_SUFFIX,
    EXP_SUFFIX,
    ObjectClasses,
    discover_scene_vocabulary,
    get_det_out_path,
    get_discovered_classes_path,
    get_exp_out_path,
    get_vlm_captions,
    load_saved_detections,
    measure_time,
    save_detection_results,
    save_edge_json,
    save_hydra_config,
    save_obj_json,
    save_objects_for_frame,
    save_pointcloud,
    should_exit_early,
)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    vis_result_fast_on_depth,
    vis_result_for_vlm,
    vis_result_fast,
    vis_numbered_masks,
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    build_final_object_graph,
    dedup_gobs_by_mask_iou,
    filter_gobs,
    slice_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects,
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_pcd,
    processing_needed,
    resize_gobs
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)
from conceptgraph.slam.geometric_fusion import (
    GEOMETRY_ONLY_MODE,
    FrameView,
    FusionDebugWriter,
    fuse_detections_geometry_only,
    geometry_fusion_params_from_cfg,
    global_geometry_consolidation,
    projected_footprint,
)
from conceptgraph.utils.model_utils import compute_clip_features_batched, compute_dinov3_dense_features, pool_dinov3_features_by_mask
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections
from conceptgraph.utils.scenegraph_viz import render_frame_scenegraph
from conceptgraph.utils.visualize_full_scenegraph import load_scene_graph, render_full_scenegraph

VERSION_TEXT = "fix mask merge 7 - weak association 2d projection"

# Disable torch gradient computation
torch.set_grad_enabled(False)

# A logger for this file
@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="rerun_realtime_mapping")
# @profile
def main(cfg : DictConfig):
    print(f"===== version {VERSION_TEXT} =====")
    # OmegaConf.set_struct(cfg, False)
    # cfg.image_height = 512
    # cfg.image_width = 512

    # hydra.verbose bumps every logger (including PIL's) to DEBUG, which floods
    # the log with harmless internal messages (e.g. "Error closing: Operation on
    # closed image" from imageio/Pillow double-closing PNG file handles). Keep
    # PIL quiet without touching the global verbose setting.
    logging.getLogger("PIL").setLevel(logging.INFO)

    # Build the two ConceptGraphs (before/after) for this SceneDiff pair in one
    # run. Each variant gets its own deep copy of cfg so nothing one variant
    # mutates (process_cfg bakes dataset_config/dataset_root into concrete
    # Path objects) leaks into the other. Detection models are the one thing
    # intentionally shared across variants -- they're stateless inference
    # weights, not scene data.
    shared_models = None
    for variant in ("before", "after"):
        variant_cfg = copy.deepcopy(cfg)
        variant_cfg.scene_variant = variant
        shared_models = run_mapping_for_scene(variant_cfg, shared_models)
        # Release cached-but-unused allocator blocks (e.g. from the "segment
        # everything" vocabulary discovery pass) back to the driver before the
        # next variant starts.
        torch.cuda.empty_cache()


def run_mapping_for_scene(cfg: DictConfig, shared_models=None):
    tracker = MappingTracker()
    tracker.reset()
    DenoisingTracker().reset()

    # Output paths nest as outputs/<scene_pair>/concept_graphs/<scene_variant>/... so
    # that convert_concept_graphs_to_scene_diff_benchmark_data.py/run_scene_diff_benchmark.py's
    # benchmark_data/benchmark_result siblings land under the same outputs/<scene_pair>/
    # folder -- distinct from cfg.scene_id (used everywhere else below for dataset
    # loading, wandb naming, etc), which has no "concept_graphs" segment.
    concept_graphs_scene_id = f"{cfg.scene_pair}/concept_graphs/{cfg.scene_variant}"

    exp_out_path = get_exp_out_path(cfg.output_root, concept_graphs_scene_id, EXP_SUFFIX, exps_dir_name=cfg.exps_dir_name)
    exp_out_path.mkdir(exist_ok=True, parents=True)

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs",
            #    entity="concept-graphs",
                name=cfg.scene_id.replace("/", "_"),
                config=cfg_to_dict(cfg),
               )
    cfg = process_cfg(cfg)

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    # cam_K = dataset.get_cam_K()

    # Full camera trajectory, used by detect_up_vector() to figure out which
    # side of the up-axis is "down" (see slam/utils.py). Available up front --
    # dataset.poses is pre-loaded for the whole sequence, not per-frame.
    camera_positions = dataset.poses[:, :3, 3].cpu().numpy()

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # output folder for this mapping experiment
    exp_out_path = get_exp_out_path(cfg.output_root, concept_graphs_scene_id, EXP_SUFFIX, exps_dir_name=cfg.exps_dir_name)

    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.output_root, concept_graphs_scene_id, DETECTIONS_EXP_SUFFIX, make_dir=False, exps_dir_name=cfg.exps_dir_name)

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)

    # Per-scene-variant vocabulary, discovered (not a fixed classes file -- see
    # discover_scene_vocabulary) once per detections_exp_suffix and cached under
    # det_exp_path for reuse. Built below, either freshly (run_detections=True)
    # or loaded from a prior run's cached file (run_detections=False).
    discovered_classes_path = get_discovered_classes_path(det_exp_path)
    # get_openai_client() just constructs a client object, no network I/O, so it's
    # safe to build unconditionally -- scene vocabulary discovery needs it whenever
    # run_detections is True, regardless of cfg.make_captions (which only gates the
    # later per-object caption VLM usage).
    openai_client = get_openai_client()

    # if we need to do detections
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)

    if run_detections:
        print("\n".join([f"Running detections...version: {VERSION_TEXT}"] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models, reusing them across scene variants if already loaded
        if shared_models is None:
            detection_model = measure_time(YOLO)('yolov8l-world.pt')
            # Move to cfg.device before the first set_classes() call. YOLO-World's
            # internal CLIP text encoder is built and cached lazily on whatever
            # device the model is on at that moment, but the cached wrapper's
            # tokenize() keeps using that snapshot device forever afterward (it
            # doesn't track later .to() moves). If set_classes() runs first on
            # cpu and predict() later moves the model to cuda, a second
            # set_classes() call (e.g. for the "after" variant sharing this
            # model) mismatches: cached cuda weights vs cpu tokens.
            detection_model.to(cfg.device)
            sam_predictor = SAM('sam_l.pt') # SAM('mobile_sam.pt') # UltraLytics SAM
            # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", "laion2b_s32b_b79k", precision="fp16"
            )
            clip_model = clip_model.to(cfg.device)
            clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
            # fp16 (.half()) was tried first to match CLIP's precision, but DINOv3
            # ViT-H+/16 produces all-NaN output under a blind .half() cast (attention
            # softmax/LayerNorm overflow, since fp16's exponent range tops out at
            # ~65504) -- confirmed by testing fp32 vs fp16 on real crops from this
            # pipeline. bfloat16 has the same exponent range as fp32 (just fewer
            # mantissa bits), so it avoids that overflow while still halving memory
            # vs fp32.
            dinov3_model = torch.hub.load(
                cfg.dinov3_repo_dir, cfg.dinov3_model_name, source='local', weights=cfg.dinov3_checkpoint_path
            ).to(cfg.device).to(torch.bfloat16).eval()
            shared_models = (detection_model, sam_predictor, clip_model, clip_preprocess, clip_tokenizer, dinov3_model)
        else:
            detection_model, sam_predictor, clip_model, clip_preprocess, clip_tokenizer, dinov3_model = shared_models

        # Discover this scene variant's object vocabulary (representative frames +
        # whole-frame VLM object listing + SAM "segment everything" per-segment VLM
        # naming, merged) instead of using a fixed classes file -- see
        # discover_scene_vocabulary(). Reuses the sam_predictor/openai_client already
        # set up above; no extra model loading.
        discovered_classes_path = discover_scene_vocabulary(
            dataset=dataset,
            sam_predictor=sam_predictor,
            openai_client=openai_client,
            det_exp_path=det_exp_path,
            detections_exp_suffix=DETECTIONS_EXP_SUFFIX,
            voxel_size=cfg.discovery_voxel_size,
            pixel_stride=cfg.discovery_pixel_stride,
            max_representative_frames=cfg.discovery_max_representative_frames,
            sam_conf=cfg.discovery_sam_conf,
            sam_min_segment_area_px=cfg.discovery_sam_min_segment_area_px,
            sam_max_segment_area_ratio=cfg.discovery_sam_max_segment_area_ratio,
            sam_max_segments_per_frame=cfg.discovery_sam_max_segments_per_frame,
            device=cfg.device,
        )
        # discover_scene_vocabulary's SAM "segment everything" pass can reserve
        # large allocator blocks (many masks/frame); release what's unused before
        # the per-frame main loop starts.
        torch.cuda.empty_cache()
        obj_classes = ObjectClasses(
            classes_file_path=discovered_classes_path,
            bg_classes=detections_exp_cfg['bg_classes'],
            skip_bg=detections_exp_cfg['skip_bg'],
        )

        # Set the classes for the detection model. cache_clip_model=False: the cached
        # self.clip_model (default cache_clip_model=True) is a real submodule of
        # detection_model.model, so later predict(..., quantize=16) calls -- which run
        # model.half() on the whole model to enable fp16 inference -- flip its LayerNorm
        # to fp16 too, breaking CLIP's fp16/fp32 mixed-precision design. The "before"
        # variant's frame loop calls predict() many times before the "after" variant
        # reuses this shared detection_model and calls set_classes() again, so the second
        # call would hit the now-corrupted cached clip_model and crash with
        # "RuntimeError: expected scalar type Float but found Half". Rebuilding a fresh,
        # uncached CLIP text encoder here (cheap: once per variant) avoids that.
        # detection_model.set_classes() (the YOLO wrapper) doesn't forward
        # cache_clip_model -- only the underlying WorldModel.set_classes() accepts it --
        # so call that directly and replicate the wrapper's own post-processing
        # (background-token removal, names bookkeeping) ourselves.
        detection_classes = obj_classes.get_classes_arr()
        detection_model.model.set_classes(detection_classes, cache_clip_model=False)
        if " " in detection_classes:
            detection_classes.remove(" ")
        detection_model.model.names = detection_classes
        if detection_model.predictor:
            detection_model.predictor.model.names = detection_classes
    else:
        if not discovered_classes_path.exists():
            raise FileNotFoundError(
                f"No cached discovered vocabulary at {discovered_classes_path}. Run with "
                f"force_detection=True for detections_exp_suffix='{DETECTIONS_EXP_SUFFIX}' "
                f"at least once before reusing cached detections."
            )
        obj_classes = ObjectClasses(
            classes_file_path=discovered_classes_path,
            bg_classes=detections_exp_cfg['bg_classes'],
            skip_bg=detections_exp_cfg['skip_bg'],
        )
        print("\n".join([f"NOT Running detections... version: {VERSION_TEXT}"] * 10))

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{DETECTIONS_EXP_SUFFIX}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    scenegraph_color_cache = {}
    scenegraph_viz_out_path = exp_out_path / "scenegraph_viz"

    # Three per-frame mask overlays, one per pipeline stage, all numbered so a specific
    # object can be named by eye when reporting a detection problem. The names are the
    # stage order on purpose -- an earlier "final_masks" was read as the finished result
    # when it is in fact the 2D input to 3D fusion. Comparing frames of it is misleading:
    # one object routinely gets several boxes in a frame ("couch" and "sofa" on the same
    # sofa), and how many survive varies frame to frame even though 3D fusion puts them
    # all on one node regardless.
    #   detected_masks/ -- straight out of YOLO+SAM, before anything is dropped, so a
    #                      small object the detector did find but that fell under the
    #                      confidence/area thresholds is still visible here.
    #   filtered_masks/ -- what reached mapping: after filter_gobs and
    #                      mask_subtract_contained. Numbered by mask_idx, the index the
    #                      fusion debug log records.
    #   fused_masks/    -- the live 3D map as of THIS frame: written inside the frame
    #                      loop, right after this frame's matching/merging (and any
    #                      periodic denoise/filter/merge) has been applied, using
    #                      whatever `objects` looks like at that point -- not the final
    #                      post-loop map. So watching the frames in order shows objects
    #                      progressively merging (two masks/numbers becoming one) instead
    #                      of only ever showing the finished result. Masks that already
    #                      fused into one node this frame are unioned into one mask. An
    #                      object with no fresh 2D detection this frame (nothing in
    #                      obj['image_idx'] matches frame_idx -- e.g. the detector missed
    #                      it) is still drawn if it would be visible from this frame's
    #                      camera: its stored world-frame point cloud is reprojected via
    #                      geometric_fusion.projected_footprint() (in-frustum, unoccluded
    #                      per the frame's own depth map, then morphologically closed into
    #                      a filled footprint) and drawn with its class name suffixed
    #                      "-r" so it reads as a geometric guess, not a real
    #                      detection. Objects with too few visible points (below
    #                      fusion_min_visible_points) or nothing in frustum are skipped
    #                      that frame, same as before. Numbered by curr_obj_num (stable
    #                      once an object is created, even across merges), but colors are
    #                      positional per frame like detected_masks/filtered_masks -- not
    #                      fixed across frames.
    # detected/filtered separate "never detected" from "detected, then filtered out";
    # filtered/fused separate "still a separate detection" from "still a separate object".
    # All three are written on cached-detection runs too: the saved .pkl.gz holds the
    # raw detections as-is (nothing rewrites them before saving), so detected_masks/ can
    # be drawn from it just as well as from a fresh detection pass.
    detected_masks_out_path = exp_out_path / "detected_masks"
    filtered_masks_out_path = exp_out_path / "filtered_masks"
    fused_masks_out_path = exp_out_path / "fused_masks"
    detected_masks_out_path.mkdir(parents=True, exist_ok=True)
    filtered_masks_out_path.mkdir(parents=True, exist_ok=True)
    if cfg.save_fused_masks:
        fused_masks_out_path.mkdir(parents=True, exist_ok=True)

    # Resolved once per scene (see slam/geometric_fusion.py). Used unconditionally --
    # not only by geometry-only fusion's own per-frame gating, but also by
    # write_progressive_fused_mask() below, which reprojects objects with no fresh
    # detection this frame regardless of which object_fusion_mode is active. In
    # "appearance" mode, CLIP/DINO features are still extracted and stored below, they
    # just never take part in the fusion decision.
    geometry_only_fusion = cfg.object_fusion_mode == GEOMETRY_ONLY_MODE
    geo_fusion_params = geometry_fusion_params_from_cfg(cfg)
    fusion_debug = FusionDebugWriter(
        exp_out_path / "fusion_debug", enabled=geometry_only_fusion and cfg.fusion_debug_log
    )

    # How much SAM speckle the cleanup took out this scan. detected_masks/ draws the
    # cleaned masks, so this summary is the only place the effect is visible.
    dedup_dropped = 0
    speckle_masks_touched = 0
    speckle_components_removed = 0
    speckle_pixels_removed = 0

    def write_progressive_fused_mask():
        # Snapshot of `objects` exactly as it stands right now -- this frame's own
        # matching/merging (and any periodic denoise/filter/merge) already applied,
        # but not later frames' -- so calling this once per frame_idx, in order,
        # renders the map's progressive merge history rather than only the finished
        # result. Reads frame_idx/image_rgb/color_path/objects/frame_view/
        # geo_fusion_params from the enclosing loop, so it must be called from inside
        # it.
        #
        # An object with no fresh detection this frame (local_indices empty -- the
        # detector missed it, or it just hasn't been reached yet) is not simply
        # skipped: its stored world-frame point cloud is reprojected into this frame
        # (same in-frustum/unoccluded test geometry-only fusion's own gate uses), so a
        # detector miss on an object that's still plainly in view doesn't make it
        # silently vanish from this debug view. Suffixed "-r" in the label so
        # it's visually distinguishable from a mask backed by an actual detection.
        frame_objects = []
        for obj in objects:
            local_indices = [i for i, idx in enumerate(obj['image_idx']) if idx == frame_idx]
            if local_indices:
                # A merge can fold more than one raw detection from this frame into
                # the same 3D object -- union their masks instead of picking just the
                # first.
                combined_mask = np.logical_or.reduce([obj['mask'][i] for i in local_indices])
                frame_objects.append({
                    'obj_num': obj['curr_obj_num'],
                    'class_name': obj['class_name'],
                    'mask': combined_mask,
                })
                continue

            points = np.asarray(obj['pcd'].points)
            footprint, visible, _ = projected_footprint(points, frame_view, geo_fusion_params)
            if footprint is None or int(visible.sum()) < geo_fusion_params.min_visible_points:
                continue  # out of frustum, occluded, or too few surviving points to mean anything
            frame_objects.append({
                'obj_num': obj['curr_obj_num'],
                'class_name': obj['class_name'] + "-r",
                'mask': footprint,
            })
        vis_numbered_masks(
            image_rgb,
            np.stack([o['mask'] for o in frame_objects]) if frame_objects
            else np.zeros((0, *image_rgb.shape[:2]), dtype=bool),
            [o['class_name'] for o in frame_objects],
            (fused_masks_out_path / color_path.name).with_suffix(".jpg"),
            ids=[o['obj_num'] for o in frame_objects],
        )

    exit_early_flag = False
    counter = 0
    for frame_idx in trange(len(dataset)):
        tracker.curr_frame_idx = frame_idx
        counter+=1

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print(f"Exit early signal detected. Skipping to the final frame... version: {VERSION_TEXT}")
            exit_early_flag = True

        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # Read info about current frame from dataset
        # color image
        color_path = Path(dataset.color_paths[frame_idx])
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")

        if run_detections:
            results = None
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path)) # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Do initial object detection
            results = detection_model.predict(color_path, conf=0.1, verbose=False, quantize=16)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            if xyxy_tensor.numel() != 0:
                sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False, quantize=16)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()
            else:
                masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)

            # Drop SAM's boundary speckle before anything downstream sees these masks, so
            # the CLIP crops, the pooled DINO features, the saved detections and the 3D
            # backprojection all work off the cleaned version -- and a cached-detection
            # rerun inherits it, since what gets pickled below is already clean.
            masks_np, n_comp_removed, n_px_removed = remove_small_mask_components(
                masks_np, cfg.mask_min_component_ratio)
            speckle_masks_touched += int(n_comp_removed > 0)
            speckle_components_removed += n_comp_removed
            speckle_pixels_removed += n_px_removed

            # Create a detections object that we will save later
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )

            # One DINOv3 forward pass per frame; every object gets a dino_ft pooled
            # from it below. Stored for the later before/after graph matching, never
            # consulted when deciding what merges with what.
            dino_dense = compute_dinov3_dense_features(image_rgb, dinov3_model, cfg.device)

            # Captions still come from the VLM, gated by cfg.make_captions; relations
            # are derived from 3D geometry once, after the whole frame loop, from the
            # final point cloud (build_final_object_graph, called after "LOOP OVER" below).
            labels, captions = get_vlm_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, cfg.make_captions, openai_client)

            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

            dino_feats = pool_dinov3_features_by_mask(dino_dense, curr_det.mask, cfg.dino_mask_erosion_px)

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection 
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "dino_feats": dino_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "captions": captions,
            }

            raw_gobs = results

            # save the detections if needed
            if cfg.save_detections:

                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
        else:
            # Support current and old saving formats
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                # if no detections, throw an error
                raise FileNotFoundError(f"No detections found for frame {frame_idx}at paths \n{det_exp_pkl_path / color_path.stem} or \n{det_exp_pkl_path / f'{int(color_path.stem):06}'}.")

        # Raw detections, before filter_gobs drops anything. Drawn from raw_gobs rather
        # than inside the detection branch above so a cached-detection run gets this view
        # too. Numbering is the raw detection index, independent of filtered_masks',
        # since filtering renumbers whatever it keeps.
        vis_numbered_masks(
            image_rgb,
            raw_gobs['mask'],
            [f"{raw_gobs['classes'][cid]} {conf:0.2f}"
             for cid, conf in zip(raw_gobs['class_id'], raw_gobs['confidence'])],
            (detected_masks_out_path / color_path.name).with_suffix(".jpg"),
        )

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose

        # Everything needed to ask "would this frame's camera have seen a given world
        # point" -- built once per frame, unconditionally (a 4x4 inverse plus an
        # intrinsics unpack, trivial next to the detection/CLIP/DINO work already done
        # per frame). Reused both by write_progressive_fused_mask() (falling back to
        # geometric reprojection when an object has no fresh detection this frame) and,
        # further down, as the view= kwarg into fuse_detections_geometry_only() in
        # geometry-only fusion mode.
        frame_view = FrameView.from_frame(
            cam_to_world=adjusted_pose,
            cam_K=intrinsics.cpu().numpy()[:3, :3],
            depth_array=depth_array,
        )

        # resize the observation if needed
        gobs = resize_gobs(raw_gobs, image_rgb)

        # One object routinely picks up several near-identical masks under different
        # labels ("sofa" and "couch" on the same sofa) -- drop all but the most confident
        # before anything else looks at them.
        gobs, n_dup = dedup_gobs_by_mask_iou(gobs, cfg.dedup_mask_iou_thresh)
        dedup_dropped += n_dup

        # Separate the masks BEFORE filtering, not after, and let low-confidence
        # detections take part. Confidence answers two different questions here -- "is
        # this its own object?" and "is something there at all?" -- and running the
        # subtraction after filter_gobs silently conflated them: a blanket detected at
        # 0.178 was dropped, so nothing carved it out of the sofa mask, and the sofa
        # detection carried the blanket's pixels into 3D. Measured on one scan, dropped
        # detections covered 13-53% of the surviving sofa mask. Now a weak detection
        # still claims its pixels without being promoted to an object; filter_gobs below
        # decides only the latter.
        if cfg.mask_subtract_min_conf > 0:
            gobs = slice_gobs(gobs, [i for i, c in enumerate(gobs['confidence'])
                                     if c >= cfg.mask_subtract_min_conf])
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        # Now decide what becomes an object.
        gobs = filter_gobs(gobs, image_rgb,
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        # What actually reached mapping. The "#N" prefix is mask_idx, the same index
        # the fusion debug log records, so a row in analyze_fusion_debug's output points
        # at exactly one numbered mask in this image.
        vis_numbered_masks(
            image_rgb,
            gobs['mask'],
            [f"{gobs['classes'][gobs['class_id'][i]]} {gobs['confidence'][i]:0.2f}"
             for i in range(len(gobs['mask']))],
            (filtered_masks_out_path / color_path.name).with_suffix(".jpg"),
        )

        if len(gobs['mask']) == 0: # no detections in this frame
            if cfg.save_fused_masks:
                write_progressive_fused_mask()
            continue

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs['mask'],
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )

        if len(detection_list) == 0: # no detections, skip
            if cfg.save_fused_masks:
                write_progressive_fused_mask()
            continue

        if geometry_only_fusion:
            # Geometry-only association: bbox -> directional point overlap -> normal
            # consistency, fusing with every existing node that passes. No empty-map
            # special case is needed -- with no objects yet, every detection simply
            # falls through to "new object".
            objects = fuse_detections_geometry_only(
                detection_list=detection_list,
                objects=objects,
                params=geo_fusion_params,
                frame_idx=frame_idx,
                # Lets the point gate ask what this camera should have seen of each
                # existing object, so a large or mostly-unobserved object isn't
                # penalised by the parts no viewpoint could have covered.
                view=frame_view,
                debug=fusion_debug,
            )
            # These helpers return a new MapObjectList rather than mutating in place,
            # so map_edges' reference has to follow it (build_final_object_graph
            # indexes map_edges.objects when adding edges).
            map_edges.update_objects_list(objects)
        else:
            # if no objects yet in the map,
            # just add all the objects from the current frame
            # then continue, no need to match or merge
            if len(objects) == 0:
                objects.extend(detection_list)
                tracker.increment_total_objects(len(detection_list))
                owandb.log({
                        "total_objects_so_far": tracker.get_total_objects(),
                        "objects_this_frame": len(detection_list),
                    })
                if cfg.save_fused_masks:
                    write_progressive_fused_mask()
                continue

            ### compute similarities and then merge
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=cfg['spatial_sim_type'], 
                detection_list=detection_list, 
                objects=objects,
                downsample_voxel_size=cfg['downsample_voxel_size']
            )

            visual_sim = compute_visual_similarities(detection_list, objects)

            agg_sim = aggregate_similarities(
                match_method=cfg['match_method'], 
                phys_bias=cfg['phys_bias'], 
                spatial_sim=spatial_sim, 
                visual_sim=visual_sim
            )

            # Perform matching of detections to existing objects
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim, 
                detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
            )

            # Now merge the detected objects into the existing objects based on the match indices
            objects = merge_obj_matches(
                detection_list=detection_list, 
                objects=objects, 
                match_indices=match_indices,
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device']
                # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
            )
        # fix the class names for objects
        # they should be the most popular name, not the first name
        for idx, obj in enumerate(objects):
            temp_class_name = obj["class_name"]
            curr_obj_class_id_counter = Counter(obj['class_id'])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        is_final_frame = frame_idx == len(dataset) - 1
        if is_final_frame:
            print(f"Final frame detected. Performing final post-processing... version: {VERSION_TEXT}")

        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )

        # Filtering
        if processing_needed(
            cfg["filter_interval"],
            cfg["run_filter_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects,
                map_edges=map_edges
            )

        # Merging. Skipped entirely in geometry-only mode: online fusion already
        # merges across multiple nodes per detection, and a single global
        # consolidation runs after the loop instead of this periodic pass.
        if not geometry_only_fusion and processing_needed(
            cfg["merge_interval"],
            cfg["run_merge_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            if cfg["make_edges"]:
                objects, map_edges = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=True,
                    map_edges=map_edges
                )
            else:
                objects = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=False,
                    map_edges=None
                )

        if cfg.save_fused_masks:
            write_progressive_fused_mask()

        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                cfg.obj_min_detections,
                adjusted_pose,
                color_path
            )
        
        if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
            # save the pointcloud
            save_pointcloud(
                exp_suffix=EXP_SUFFIX,
                exp_out_path=exp_out_path,
                cfg=cfg,
                objects=objects,
                obj_classes=obj_classes,
            )

        owandb.log({
            "frame_idx": frame_idx,
            "counter": counter,
            "exit_early_flag": exit_early_flag,
            "is_final_frame": is_final_frame,
        })

        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log({
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
                })
    # LOOP OVER -----------------------------------------------------

    print(f"Near-identical mask dedup (dedup_mask_iou_thresh={cfg.dedup_mask_iou_thresh}): "
          f"dropped {dedup_dropped} duplicate detections")

    if run_detections:
        print(f"SAM speckle cleanup (mask_min_component_ratio="
              f"{cfg.mask_min_component_ratio}): removed {speckle_components_removed} "
              f"components / {speckle_pixels_removed} px from {speckle_masks_touched} masks")

    # Geometry-only mode does no periodic merging during the scan, so object-object
    # merging happens here, once, over the surviving objects -- repeated until a full
    # sweep finds no mergeable pair, recomputing candidates/overlaps whenever a merge
    # changes an object's geometry.
    if geometry_only_fusion:
        objects = measure_time(global_geometry_consolidation)(
            objects=objects,
            params=geo_fusion_params,
            debug=fusion_debug,
        )
        map_edges.update_objects_list(objects)
    fusion_debug.close()

    # Build the whole scene's object graph once, from the final (fully
    # merged/denoised) point clouds -- see build_final_object_graph() docstring.
    up_axis, up_direction = None, None
    if cfg.make_edges:
        map_edges, up_axis, up_direction = build_final_object_graph(
            objects, camera_positions, map_edges, frame_idx=len(dataset) - 1
        )

    # Consolidate captions only when captions are enabled
    if cfg.make_captions:
        for object in objects:
            obj_captions = object['captions'][:20]
            consolidated_caption = consolidate_captions(openai_client, obj_captions)
            object['consolidated_caption'] = consolidated_caption

    # Per-frame scene graph overlay of the FINAL, fully merged/filtered map (not the
    # intermediate per-frame state), so every frame's overlay reflects the same
    # completed concept graph. fused_masks/ is unrelated to this pass -- it's written
    # progressively inside the main frame loop above (see write_progressive_fused_mask)
    # so it shows the map merging over time instead of only the finished result.
    if cfg.save_scenegraph_viz:
        print(f"Rendering final per-frame scene graph visualizations... version: {VERSION_TEXT}")
        for frame_idx in trange(len(dataset)):
            color_path = Path(dataset.color_paths[frame_idx])
            color_tensor, *_ = dataset[frame_idx]
            image_rgb = color_tensor.cpu().numpy().astype(np.uint8)

            frame_obj_indices = set()
            frame_objects = []
            for obj_idx, obj in enumerate(objects):
                local_indices = [i for i, idx in enumerate(obj['image_idx']) if idx == frame_idx]
                if not local_indices:
                    continue
                frame_obj_indices.add(obj_idx)
                # A merge_overlap_objects pass (or a same-frame double match) can fold more
                # than one raw detection from this frame into the same 3D object -- union
                # their masks/boxes instead of picking just the first one, so the larger of
                # the two isn't silently dropped from this frame's viz.
                combined_mask = np.logical_or.reduce([obj['mask'][i] for i in local_indices])
                xyxy_stack = np.stack([obj['xyxy'][i] for i in local_indices]).astype(float)
                combined_xyxy = np.concatenate([xyxy_stack[:, :2].min(axis=0), xyxy_stack[:, 2:].max(axis=0)])
                frame_objects.append({
                    'obj_num': obj['curr_obj_num'],
                    'class_name': obj['class_name'],
                    'caption': obj['captions'][local_indices[0]].get('caption', '') if obj['captions'] else '',
                    'mask': combined_mask,
                    'xyxy': combined_xyxy,
                })

            if not frame_objects:
                continue

            frame_edges = [
                (objects[obj1_idx]['curr_obj_num'], edge.rel_type, objects[obj2_idx]['curr_obj_num'])
                for (obj1_idx, obj2_idx), edge in map_edges.edges_by_index.items()
                if obj1_idx in frame_obj_indices and obj2_idx in frame_obj_indices
            ]

            render_frame_scenegraph(
                image_rgb,
                frame_objects,
                frame_edges,
                scenegraph_viz_out_path / f"{color_path.stem}_viz.png",
                scenegraph_color_cache,
            )

    # Save the pointcloud
    save_pointcloud(
        exp_suffix=EXP_SUFFIX,
        exp_out_path=exp_out_path,
        cfg=cfg,
        objects=objects,
        obj_classes=obj_classes,
        edges=map_edges,
        up_axis=up_axis,
        up_direction=up_direction,
    )

    save_obj_json(
        exp_suffix=EXP_SUFFIX,
        exp_out_path=exp_out_path,
        objects=objects
    )

    save_edge_json(
        exp_suffix=EXP_SUFFIX,
        exp_out_path=exp_out_path,
        objects=objects,
        edges=map_edges
    )

    if cfg.save_scenegraph_full:
        full_scenegraph = load_scene_graph(
            exp_out_path / f"obj_json_{EXP_SUFFIX}.json",
            exp_out_path / f"edge_json_{EXP_SUFFIX}.json",
        )
        full_scenegraph_path = render_full_scenegraph(
            full_scenegraph,
            exp_out_path / "scenegraph_full.png",
            title=f"{cfg.scene_id} / {EXP_SUFFIX}",
        )
        print(f"Saved full scene graph to {full_scenegraph_path} version: {VERSION_TEXT}")

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
            }, f)

    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    owandb.finish()

    return shared_models

if __name__ == "__main__":
    main()
