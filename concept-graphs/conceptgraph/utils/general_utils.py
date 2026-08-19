import gzip
import json
import logging
import os
from pathlib import Path
import pickle
# from conceptgraph.utils.vis import annotate_for_vlm, filter_detections, plot_edges_from_vlm
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import prepare_objects_save_vis, select_representative_frames
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.model_utils import crop_with_padding
import supervision as sv
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from conceptgraph.utils.vlm import (
    get_obj_captions_from_image_gpt4v,
    vlm_extract_object_captions,
    get_frame_object_list,
    get_segment_object_name,
)
import cv2
import re


from omegaconf import OmegaConf
import torch
import numpy as np
import time
from PIL import Image

# Fixed labels for pipeline output folders, i.e.
# <dataset_root>/<scene_id>/<exps_dir_name>/<EXP_SUFFIX|DETECTIONS_EXP_SUFFIX>/.
# Pinned here (previously configurable via rerun_realtime_mapping.yaml) since both
# rerun_realtime_mapping.py and convert_concept_graphs_to_scene_diff_benchmark_data.py
# need the same value and the project only ever uses one value for each.
EXP_SUFFIX = "r_mapping_pilot"  # helpful label to identify the mapping experiment
DETECTIONS_EXP_SUFFIX = "s_detections_pilot"  # helpful label to identify the detections experiment

def cfg_to_dict(input_cfg):
    """ Convert a Hydra configuration object to a native Python dictionary,
    ensuring all special types (e.g., ListConfig, DictConfig, PosixPath) are
    converted to serializable types for JSON. Checks for non-serializable objects. """
    
    def convert_to_serializable(obj):
        """ Recursively convert non-serializable objects to serializable types. """
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def check_serializability(obj, context=""):
        """ Attempt to serialize the object, raising an error if not possible. """
        try:
            json.dumps(obj)
        except TypeError as e:
            raise TypeError(f"Non-serializable object encountered in {context}: {e}")

        if isinstance(obj, dict):
            for k, v in obj.items():
                check_serializability(v, context=f"{context}.{k}" if context else str(k))
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                check_serializability(item, context=f"{context}[{idx}]")

    # Convert Hydra configs to native Python types
    # check if its already a dictionary, in which case we don't need to convert it
    if not isinstance(input_cfg, dict):
        native_cfg = OmegaConf.to_container(input_cfg, resolve=True)
    else:
        native_cfg = input_cfg
    # Convert all elements to serializable types
    serializable_cfg = convert_to_serializable(native_cfg)
    # Check for serializability of the entire config
    check_serializability(serializable_cfg)

    return serializable_cfg

def get_exp_out_path(dataset_root, scene_id, exp_suffix, make_dir=True, exps_dir_name="exps"):
    exp_out_path = Path(dataset_root) / scene_id / exps_dir_name / f"{exp_suffix}"
    if make_dir:
        exp_out_path.mkdir(exist_ok=True, parents=True)
    return exp_out_path

def get_vis_out_path(exp_out_path):
    vis_folder_path = exp_out_path / "vis"
    vis_folder_path.mkdir(exist_ok=True, parents=True)
    return vis_folder_path

def get_det_out_path(exp_out_path, make_dir=True):
    detections_folder_path = exp_out_path / "detections"
    if make_dir:
        detections_folder_path.mkdir(exist_ok=True, parents=True)
    return detections_folder_path

def check_run_detections(force_detection, det_exp_path):
    # first check if det_exp_path directory exists
    if force_detection:
        return True
    if not det_exp_path.exists():
        return True
    return False

def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def _class_id_to_color(class_id: int) -> tuple:
    """Deterministic per-class-id color (0-1 float RGB) for annotate_for_vlm's contour
    drawing -- no external storage needed, just a colormap indexed by class_id (same
    approach as scenegraph_viz.get_object_color, indexed by obj_num there instead)."""
    return plt.get_cmap("tab20")(class_id % 20)[:3]

def annotate_for_vlm(
    image: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
    save_path=None, 
    color: tuple=(0, 255, 0), 
    thickness: int=2, 
    text_color: tuple=(255, 255, 255), 
    text_scale: float=0.6, 
    text_thickness: int=2, 
    text_bg_color: tuple=(255, 255, 255), 
    text_bg_opacity: float=0.95,  # Opacity from 0 (transparent) to 1 (opaque)
    small_mask_threshold = 0.002,
    mask_opacity: float = 0.2  # Opacity for mask fill
) -> np.ndarray:
    annotated_image = image.copy()
    
    
    # if image.shape[0] > 700:
    #     print(f"Line 604, image.shape[0]: {image.shape[0]}")
    #     text_scale = 2.5
    #     text_thickness = 5
    total_pixels = image.shape[0] * image.shape[1]
    small_mask_size = total_pixels * small_mask_threshold
    
    detections_mask = detections.mask
    detections_mask = mask_subtract_contained(detections.xyxy, detections_mask)
    
    # Sort detections by mask area, large to small, and keep track of original indices
    mask_areas = [np.count_nonzero(mask) for mask in detections_mask]
    sorted_indices = sorted(range(len(mask_areas)), key=lambda x: mask_areas[x], reverse=True)
    
    # Iterate over each mask and corresponding label in the detections in sorted order
    for i in sorted_indices:
        mask = detections_mask[i]
        label = labels[i]
        label_num = label.split(" ")[-1]
        label_name = re.sub(r'\s*\d+$', '', label).strip()
        bbox = detections.xyxy[i]
        
        obj_color = _class_id_to_color(int(detections.class_id[i]))
        # multiply by 255 to convert to BGR
        obj_color = tuple([int(c * 255) for c in obj_color])
        
        # Add color over mask for this object 
        mask_uint8 = mask.astype(np.uint8)
        mask_color_image = np.zeros_like(annotated_image)
        mask_color_image[mask_uint8 > 0] = obj_color
        # cv2.addWeighted(annotated_image, 1, mask_color_image, mask_opacity, 0, annotated_image)

        # Draw contours
        contours, _ = cv2.findContours(mask_uint8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_image, contours, -1, obj_color, thickness)

        # Determine if the mask is considered "small"
        if mask_areas[i] < small_mask_size:
            x_center = int(bbox[2])  # Place the text to the right of the bounding box
            y_center = int(bbox[1])  # Place the text above the top of the bounding box
        else:
            # Calculate the centroid of the mask
            ys, xs = np.nonzero(mask)
            y_center, x_center = ndi.center_of_mass(mask)
            x_center, y_center = int(x_center), int(y_center)

        # Prepare text background
        text = label_num + ": " + label_name 
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        text_x_left = x_center - text_width // 2
        text_y_top = y_center + (text_height) // 2
        
        # Create a rectangle sub-image for the text background
        b_pad = 2 # background rectangle padding
        rect_top_left = (text_x_left - b_pad, text_y_top - text_height - baseline - b_pad)
        rect_bottom_right = (text_x_left + text_width + b_pad, text_y_top - baseline//2 + b_pad)
        sub_img = annotated_image[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
        
        # Create the background rectangle with the specified color and opacity
        # make the text bg color be the negative of the text color
        text_bg_color = tuple([255 - c for c in obj_color])
        # now make text bg color grayscale
        text_bg_color = tuple([int(sum(text_bg_color) / 3)] * 3)
        background_rect = np.full(sub_img.shape, text_bg_color, dtype=np.uint8)
        # cv2.addWeighted(sub_img, 1 - text_bg_opacity, background_rect, text_bg_opacity, 0, sub_img)

        # Draw text with background
        cv2.putText(
            annotated_image, 
            text, 
            (text_x_left, text_y_top - baseline), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_scale, 
            # obj_color,
            # (255,255,255),
            (0,0,0),
            text_thickness, 
            cv2.LINE_AA
        )
        
        # Draw text with background
        cv2.putText(
            annotated_image, 
            text, 
            (text_x_left, text_y_top - baseline), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_scale,
            # (0,0,0), 
            obj_color,
            text_thickness - 1, 
            cv2.LINE_AA
        )
        
        if save_path:
            cv2.imwrite(save_path, annotated_image)

    return annotated_image, sorted_indices

def filter_detections(
    image,
    detections: sv.Detections, 
    classes, 
    top_x_detections = None, 
    confidence_threshold: float = 0.0,
    given_labels = None,
    iou_threshold: float = 0.80,  # IoU similarity threshold
    proximity_threshold: float = 20.0,  # Default proximity threshold
    keep_larger: bool = True,  # Keep the larger bounding box by area if True, else keep the smaller
    min_mask_size_ratio=0.00025
) -> tuple[sv.Detections, list[str]]:
    '''
    Filter detections based on confidence, top X detections, and proximity of bounding boxes.
    Args:
        proximity_threshold (float): The minimum distance between centers of bounding boxes to consider them non-overlapping.
        keep_larger (bool): If True, keeps the larger bounding box when overlaps occur; otherwise keeps the smaller.
    Returns:
        tuple[sv.Detections, list[str]]: Filtered detections and labels.
    '''
    if not (hasattr(detections, 'confidence') and hasattr(detections, 'class_id') and hasattr(detections, 'xyxy')):
        print("Detections object is missing required attributes.")
        return detections, []

    # Sort by confidence initially
    detections_combined = sorted(
        zip(detections.confidence, detections.class_id, detections.xyxy, detections.mask, range(len(given_labels))),
        key=lambda x: x[0], reverse=True
    )

    if top_x_detections is not None:
        detections_combined = detections_combined[:top_x_detections]

    # Further filter based on proximity
    filtered_detections = []
    for idx, current_det in enumerate(detections_combined):
        _, curr_class_id, curr_xyxy, curr_mask, _ = current_det
        curr_center = ((curr_xyxy[0] + curr_xyxy[2]) / 2, (curr_xyxy[1] + curr_xyxy[3]) / 2)
        curr_area = (curr_xyxy[2] - curr_xyxy[0]) * (curr_xyxy[3] - curr_xyxy[1])
        keep = True
        
            # Calculate the total number of pixels as a threshold for small masks
        total_pixels = image.shape[0] * image.shape[1]
        small_mask_size = total_pixels * min_mask_size_ratio

        # check mask size and remove if too small
        mask_size = np.count_nonzero(current_det[3])
        if mask_size < small_mask_size:
            print(f"Removing {classes.get_classes_arr()[curr_class_id]} because the mask size is too small.")
            keep = False

        for other in filtered_detections:
            _, other_class_id, other_xyxy, other_mask, _ = other
            
            if mask_iou(curr_mask, other_mask) > iou_threshold:
                keep = False
                print(f"Removing {classes.get_classes_arr()[curr_class_id]} because it has an IoU of {mask_iou(curr_mask, other_mask)} with object {classes.get_classes_arr()[other_class_id]}.")
                break
            
            
            other_center = ((other_xyxy[0] + other_xyxy[2]) / 2, (other_xyxy[1] + other_xyxy[3]) / 2)
            other_area = (other_xyxy[2] - other_xyxy[0]) * (other_xyxy[3] - other_xyxy[1])

            # Calculate distance between centers
            dist = np.sqrt((curr_center[0] - other_center[0]) ** 2 + (curr_center[1] - other_center[1]) ** 2)
            if dist < proximity_threshold:
                if (keep_larger and curr_area > other_area) or (not keep_larger and curr_area < other_area):
                    filtered_detections[:] = [d for d in filtered_detections if d is not other]
                else:
                    keep = False
                    break
        # print(given_labels[idx])
        if classes.get_classes_arr()[curr_class_id] in classes.bg_classes:
            print(f"Removing {classes.get_classes_arr()[curr_class_id]} because it is a background class, specifically {classes.bg_classes}.")
            keep = False

        if keep:
            filtered_detections.append(current_det)

    # Unzip the filtered results
    if not filtered_detections:
        # Every detection got filtered out (small mask / IoU dup / bg class) -- zip(*[])
        # has nothing to unpack, so build the empty result directly. mask is shaped
        # (0, H, W), not None, since downstream code (e.g. mask_subtract_contained) calls
        # .copy() on detections.mask unconditionally.
        return sv.Detections(
            class_id=np.array([], dtype=np.int64),
            confidence=np.array([], dtype=np.float32),
            xyxy=np.zeros((0, 4), dtype=np.float32),
            mask=np.zeros((0, *image.shape[:2]), dtype=np.bool_),
        ), []
    confidences, class_ids, xyxy, masks, indices = zip(*filtered_detections)
    filtered_labels = [given_labels[i] for i in indices]

    # Create new detections object
    filtered_detections = sv.Detections(
        class_id=np.array(class_ids, dtype=np.int64),
        confidence=np.array(confidences, dtype=np.float32),
        xyxy=np.array(xyxy, dtype=np.float32),
        mask=np.array(masks, dtype=np.bool_)
    )

    return filtered_detections, filtered_labels

def get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=False, suffix="annotated_for_vlm.jpg", ):

    # Define suffixes based on whether edges are included
    if w_edges:
        suffix = suffix.replace(".jpg", "_w_edges.jpg")

    # Create the file path
    vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg").with_name(
        (det_exp_vis_path / color_path.name).stem + suffix
    )
    return str(vis_save_path)

def get_vlm_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, make_captions_flag, openai_client):
    """
    Filter and annotate detections, then get per-object captions from a VLM.
    Object relations are no longer sourced from a VLM here -- they're derived
    from 3D geometry once, after the whole frame loop (slam/utils.py's
    build_final_object_graph), from the final point cloud.

    Args:
        image (numpy.ndarray): The image on which detections are performed.
        curr_det (list): Current detections from the detection model.
        obj_classes (list): Object classes used in detection.
        detection_class_labels (list): Labels for each detection class.
        det_exp_vis_path (str): Directory path for saving visualizations.
        color_path (str): Additional path element for creating unique save paths.
        make_captions_flag (bool): Flag indicating whether to caption detected objects.
        openai_client (OpenAIClient): Client object for OpenAI used in captioning.

    Returns:
        tuple: A tuple containing the following elements:
            - labels (list): The labels after filtering detections.
            - captions (list): List of captions for each detected object if `make_captions_flag` is True, otherwise None.
    """
    # Filter the detections
    filtered_detections, labels = filter_detections(
        image=image,
        detections=curr_det,
        classes=obj_classes,
        top_x_detections=150000,
        confidence_threshold=0.00001,
        given_labels=detection_class_labels,
    )

    captions = None
    if make_captions_flag:
        vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        annotated_image_for_vlm, _ = annotate_for_vlm(image, filtered_detections, labels, save_path=vis_save_path_for_vlm)

        label_list = []
        for label in labels:
            label_num = str(label.split(" ")[-1])
            label_name = re.sub(r'\s*\d+$', '', label).strip()
            full_label = f"{label_num}: {label_name}"
            label_list.append(full_label)

        cv2.imwrite(str(vis_save_path_for_vlm), annotated_image_for_vlm)
        print(f"Line 313, vis_save_path_for_vlm: {vis_save_path_for_vlm}")

        captions = get_obj_captions_from_image_gpt4v(openai_client, vis_save_path_for_vlm, label_list)

    return labels, captions


def get_discovered_classes_path(det_exp_path):
    return det_exp_path / "discovered_classes.txt"


def get_representative_frames_path(det_exp_path):
    return det_exp_path / "representative_frames.json"


def discover_scene_vocabulary(
    dataset,
    sam_predictor,
    openai_client,
    det_exp_path,
    detections_exp_suffix,
    voxel_size,
    pixel_stride,
    max_representative_frames,
    sam_conf,
    sam_min_segment_area_px,
    sam_max_segment_area_ratio,
    sam_max_segments_per_frame,
    device,
):
    """
    Discovers a per-scene-variant object vocabulary to replace a fixed classes
    file: selects a minimal set of representative frames covering the scene's
    3D extent (see slam.utils.select_representative_frames), asks the VLM to
    enumerate objects visible in each whole frame, and separately asks it to
    name each SAM "segment everything" mask crop in that frame. The union of
    both becomes the vocabulary. Returns the path to the saved classes .txt
    file, one name per line -- the format ObjectClasses reads.

    The representative frame list is cached under representative_frames_path
    (reused verbatim on a later call with the same det_exp_path, even if the
    VLM/SAM naming is redone) so it can be inspected/reused later.
    """
    det_exp_path.mkdir(parents=True, exist_ok=True)
    discovered_classes_path = get_discovered_classes_path(det_exp_path)
    representative_frames_path = get_representative_frames_path(det_exp_path)

    if representative_frames_path.exists():
        with open(representative_frames_path, "r") as f:
            representative_frame_indices = json.load(f)["representative_frame_indices"]
        coverage_info = {"voxel_size": voxel_size, "pixel_stride": pixel_stride, "total_scene_voxels": None}
    else:
        frame_indices = list(range(len(dataset)))
        depth_arrays, poses, cam_Ks = [], [], []
        for idx in frame_indices:
            _, depth_tensor, intrinsics, *_ = dataset[idx]
            depth_arrays.append(depth_tensor[..., 0].cpu().numpy())
            poses.append(dataset.poses[idx].cpu().numpy())
            cam_Ks.append(intrinsics.cpu().numpy()[:3, :3])

        representative_frame_indices, coverage_info = select_representative_frames(
            frame_indices, depth_arrays, np.stack(poses), cam_Ks,
            voxel_size=voxel_size, pixel_stride=pixel_stride,
        )

    if max_representative_frames is not None:
        representative_frame_indices = representative_frame_indices[:max_representative_frames]

    discovered = set()
    per_frame_stats = {}

    for frame_idx in representative_frame_indices:
        color_path = Path(dataset.color_paths[frame_idx])
        image = cv2.imread(str(color_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        H, W = image_rgb.shape[:2]

        frame_objects = get_frame_object_list(openai_client, str(color_path))
        discovered.update(frame_objects)

        # Passing bboxes/points/labels=None here triggers ultralytics SAM's automatic
        # "segment everything" mode internally. points_stride/conf_thres etc. (the
        # generate()-specific knobs) aren't reachable through this predict() call in
        # the installed ultralytics version -- Model.predict() validates all extra
        # kwargs as top-level cfg overrides (check_dict_alignment) and doesn't forward
        # them to inference()/generate() -- so only the standard `conf` (box-score
        # filter) is configurable here; points density uses ultralytics' own default.
        sam_out = sam_predictor.predict(str(color_path), verbose=False, conf=sam_conf, quantize=16)
        has_masks = sam_out[0].masks is not None and len(sam_out[0].masks.data) > 0
        masks = sam_out[0].masks.data.cpu().numpy() if has_masks else np.empty((0, H, W), dtype=bool)
        boxes = sam_out[0].boxes.xyxy.cpu().numpy() if has_masks else np.empty((0, 4))

        areas = masks.reshape(len(masks), -1).sum(axis=1) if len(masks) else np.empty((0,))
        keep_idx = np.nonzero((areas >= sam_min_segment_area_px) & (areas <= sam_max_segment_area_ratio * H * W))[0]
        if len(keep_idx) > sam_max_segments_per_frame:
            keep_idx = keep_idx[np.argsort(-areas[keep_idx])[:sam_max_segments_per_frame]]

        segment_names = []
        for i in keep_idx:
            crop = crop_with_padding(pil_image, boxes[i], padding=20)
            name = get_segment_object_name(openai_client, crop)
            if name:
                segment_names.append(name)
        discovered.update(segment_names)

        per_frame_stats[str(frame_idx)] = {
            "full_frame_objects": len(frame_objects),
            "segment_masks_total": int(len(masks)),
            "segment_objects_kept": len(segment_names),
        }

    discovered_classes = sorted({c.strip().lower() for c in discovered if c.strip()})

    with open(discovered_classes_path, "w") as f:
        for cls in discovered_classes:
            f.write(cls + "\n")

    with open(representative_frames_path, "w") as f:
        json.dump({
            "detections_exp_suffix": detections_exp_suffix,
            "num_total_frames": len(dataset),
            "representative_frame_indices": list(representative_frame_indices),
            "representative_color_paths": [str(dataset.color_paths[i]) for i in representative_frame_indices],
            "coverage": coverage_info,
            "discovery_params": {
                "sam_conf": sam_conf,
                "sam_min_segment_area_px": sam_min_segment_area_px,
                "sam_max_segment_area_ratio": sam_max_segment_area_ratio,
                "sam_max_segments_per_frame": sam_max_segments_per_frame,
                "max_representative_frames": max_representative_frames,
            },
            "per_frame_object_counts": per_frame_stats,
            "num_discovered_classes": len(discovered_classes),
        }, f, indent=2)

    return discovered_classes_path


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)  # Call the function with any arguments it was called with
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds")
        return result  # Return the result of the function call
    return wrapper

def get_exp_config_save_path(exp_out_path, is_detection_config=False):
    params_file_name = "config_params"
    if is_detection_config:
        params_file_name += "_detections"
    return exp_out_path / f"{params_file_name}.json"

def save_hydra_config(hydra_cfg, exp_out_path, is_detection_config=False):
    exp_out_path.mkdir(exist_ok=True, parents=True)
    with open(get_exp_config_save_path(exp_out_path, is_detection_config), "w") as f:
        dict_to_dump = cfg_to_dict(hydra_cfg)
        json.dump(dict_to_dump, f, indent=2)

def should_exit_early(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Check if we should exit early
        if data.get("exit_early", False):
            # Reset the exit_early flag to False
            data["exit_early"] = False
            # Write the updated data back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file)
            return True
        else:
            return False
    except Exception as e:
        # If there's an error reading the file or the key doesn't exist, 
        # log the error and return False
        print(f"Error reading {file_path}: {e}")
        logging.info(f"Error reading {file_path}: {e}")
        return False

def save_detection_results(base_path, results):
    base_path.mkdir(exist_ok=True, parents=True)
    for key, value in results.items():
        save_path = Path(base_path) / f"{key}"
        if isinstance(value, np.ndarray):
            # Save NumPy arrays using .npz for efficient storage
            np.savez_compressed(f"{save_path}.npz", value)
        else:
            # For other types, fall back to pickle
            with gzip.open(f"{save_path}.pkl.gz", "wb") as f:
                pickle.dump(value, f)
                
def load_saved_detections(base_path):
    base_path = Path(base_path)
    
    # Construct potential .pkl.gz file path based on the base_path
    potential_pkl_gz_path = Path(str(base_path) + '.pkl.gz')

    # Check if the constructed .pkl.gz file exists
    # This is the old wat 
    if potential_pkl_gz_path.exists() and potential_pkl_gz_path.is_file():
        # The path points directly to a .pkl.gz file
        with gzip.open(potential_pkl_gz_path, "rb") as f:
            return pickle.load(f)
    elif base_path.is_dir():
        loaded_detections = {}
        for file_path in base_path.iterdir():
            # Handle files based on their extension, adjusting the key extraction method
            if file_path.suffix == '.npz':
                key = file_path.name.replace('.npz', '')
                with np.load(file_path, allow_pickle=True) as data:
                    loaded_detections[key] = data['arr_0']
            elif file_path.suffix == '.gz' and file_path.suffixes[-2] == '.pkl':
                key = file_path.name.replace('.pkl.gz', '')
                with gzip.open(file_path, "rb") as f:
                    loaded_detections[key] = pickle.load(f)
        return loaded_detections
    else:
        raise FileNotFoundError(f"No valid file or directory found at {base_path}")
        
        
class ObjectClasses:
    """
    Manages object classes, allowing for exclusion of background classes.

    This class facilitates the loading of class names from a specified file. It also manages
    background classes based on configuration, allowing for their inclusion or exclusion.
    Background classes are ["wall", "floor", "ceiling"] by default.

    Attributes:
        classes_file_path (str): Path to the file containing class names, one per line.

    Usage:
        obj_classes = ObjectClasses(classes_file_path, skip_bg=True)
        model.set_classes(obj_classes.get_classes_arr())
    """
    def __init__(self, classes_file_path, bg_classes, skip_bg):
        self.classes_file_path = Path(classes_file_path)
        self.bg_classes = bg_classes
        self.skip_bg = skip_bg
        self.classes = self._load_classes()

    def _load_classes(self):
        with open(self.classes_file_path, "r") as f:
            all_classes = [cls.strip() for cls in f.readlines() if cls.strip()]

        if self.skip_bg:
            return [cls for cls in all_classes if cls not in self.bg_classes]
        return all_classes

    def get_classes_arr(self):
        """
        Returns the list of class names, excluding background classes if configured to do so.
        """
        return self.classes

    def get_bg_classes_arr(self):
        """
        Returns the list of background class names, if configured to do so.
        """
        return self.bg_classes

def save_obj_json(exp_suffix, exp_out_path, objects):
    """
    Saves the objects to a JSON file with the specified suffix.

    Args:
    - exp_suffix (str): Suffix for the experiment, used in naming the saved file.
    - exp_out_path (Path or str): Output path for the experiment's saved files.
    - objects: The objects to save, assumed to have necessary attributes.
    """
    json_obj_list = {}
    for curr_idx, curr_obj in enumerate(objects):
        obj_key = f"object_{curr_idx + 1}"
        curr_bbox = curr_obj['bbox']
        bbox_extent_raw = curr_bbox.extent if hasattr(curr_bbox, 'extent') else curr_bbox.get_extent()
        bbox_extent = [round(val, 2) for val in bbox_extent_raw]  # Round values to 2 decimal places
        bbox_center = [round(val, 2) for val in curr_obj['bbox'].center]  # Assuming `center` is an iterable like a list or tuple
        bbox_volume = round(bbox_extent[0] * bbox_extent[1] * bbox_extent[2], 2)  # Calculate volume and round to 2 decimal places
        
        obj_dict = {
            "id": curr_obj['curr_obj_num'],
            "object_tag": curr_obj['class_name'],
            "object_caption": curr_obj.get('consolidated_caption', None),
            "bbox_extent": bbox_extent,
            "bbox_center": bbox_center,
            "bbox_volume": bbox_volume  # Add the volume to the dictionary
        }
        json_obj_list[obj_key] = obj_dict
        
    json_obj_out_path = Path(exp_out_path) / f"obj_json_{exp_suffix}.json"
    json_obj_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_obj_out_path, "w") as f:
        json.dump(json_obj_list, f, indent=2)
    print(f"Saved object JSON to {json_obj_out_path}")
    

def save_edge_json(exp_suffix, exp_out_path, objects, edges):
    """
    Saves the edges to a JSON file with the specified suffix.

    Args:
    - exp_suffix (str): Suffix for the experiment, used in naming the saved file.
    - exp_out_path (Path or str): Output path for the experiment's saved files.
    - objects: The objects involved in the edges.
    - edges: The edges to save, assumed to have necessary attributes.
    """
    json_edge_list = {}
    for curr_idx, curr_edge_item in enumerate(list(edges.edges_by_index.items())):
        curr_edj_tup, curr_edge = curr_edge_item
        obj1_idx = curr_edge.obj1_idx
        obj2_idx = curr_edge.obj2_idx
        rel_type = curr_edge.rel_type
        obj1_class_name = objects[obj1_idx]['class_name']
        obj2_class_name = objects[obj2_idx]['class_name']
        obj1_curr_obj_num = objects[obj1_idx]['curr_obj_num']
        obj2_curr_obj_num = objects[obj2_idx]['curr_obj_num']

        edj_dict = {
            "edge_id": curr_idx,
            "edge_description": f"{obj1_class_name} {rel_type} {obj2_class_name}",
            # Kept for utils/visualize_full_scenegraph.py compatibility (edge merging,
            # rendered line width) -- always 1 now that edges are computed once from the
            # final point cloud rather than reconfirmed across frames.
            "num_detections": curr_edge.num_detections,
            "object_1_id": obj1_curr_obj_num,
            "object_1_tag": obj1_class_name,
            "object_2_id": obj2_curr_obj_num,
            "object_2_tag": obj2_class_name,
            "relationship": rel_type,
            "center_distance": curr_edge.center_distance,
            "center_diff": curr_edge.center_diff,
            "surface_min_distance": curr_edge.surface_min_distance,
            "surface_diff": curr_edge.surface_diff,
            "iou": curr_edge.iou,
            "giou": curr_edge.giou,
            "iom": curr_edge.iom,
        }
        json_edge_list[f"edge_{curr_idx}"] = edj_dict
        
    json_edge_out_path = Path(exp_out_path) / f"edge_json_{exp_suffix}.json"
    json_edge_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_edge_out_path, "w") as f:
        json.dump(json_edge_list, f, indent=2)
    print(f"Saved edge JSON to {json_edge_out_path}")

def save_pointcloud(exp_suffix, exp_out_path, cfg, objects, obj_classes, edges=None, up_axis=None, up_direction=None):
    """
    Saves the point cloud data to a .pkl.gz file.

    Args:
    - exp_suffix (str): Suffix for the experiment, used in naming the saved file.
    - exp_out_path (Path or str): Output path for the experiment's saved files.
    - objects: The objects to save, assumed to have a `to_serializable()` method.
    - obj_classes: The object classes, assumed to have a `get_classes_arr()` method.
    - up_axis, up_direction: detect_up_vector()'s result (slam/utils.py), persisted here so downstream
      debug tooling (e.g. convert_concept_graphs_to_scene_diff_benchmark_data.py's moved-objects plot) can
      reuse this camera-grounded estimate instead of re-deriving a weaker one from the saved points alone.
    """
    print("saving map...")
    # Prepare the results dictionary
    results = {
        'objects': objects.to_serializable(),
        'cfg': cfg_to_dict(cfg),
        'class_names': obj_classes.get_classes_arr(),
        'edges': edges.to_serializable() if edges is not None else None,
        'up_axis': up_axis,
        'up_direction': up_direction,
    }

    # Define the save path for the point cloud
    pcd_save_path = Path(exp_out_path) / f"pcd_{exp_suffix}.pkl.gz"
    # Make the directory if it doesn't exist
    pcd_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the point cloud data
    with gzip.open(pcd_save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved point cloud to {pcd_save_path}")
    if edges is not None:
        print(f"Also saved edges to {pcd_save_path}")


def save_objects_for_frame(obj_all_frames_out_path, frame_idx, objects, obj_min_detections, adjusted_pose, color_path):
    save_path = obj_all_frames_out_path / f"{frame_idx:06d}.pkl.gz"
    filtered_objects = [obj for obj in objects if obj['num_detections'] >= obj_min_detections]
    prepared_objects = prepare_objects_save_vis(MapObjectList(filtered_objects))
    result = {
        "camera_pose": adjusted_pose, 
        "objects": prepared_objects,
        "frame_idx": frame_idx,
        "num_objects": len(filtered_objects),
        "color_path": str(color_path)
    }
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(result, f)
        

