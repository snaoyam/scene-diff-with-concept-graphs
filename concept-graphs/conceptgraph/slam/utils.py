from collections import Counter
import copy
import gzip
import json
import logging
import pickle
from pathlib import Path
# from conceptgraph.utils.logging_metrics import track_denoising,
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
import cv2
# from line_profiler import profile

import numpy as np
from omegaconf import DictConfig
import omegaconf
import open3d as o3d
from scipy.spatial import cKDTree
import torch

import torch.nn.functional as F

import faiss
import uuid

from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList, DetectionList, to_tensor

from conceptgraph.utils.ious import compute_3d_iou_accurate_batch

tracker = MappingTracker()


def to_scalar(d: np.ndarray | torch.Tensor | float) -> int | float:
    '''
    Convert the d to a scalar
    '''
    if isinstance(d, float):
        return d
    
    elif "numpy" in str(type(d)):
        assert d.size == 1
        return d.item()
    
    elif isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()
    
    else:
        raise TypeError(f"Invalid type for conversion: {type(d)}")

def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    '''
    Get fx, fy, cx, cy from the intrinsics matrix
    
    return 4 scalars
    '''
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    # Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def init_pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ## Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan( # inint
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def init_process_pcd(pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    
    if dbscan_remove_noise and run_dbscan:
        pcd = init_pcd_denoise_dbscan(
            pcd, 
            eps=dbscan_eps, 
            min_points=dbscan_min_points
        )
        
    return pcd

# @profile
def process_pcd(pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    
    if dbscan_remove_noise and run_dbscan:
        pass
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=dbscan_eps, 
            min_points=dbscan_min_points
        )
        
    return pcd

# @profile
def get_bounding_box(spatial_sim_type, pcd):
    if ("accurate" in spatial_sim_type or "overlap" in spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()

# @profile
def merge_obj2_into_obj1(obj1, obj2, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type, device, run_dbscan=True):

    '''
    Merges obj2 into obj1 with structured attribute handling, including explicit checks for unhandled keys.

    Parameters:
    - obj1, obj2: Objects to merge.
    - downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type: Parameters for point cloud processing.
    - device: Computation device.
    - run_dbscan: Whether to run DBSCAN for noise removal.

    Returns:
    - obj1: Updated object after merging.
    '''
    global tracker
    
    tracker.track_merge(obj1, obj2)
    
    # Attributes to be explicitly handled
    extend_attributes = ['image_idx', 'mask_idx', 'color_path', 'class_id', 'mask', 'xyxy', 'conf', 'contain_number', 'captions',
                          'mask_coverage']
    add_attributes = ['num_detections', 'num_obj_in_class']
    # 'is_seeded_prior'/'seed_source_before_id' (load_prior_scene_objects_as_seeds):
    # obj2 can itself be a seed here, not just obj1 -- fuse_detections_geometry_only
    # bridges multiple matched candidates into one anchor, and a seed can be one of the
    # OTHER candidates being folded in, not only the anchor. Skipping them means obj1
    # keeps its own seed status (or lack of it) regardless of obj2's; the fields exist
    # only for post-hoc traceability and are never read to make a fusion decision.
    # 'is_large'/'mask_coverage_stat'/'extent_ratio' are derived, not carried: the merged
    # object's size is not obj1's or obj2's but the union's, so keeping obj1's value here
    # would be wrong in exactly the case that matters. fuse_detections_geometry_only
    # recomputes them on the anchor right after each merge; listing them here only stops
    # the unhandled-key check below from rejecting an object that already has them.
    skip_attributes = ['id', 'class_name', 'is_background', 'new_counter', 'curr_obj_num', 'inst_color',
                        'is_seeded_prior', 'seed_source_before_id',
                        'is_large', 'mask_coverage_stat', 'extent_ratio']  # 'inst_color' just keeps obj1's
    custom_handled = ['pcd', 'bbox', 'clip_ft', 'dino_ft', 'clip_ft_mean', 'dino_ft_mean', 'text_ft', 'n_points',
                       'confirmed_pcd']

    # Check for unhandled keys and throw an error if there are
    all_handled_keys = set(extend_attributes + add_attributes + skip_attributes + custom_handled)
    unhandled_keys = set(obj2.keys()) - all_handled_keys
    if unhandled_keys:
        raise ValueError(f"Unhandled keys detected in obj2: {unhandled_keys}. Please update the merge function to handle these attributes.")

    # Custom handling for 'pcd', 'bbox', 'clip_ft', and 'text_ft'
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    # Process extend and add attributes
    for attr in extend_attributes:
        if attr in obj1 and attr in obj2:
            obj1[attr].extend(obj2[attr])
    
    for attr in add_attributes:
        if attr in obj1 and attr in obj2:
            obj1[attr] += obj2[attr]

    # Handling 'caption'
    if 'caption' in obj1 and 'caption' in obj2:
        # n_obj1_det = obj1['num_detections']
        for key, value in obj2['caption'].items():
            obj1['caption'][key + n_obj1_det] = value

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan)
    # update n_points
    obj1['n_points'] = len(np.asarray(obj1['pcd'].points))

    # Update 'bbox'
    obj1['bbox'] = get_bounding_box(spatial_sim_type, obj1['pcd'])
    obj1['bbox'].color = [0, 1, 0]

    # confirmed_pcd only exists on seed-descended objects (load_prior_scene_objects_as_seeds)
    # -- grow it in lockstep with 'pcd', but only from obj2's own CONFIRMED share: if obj2 is
    # itself a seed, that's obj2['confirmed_pcd'] (not its whole pcd, which may still carry
    # unconfirmed seed geometry); otherwise obj2 is a real detection or an ordinary
    # (never-seeded) object, and its entire pcd counts as confirmed by this scan.
    if 'confirmed_pcd' in obj1:
        obj1['confirmed_pcd'] += obj2.get('confirmed_pcd', obj2['pcd'])
        obj1['confirmed_pcd'] = process_pcd(
            obj1['confirmed_pcd'], downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan
        )

    # Merge and normalize 'clip_ft'
    obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det + obj2['clip_ft'] * n_obj2_det) / (n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # Merge and normalize 'dino_ft'
    obj1['dino_ft'] = (obj1['dino_ft'] * n_obj1_det + obj2['dino_ft'] * n_obj2_det) / (n_obj1_det + n_obj2_det)
    obj1['dino_ft'] = F.normalize(obj1['dino_ft'], dim=0)

    # The same running mean over the per-detection features, but deliberately NOT
    # renormalized. Every per-frame CLIP/DINO feature arrives L2-normalized, so this
    # recurrence holds the plain arithmetic mean of unit vectors -- and for unit vectors
    #     mean over all N*M frame pairs of cos(b_i, a_j) == dot(mean(b), mean(a))
    # exactly. That identity is what lets the before/after benchmark comparison score a
    # pair of objects over every frame pair with one dot product instead of storing
    # per-frame features (see convert_concept_graphs_to_scene_diff_benchmark_data.py).
    # The magnitude these vectors lose by renormalizing is precisely the object's
    # cross-frame appearance consistency, which is the part that identity needs; the
    # 'clip_ft'/'dino_ft' above stay normalized because everything else expects that.
    for attr in ('clip_ft_mean', 'dino_ft_mean'):
        if attr in obj1 and attr in obj2:
            obj1[attr] = (obj1[attr] * n_obj1_det + obj2[attr] * n_obj2_det) / (n_obj1_det + n_obj2_det)

    # merge text_ft
    # obj2['text_ft'] = to_tensor(obj2['text_ft'], device)
    # obj1['text_ft'] = to_tensor(obj1['text_ft'], device)
    # obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
    #                    obj2['text_ft'] * n_obj2_det) / (
    #                    n_obj1_det + n_obj2_det)
    # obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)

    return obj1

# @profile
def compute_overlap_matrix_general(objects_a: MapObjectList, objects_b = None, downsample_voxel_size = None) -> np.ndarray:
    """
    Compute the overlap matrix between two sets of objects represented by their point clouds. This function can also perform self-comparison when `objects_b` is not provided. The overlap is quantified based on the proximity of points from one object to the nearest points of another, within a threshold specified by `downsample_voxel_size`.

    Parameters
    ----------
    objects_a : MapObjectList
        A list of object representations where each object contains a point cloud ('pcd') and bounding box ('bbox').
        This is the primary set of objects for comparison.

    objects_b : Optional[MapObjectList]
        A second list of object representations similar to `objects_a`. If None, `objects_a` will be compared with itself to calculate self-overlap. Defaults to None.

    downsample_voxel_size : Optional[float]
        The threshold for determining whether points are close enough to be considered overlapping. Specifically, it's the square of the maximum distance allowed between points from two objects to consider those points as overlapping.
        Must be provided; if None, a ValueError is raised.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (len(objects_a), len(objects_b)) containing the overlap ratios between objects.
        The overlap ratio is defined as the fraction of points in the second object's point cloud that are within `downsample_voxel_size` distance to any point in the first object's point cloud.

    Raises
    ------
    ValueError
        If `downsample_voxel_size` is not provided.

    Notes
    -----
    The function uses the FAISS library for efficient nearest neighbor searches to compute the overlap.
    Additionally, it employs a 3D IoU (Intersection over Union) computation for bounding boxes to quickly filter out pairs of objects without spatial overlap, improving performance.
    - The overlap matrix helps identify potential duplicates or matches between new and existing objects based on spatial overlap.
    - High values (e.g., >0.8) in the matrix suggest a significant overlap, potentially indicating duplicates or very close matches.
    - Moderate values (e.g., 0.5-0.8) may indicate similar objects with partial overlap.
    - Low values (<0.5) generally suggest distinct objects with minimal overlap.
    - The choice of a "match" threshold depends on the application's requirements and may require adjusting based on observed outcomes.

    Examples
    --------
    >>> objects_a = [{'pcd': pcd1, 'bbox': bbox1}, {'pcd': pcd2, 'bbox': bbox2}]
    >>> objects_b = [{'pcd': pcd3, 'bbox': bbox3}, {'pcd': pcd4, 'bbox': bbox4}]
    >>> downsample_voxel_size = 0.05
    >>> overlap_matrix = compute_overlap_matrix_general(objects_a, objects_b, downsample_voxel_size)
    >>> print(overlap_matrix)
    """
    # if downsample_voxel_size is None, raise an error
    if downsample_voxel_size is None:
        raise ValueError("downsample_voxel_size is not provided")

    # hardcoding for now because its this value is actually not supposed to be the downsample voxel size
    downsample_voxel_size = 0.025

    # are we doing self comparison?
    same_objects = objects_b is None
    objects_b = objects_a if same_objects else objects_b

    len_a = len(objects_a)
    len_b = len(objects_b)
    overlap_matrix = np.zeros((len_a, len_b))

    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_a = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_a] # m arrays
    indices_a = [faiss.IndexFlatL2(points_a_arr.shape[1]) for points_a_arr in points_a] # m indices

    # Add the points from the numpy arrays to the corresponding FAISS indices
    for idx_a, points_a_arr in zip(indices_a, points_a):
        idx_a.add(points_a_arr)

    points_b = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_b] # n arrays

    bbox_a = objects_a.get_stacked_values_torch('bbox')
    bbox_b = objects_b.get_stacked_values_torch('bbox')
    
    # def compute_3d_iou_accurate_batch_safe(bbox1, bbox2):
    #     try:
    #         return compute_3d_iou_accurate_batch(bbox1, bbox2)
    #     except ValueError as e:
    #         if str(e) == "Plane vertices are not coplanar":
    #             # Log the error or handle it in a way that's appropriate for your application
    #             print("Non-coplanar boxes detected; returning zero IoU.")
    #             return torch.zeros((bbox1.size(0), bbox2.size(0)))  # Return a zero IoU matrix
    #         else:
    #             raise  # Re-raise other unexpected exceptions
    # ious = compute_3d_iou_accurate_batch_safe(bbox_a, bbox_b)        
    
    ious = compute_3d_iou_accurate_batch(bbox_a, bbox_b) # (m, n)


    # Compute the pairwise overlaps
    for idx_a in range(len_a):
        for idx_b in range(len_b):

            # skip same object comparison if same_objects is True
            if same_objects and idx_a == idx_b:
                continue

            # skip if the boxes do not overlap at all
            if ious[idx_a,idx_b] < 1e-6:
                continue

            # get the distance of the nearest neighbor of
            # each point in points_b[idx_b] to the points_a[idx_a]
            D, I = indices_a[idx_a].search(points_b[idx_b], 1) 
            overlap = (D < downsample_voxel_size ** 2).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[idx_a, idx_b] = overlap / len(points_b[idx_b])

    return overlap_matrix

# @profile
def merge_overlap_objects(
    merge_overlap_thresh: float,
    merge_visual_sim_thresh: float,
    merge_text_sim_thresh: float,
    objects: MapObjectList,
    overlap_matrix: np.ndarray,
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    map_edges = None,
):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]
    
    # Sort indices of overlap ratios in descending order
    sort = np.argsort(overlap_ratio)[::-1]  
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]
    
    merge_operations = []  # to track merge operations
    kept_objects = np.ones(
        len(objects), dtype=bool
    )  # Initialize all objects as 'kept' initially
    
    index_updates = list(range(len(objects)))  # Initialize index updates with the same indices

    for i, j, ratio in zip(x, y, overlap_ratio):
        if ratio > merge_overlap_thresh:
            visual_sim = F.cosine_similarity(
                to_tensor(objects[i]["clip_ft"]),
                to_tensor(objects[j]["clip_ft"]),
                dim=0,
            )
            # text_sim = F.cosine_similarity(
            #     to_tensor(objects[i]["text_ft"]),
            #     to_tensor(objects[j]["text_ft"]),
            #     dim=0,
            # )
            text_sim = visual_sim
            if (visual_sim > merge_visual_sim_thresh) and (text_sim > merge_text_sim_thresh):
                if kept_objects[j]:  # Check if the target object has not been merged into another
                    # Merge object i into object j
                    objects[j] = merge_obj2_into_obj1(
                        objects[j],
                        objects[i],
                        downsample_voxel_size,
                        dbscan_remove_noise,
                        dbscan_eps,
                        dbscan_min_points,
                        spatial_sim_type,
                        device,
                        run_dbscan=True,
                    )
                    kept_objects[i] = False  # Mark object i as 'merged'
                    merge_operations.append((i, j))  # Record this merge for edge updates 
                    index_updates[i] = None  # Update index as merged
        else:
            break  # Stop processing if the current overlap ratio is below the threshold
        
    # Update remaining indices in index_updates
    current_index = 0
    for original_index, is_kept in enumerate(kept_objects):
        if is_kept:
            index_updates[original_index] = current_index
            current_index += 1
        else:
            index_updates[original_index] = None

    # Create a new list of objects excluding those that were merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)

    return objects, index_updates

# @profile
def denoise_objects(
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    objects: MapObjectList,
):
    tracker = DenoisingTracker()  # Get the singleton instance of DenoisingTracker
    logging.debug(f"Starting denoising with {len(objects)} objects")
    for i in range(len(objects)):
        og_object_pcd = objects[i]["pcd"]
        
        if len(og_object_pcd.points) <= 1: # no need to denoise
            objects[i]["pcd"] = og_object_pcd
        else:
            # Adjust the call to process_pcd with explicit parameters
            objects[i]["pcd"] = process_pcd(
                objects[i]["pcd"],
                downsample_voxel_size,
                dbscan_remove_noise,
                dbscan_eps,
                dbscan_min_points,
                run_dbscan=True,
            )
            if len(objects[i]["pcd"].points) < 4:
                objects[i]["pcd"] = og_object_pcd

        # Adjust the call to get_bounding_box with explicit parameters
        objects[i]["bbox"] = get_bounding_box(spatial_sim_type, objects[i]["pcd"])
        objects[i]["bbox"].color = [0, 1, 0]
        logging.debug(f"Finished denoising object {i} out of {len(objects)}")
        # Use the tracker's method
        tracker.track_denoising(objects[i]["id"], len(og_object_pcd.points), len(objects[i]["pcd"].points))
        
        # track_denoising(objects[i]["id"], len(og_object_pcd.points), len(objects[i]["pcd"].points))
        logging.debug(f"before denoising: {len(og_object_pcd.points)}, after denoising: {len(objects[i]['pcd'].points)}")
    logging.debug(f"Finished denoising with {len(objects)} objects")
    return objects

# @profile
def merge_objects(
    merge_overlap_thresh: float,
    merge_visual_sim_thresh: float,
    merge_text_sim_thresh: float,
    objects: MapObjectList,
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    do_edges: bool = False,
    map_edges = None,
):
    if len(objects) == 0:
        return objects
    if merge_overlap_thresh <= 0:
        return objects

    # Assuming compute_overlap_matrix requires only `objects` and `downsample_voxel_size`
    overlap_matrix = compute_overlap_matrix_general(
        objects_a=objects,
        objects_b=None,
        downsample_voxel_size=downsample_voxel_size,
    )
    print("Before merging:", len(objects))
    # old_objects = copy.deepcopy(objects)
    # Pass all necessary configuration parameters to merge_overlap_objects
    objects, index_updates = merge_overlap_objects(
        merge_overlap_thresh=merge_overlap_thresh,
        merge_visual_sim_thresh=merge_visual_sim_thresh,
        merge_text_sim_thresh=merge_text_sim_thresh,
        objects=objects,
        overlap_matrix=overlap_matrix,
        downsample_voxel_size=downsample_voxel_size,
        dbscan_remove_noise=dbscan_remove_noise,
        dbscan_eps=dbscan_eps,
        dbscan_min_points=dbscan_min_points,
        spatial_sim_type=spatial_sim_type,
        device=device,
        map_edges=map_edges,
    )
    
    # print(f"MERGE OPERATIONS: \n{merge_operations}")
    
    # print("MERGE OPERATIONS: ")
    # for oper in merge_operations:
    #     obj_1_curr_num = old_objects[oper[0]]['curr_obj_num']
    #     obj_2_curr_num = old_objects[oper[1]]['curr_obj_num']
    #     print(f"Merge {obj_1_curr_num} into {obj_2_curr_num}")
    
    # k=1
    # for i, j in zip(list(range(len(old_objects))), index_updates):
    #     print(i,j)
    
    # if map_edges is not None:
    if do_edges:
        map_edges.merge_update_indices(index_updates)
        map_edges.update_objects_list(objects)
        print("After merging:", len(objects))

    # if map_edges is not None:
    #     # Apply each recorded merge operation to the edges
    #     for source_idx, dest_idx in merge_operations:
    #         map_edges.merge_objects_edges(source_idx, dest_idx)
    #     map_edges.update_objects_list(objects)
    #     print("After merging:", len(objects))
        
        # now update all the edge indices using the index_updates, how?

    if do_edges:
        return objects, map_edges
    else:
        return objects
    
def filter_captions(captions, detection_class_labels):
    # Gracefully handle missing captions or labels
    if detection_class_labels is None:
        return []

    if captions is None:
        captions = []

    # Create a dictionary to map id to the index in the captions list
    captions_index = {item['id']: index for index, item in enumerate(captions)}
    
    # Initialize a new list to store the cleaned and matched captions
    new_captions = []
    
    # Process each detection class label
    for label in detection_class_labels:
        # Split the label by spaces
        parts = label.split()
        # The last part is the id
        id_str = parts[-1]
        # The rest are the name
        name = ' '.join(parts[:-1])
        
        # Check if the id exists in the captions dictionary
        if id_str in captions_index:
            # Add the caption from the captions list to the new list
            new_captions.append(captions[captions_index[id_str]])
        else:
            # Add a new entry with a default/empty caption to avoid NoneType errors downstream
            new_captions.append({"id": id_str, "name": name, "caption": None})
    
    return new_captions


# @profile
def filter_gobs(
    gobs: dict,
    image: np.ndarray,
    skip_bg: bool = None,  # Explicitly passing skip_bg
    BG_CLASSES: list = None,  # Explicitly passing BG_CLASSES
    mask_area_threshold: float = 10,  # Default value as fallback
    max_bbox_area_ratio: float = None,  # Explicitly passing max_bbox_area_ratio
    max_mask_area_ratio: float = None,  # Explicitly passing max_mask_area_ratio
    mask_conf_threshold: float = None,  # Explicitly passing mask_conf_threshold
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs

    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]

        # Skip masks that are too small
        mask_area = gobs['mask'][mask_idx].sum()
        if mask_area < max(mask_area_threshold, 10):
            logging.debug(f"Skipped due to small mask area ({mask_area} pixels) - Class: {class_name}")
            continue

        # Skip the BG classes
        if skip_bg and class_name in BG_CLASSES:
            logging.debug(f"Skipped background class: {class_name}")
            continue

        # Skip the non-background boxes that are too large
        image_area = image.shape[0] * image.shape[1]
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            if max_bbox_area_ratio is not None and bbox_area > max_bbox_area_ratio * image_area:
                logging.debug(f"Skipped due to large bounding box area ratio - Class: {class_name}, Area Ratio: {bbox_area/image_area:.4f}")
                continue

        # Skip masks that cover too much of the frame. Sibling of the box test above,
        # but on the mask itself, and applied to background classes too: a wall or floor
        # mask is faithfully enormous, whereas its box is enormous for any diagonally
        # oriented thin object as well, so the box test can't be tightened far enough to
        # catch the former without also catching the latter. This drops one frame's
        # detection, never the object -- it survives through the frames where it doesn't
        # fill the view -- which is what lets the threshold sit as high as it does.
        if max_mask_area_ratio is not None and mask_area > max_mask_area_ratio * image_area:
            logging.debug(f"Skipped due to large mask area ratio - Class: {class_name}, Area Ratio: {mask_area/image_area:.4f}")
            continue

        # Skip masks with low confidence
        if mask_conf_threshold is not None and gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < mask_conf_threshold:
                # logging.debug(f"Skipped due to low confidence ({gobs['confidence'][mask_idx]}) - Class: {class_name}")
                continue

        idx_to_keep.append(mask_idx)

    # for key in gobs.keys():
    #     print(key, type(gobs[key]), len(gobs[key]))

    return slice_gobs(gobs, idx_to_keep)


def slice_gobs(gobs, idx_to_keep):
    '''
    Keep only `idx_to_keep` across every per-detection array in gobs, at once.

    Factored out because more than one stage now narrows a frame's detections, and every
    one of gobs' parallel arrays has to be cut identically or masks stop lining up with
    their boxes/features. The exemptions below are load-bearing, not incidental: 'classes'
    is the frame's whole vocabulary rather than a per-detection array, and
    'detection_class_labels' deliberately is NOT exempt -- exempting it misaligns edges
    from objects.
    '''
    for attribute in gobs.keys():
        if isinstance(gobs[attribute], str) or attribute == "classes":  # Captions
            continue
        if attribute in ['labels', 'edges', 'text_feats', 'captions']:
            # Note: this statement was used to also exempt 'detection_class_labels' but that causes a bug. It causes the edges to be misalgined with the objects.
            continue
        elif isinstance(gobs[attribute], list):
            gobs[attribute] = [gobs[attribute][i] for i in idx_to_keep]
        elif isinstance(gobs[attribute], np.ndarray):
            gobs[attribute] = gobs[attribute][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")

    filtered_captions = filter_captions(gobs['captions'], gobs['detection_class_labels'])
    gobs['captions'] = filtered_captions

    return gobs


def dedup_gobs_by_mask_iou(gobs, iou_thresh):
    '''
    Where one object picked up several near-identical masks under different labels, keep
    only the highest-confidence one.

    An open-vocab detector scores every vocabulary word against the same region, so one
    sofa gets a "sofa" box and a "couch" box, and YOLO's own NMS (agnostic_nms=False)
    only dedupes within a class. SAM, prompted with two nearly-identical boxes, hands
    back the *identical* mask: over one scan, 96 detection pairs had a mask IoU of exactly
    1.000 -- sofa/couch, pillow/cushion, tote bag/shoe -- while the next-highest pair sat
    at 0.745. Nothing lives in between, so the threshold is not delicate.

    This looks only at mask geometry. An earlier version of this stage keyed off CLIP and
    DINO similarity and grouped differently frame to frame because those similarities sat
    right on their threshold; IoU here does not have that problem.

    Returns (gobs, number of detections dropped).
    '''
    masks = gobs.get('mask')
    if iou_thresh <= 0 or masks is None or len(masks) < 2:
        return gobs, 0

    masks = np.asarray(masks).astype(bool)
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    # Highest confidence first, so the survivor of each duplicate group is the one the
    # rest are compared against (and the label that reaches mapping).
    order = np.argsort(-np.asarray(gobs['confidence']))

    keep = []
    for i in order:
        is_duplicate = False
        for k in keep:
            intersection = int((masks[i] & masks[k]).sum())
            union = int(areas[i] + areas[k] - intersection)
            if union > 0 and intersection >= iou_thresh * union:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(int(i))

    n_dropped = len(masks) - len(keep)
    if n_dropped == 0:
        return gobs, 0
    return slice_gobs(gobs, sorted(keep)), n_dropped


def resize_gobs(gobs, image):

    # If the shapes are the same, no resizing is necessary
    if gobs['mask'].shape[1:] == image.shape[:2]:
        return gobs

    new_masks = []

    for mask_idx in range(len(gobs['xyxy'])):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs['mask'][mask_idx]
        # Rescale the xyxy coordinates to the image shape
        x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
        x1 = round(x1 * image.shape[1] / mask.shape[1])
        y1 = round(y1 * image.shape[0] / mask.shape[0])
        x2 = round(x2 * image.shape[1] / mask.shape[1])
        y2 = round(y2 * image.shape[0] / mask.shape[0])
        gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]

        # Reshape the mask to the image shape
        mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        new_masks.append(mask)

    if len(new_masks) > 0:
        gobs['mask'] = np.asarray(new_masks)

    return gobs


# @profile
def make_detection_list_from_pcd_and_gobs(
    obj_pcds_and_bboxes, gobs, color_path, obj_classes, image_idx
):
    '''
    This function makes a detection list for the objects
    Ideally I don't want it to be needed, the detection list has too much info and is inefficient
    '''
    global tracker
    detection_list = DetectionList()
    # bg_detection_list = DetectionList()
    for mask_idx in range(len(gobs['mask'])):
        if obj_pcds_and_bboxes[mask_idx] is None: # pointcloud was discarded
            continue

        curr_class_name = gobs['classes'][gobs['class_id'][mask_idx]]
        curr_class_idx = obj_classes.get_classes_arr().index(curr_class_name)
        
        is_bg_object = bool(curr_class_name in obj_classes.get_bg_classes_arr())
        
        # print(f"Line 937, tracker.total_object_count INCREMENTED: {tracker.total_object_count }")
        num_obj_in_class = tracker.curr_class_count[curr_class_name]
        
        
        detected_object = {
            'id' : uuid.uuid4(),
            'image_idx' : [image_idx],                             # idx of the image
            
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'class_name' : curr_class_name,                         # global class id for this detection
            'class_id' : [curr_class_idx],                         # global class id for this detection
            'captions' : [gobs['captions'][mask_idx]],           # captions for this detection
            'num_detections' : 1,                            # number of detections in this object
            'mask': [gobs['mask'][mask_idx]],
            # Fraction of the frame this one mask covered. Accumulated per detection
            # (rather than derived from 'mask' later) because annotate_large_objects and
            # the per-frame fusion loop both ask for a percentile over it repeatedly, and
            # a well-observed object ends up holding hundreds of full-frame bool arrays.
            'mask_coverage': [float(np.asarray(gobs['mask'][mask_idx]).mean())],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': len(obj_pcds_and_bboxes[mask_idx]['pcd'].points),
            # 'pixel_area': [mask.sum()],
            'contain_number': [None],                          # This will be computed later
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            'is_background': is_bg_object,
            
            # These are for the entire 3D object
            'pcd': obj_pcds_and_bboxes[mask_idx]['pcd'],
            'bbox': obj_pcds_and_bboxes[mask_idx]['bbox'],
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),
            'dino_ft': to_tensor(gobs['dino_feats'][mask_idx]),
            # Same values, but these two are averaged WITHOUT renormalizing as detections
            # merge -- see merge_obj2_into_obj1 for why that difference matters.
            'clip_ft_mean': to_tensor(gobs['image_feats'][mask_idx]),
            'dino_ft_mean': to_tensor(gobs['dino_feats'][mask_idx]),
            # 'text_ft': to_tensor(gobs['text_feats'][mask_idx]),
            'num_obj_in_class': num_obj_in_class,
            'curr_obj_num': tracker.total_object_count,
            'new_counter' : tracker.brand_new_counter,
        }
        # detected_object['curr_obj_num']
        # print(f"Line 969, detected_object['image_idx']: {detected_object['image_idx']}")
        # print(f"Line 971, detected_object['class_name']: {detected_object['class_name']}")
        # print(f"Line 966, detected_object['curr_obj_num']: {detected_object['curr_obj_num']}")
        
        # if is_bg_object:
        #     bg_detection_list.append(detected_object)
        # else:
        detection_list.append(detected_object)
        
        tracker.curr_class_count[curr_class_name] += 1
        tracker.total_object_count += 1
        tracker.brand_new_counter += 1

    return detection_list # , bg_detection_list


def load_prior_scene_objects_as_seeds(
    prior_pcd_path,
    obj_classes,
    downsample_voxel_size,
    dbscan_remove_noise,
    dbscan_eps,
    dbscan_min_points,
    spatial_sim_type,
):
    '''
    Loads a previously-built scene variant's final object list (e.g. "before", for
    the "after" variant currently being scanned) and reconstructs its trusted,
    non-background objects as seed candidates for this scan's geometry-only fusion.

    A seed only carries geometry/appearance/class identity forward -- num_detections
    starts at 0 and image_idx/mask/xyxy/etc start empty, exactly like a real
    detection dict except with zero detections instead of one. It participates in
    the same bbox/point-overlap/normal-consistency gates as any accumulated object
    (see geometric_fusion.fuse_detections_geometry_only), so a real per-frame
    detection either merges into it (num_detections becomes >0, extend_attributes
    get populated -- the object behaves as if it had always been in `objects`) or
    doesn't (the physical object moved/is gone, or the detector never proposed
    anything there). Either way num_detections is the ground truth of whether this
    scan itself ever re-confirmed the object -- the caller MUST drop any seed still
    at num_detections==0 before this scan's objects are treated as final, or an
    unconfirmed seed would misrepresent a removed/moved object as still present.

    Loaded objects only ever supply geometry/pcd_np, pcd_color_np, clip_ft, dino_ft,
    clip_ft_mean, dino_ft_mean, class_name -- pcd/bbox are not stored in the saved
    file (see MapObjectList.to_serializable) and are rebuilt here the same way a
    fresh detection's are (init_process_pcd + get_bounding_box), with normals left
    unset so they're computed lazily the same way a real new object's are.
    '''
    global tracker

    with gzip.open(prior_pcd_path, "rb") as f:
        data = pickle.load(f)

    classes_arr = obj_classes.get_classes_arr()
    seeds = DetectionList()
    for obj in data["objects"]:
        if obj.get("is_background", False):
            continue
        # Default True mirrors convert_concept_graphs_to_scene_diff_benchmark_data.py's
        # own obj.get("recognition_trusted", True) -- an object never subjected to the
        # recognition-confidence pass (cfg.compute_recognition_confidence=False) is
        # trusted by default rather than being silently excluded from seeding.
        if not obj.get("recognition_trusted", True):
            continue
        # Same reasoning one step further: a large object is barred from asserting a
        # change anyway, so seeding one buys nothing, while a seed the size of a sofa
        # sitting in the candidate list from frame 0 is the single best-placed thing in
        # the scan to absorb everything resting on it. Absent on graphs built before the
        # flag existed, which then seed exactly as they used to.
        if obj.get("is_large", False):
            continue

        class_name = obj["class_name"]
        try:
            class_id = classes_arr.index(class_name)
        except ValueError:
            logging.warning(
                f"load_prior_scene_objects_as_seeds: prior object class "
                f"'{class_name}' not found in this scan's vocabulary; skipping seed."
            )
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(obj["pcd_np"]))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(obj["pcd_color_np"]))
        pcd = init_process_pcd(pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points)
        if len(pcd.points) == 0:
            continue
        bbox = get_bounding_box(spatial_sim_type, pcd)

        num_obj_in_class = tracker.curr_class_count[class_name]

        seed = {
            'id': uuid.uuid4(),
            'image_idx': [],
            'mask_idx': [],
            'color_path': [],
            'class_name': class_name,
            'class_id': [class_id],
            'captions': [],
            'num_detections': 0,
            'mask': [],
            'mask_coverage': [],
            'xyxy': [],
            'conf': [],
            'n_points': len(pcd.points),
            'contain_number': [],
            'inst_color': np.random.rand(3),
            'is_background': False,

            'pcd': pcd,
            'bbox': bbox,
            'clip_ft': to_tensor(obj['clip_ft']),
            'dino_ft': to_tensor(obj['dino_ft']),
            'clip_ft_mean': to_tensor(obj['clip_ft_mean']),
            'dino_ft_mean': to_tensor(obj['dino_ft_mean']),
            'num_obj_in_class': num_obj_in_class,
            'curr_obj_num': tracker.total_object_count,
            'new_counter': tracker.brand_new_counter,

            # Seed-only bookkeeping, consumed by rerun_realtime_mapping.py's "after"
            # branch to drop still-unconfirmed seeds, and left on confirmed objects
            # for later analysis. merge_obj2_into_obj1 only validates obj2's (the
            # incoming real detection's) keys, so these extra keys on a seed acting
            # as obj1 pass through untouched by every merge.
            'is_seeded_prior': True,
            'seed_source_before_id': obj['id'],
            # Points contributed by THIS scan's own real detections only -- starts
            # empty since the seed itself carries none. merge_obj2_into_obj1 grows
            # this in lockstep with 'pcd' whenever this object is obj1, but only ever
            # from obj2's *confirmed* share (its own confirmed_pcd if it has one,
            # otherwise its whole pcd -- see merge_obj2_into_obj1). A plain object
            # (no seed history) never gets this key at all: geometric_fusion.py reads
            # its absence as "every one of this object's points is already confirmed
            # by this scan", which is trivially true for a non-seeded object.
            'confirmed_pcd': o3d.geometry.PointCloud(),
        }
        seeds.append(seed)

        tracker.curr_class_count[class_name] += 1
        tracker.total_object_count += 1

    return seeds

# @profile
def dynamic_downsample(points, colors=None, target=5000):
    """
    Simplified and configurable downsampling function that dynamically adjusts the 
    downsampling rate based on the number of input points. If a target of -1 is provided, 
    downsampling is bypassed, returning the original points and colors.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) for N points.
        target (int): Target number of points to aim for in the downsampled output, 
                      or -1 to bypass downsampling.
        colors (torch.Tensor, optional): Corresponding colors tensor of shape (N, 3). 
                                         Defaults to None.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Downsampled points and optionally 
                                                     downsampled colors, or the original 
                                                     points and colors if target is -1.
    """
    # Check if downsampling is bypassed
    if target == -1:
        return points, colors
    
    num_points = points.size(0)
    
    # If the number of points is less than or equal to the target, return the original points and colors
    if num_points <= target:
        return points, colors
    
    # Calculate downsampling factor to aim for the target number of points
    downsample_factor = max(1, num_points // target)
    
    # Select points based on the calculated downsampling factor
    downsampled_points = points[::downsample_factor]
    
    # If colors are provided, downsample them with the same factor
    downsampled_colors = colors[::downsample_factor] if colors is not None else None

    return downsampled_points, downsampled_colors


def batch_mask_depth_to_points_colors(
    depth_tensor: torch.Tensor,
    masks_tensor: torch.Tensor,
    cam_K: torch.Tensor,
    image_rgb_tensor: torch.Tensor = None,  # Parameter for RGB image tensor
    device: str = 'cuda'
) -> tuple:
    """
    Converts a batch of masked depth images to 3D points and corresponding colors.

    Args:
        depth_tensor (torch.Tensor): A tensor of shape (N, H, W) representing the depth images.
        masks_tensor (torch.Tensor): A tensor of shape (N, H, W) representing the masks for each depth image.
        cam_K (torch.Tensor): A tensor of shape (3, 3) representing the camera intrinsic matrix.
        image_rgb_tensor (torch.Tensor, optional): A tensor of shape (N, H, W, 3) representing the RGB images. Defaults to None.
        device (str, optional): The device to perform the computation on. Defaults to 'cuda'.

    Returns:
        tuple: A tuple containing the 3D points tensor of shape (N, H, W, 3) and the colors tensor of shape (N, H, W, 3).
    """
    N, H, W = masks_tensor.shape
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
    
    # Generate grid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(0, H, device=device), torch.arange(0, W, device=device), indexing='ij')
    z = depth_tensor.repeat(N, 1, 1) * masks_tensor  # Apply masks to depth

    valid = (z > 0).float()  # Mask out zeros

    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    points = torch.stack((x, y, z), dim=-1) * valid.unsqueeze(-1)  # Shape: (N, H, W, 3)

    if image_rgb_tensor is not None:
        # Repeat RGB image for each mask and apply masks
        repeated_rgb = image_rgb_tensor.repeat(N, 1, 1, 1) * masks_tensor.unsqueeze(-1)
        colors = repeated_rgb * valid.unsqueeze(-1)  # Apply valid mask to filter out background
    else:
        print("No RGB image provided, assigning random colors to objects")
        # log it as well
        logging.warning("No RGB image provided, assigning random colors to objects")
        # Generate a random color for each mask
        random_colors = torch.randint(0, 256, (N, 3), device=device, dtype=torch.float32) / 255.0  # RGB colors in [0, 1]
        # Expand dims to match (N, H, W, 3) and apply to valid points
        colors = random_colors.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1) * valid.unsqueeze(-1)

    return points, colors


def backproject_frame_to_voxel_indices(
    depth_array: np.ndarray,
    cam_K: np.ndarray,
    pose: np.ndarray,
    voxel_size: float,
    pixel_stride: int = 4,
    max_depth: float = None,
) -> np.ndarray:
    """
    Backprojects a single unmasked frame's depth map into world-space, on a
    strided pixel grid (cheap coverage estimate, not a reconstruction), and
    returns the set of unique voxel-grid cells it touches.

    pose must be the same raw camera-to-world convention used elsewhere in this
    file to backproject objects (dataset.poses[frame_idx], i.e. what
    detections_to_obj_pcd_and_bbox receives as trans_pose) -- not
    dataset.transformed_poses, which is a different (first-frame-relative)
    coordinate frame.

    Returns:
        (M, 3) int array of unique (vx, vy, vz) voxel indices = floor(world_xyz / voxel_size).
    """
    H, W = depth_array.shape
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]

    ys, xs = np.mgrid[0:H:pixel_stride, 0:W:pixel_stride]
    z = depth_array[ys, xs]
    valid = z > 0
    if max_depth is not None:
        valid &= z <= max_depth
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.int32)

    xs, ys, z = xs[valid], ys[valid], z[valid]
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)  # (N, 3)

    pts_world = pts_cam @ pose[:3, :3].T + pose[:3, 3]

    voxel_idx = np.floor(pts_world / voxel_size).astype(np.int32)
    return np.unique(voxel_idx, axis=0)


def select_representative_frames(
    frame_indices: list,
    depth_arrays: list,
    poses: np.ndarray,
    cam_Ks: list,
    voxel_size: float = 0.1,
    pixel_stride: int = 4,
) -> tuple:
    """
    Greedily selects the smallest subset of frames whose combined 3D coverage
    equals the coverage achievable by using every candidate frame -- i.e. every
    voxel touched by any frame ends up touched by some selected frame. The
    coverage target is derived from the scene's own frames rather than a
    hand-picked percentage, in the same spirit as build_final_object_graph's
    auto-selected edge-distance multiplier.

    Args:
        frame_indices: candidate frame indices, aligned with depth_arrays/poses/cam_Ks.
        depth_arrays: list of (H, W) depth arrays in meters, one per frame_indices entry.
        poses: (len(frame_indices), 4, 4) raw camera-to-world poses.
        cam_Ks: list of (3, 3) camera intrinsics, one per frame_indices entry.

    Returns:
        (selected_frame_indices, info) where selected_frame_indices is a list of
        entries from frame_indices in greedy pick order (largest marginal coverage
        gain first), and info is a dict with voxel_size/pixel_stride/total_scene_voxels.
    """
    frame_voxels = []
    for depth_array, pose, cam_K in zip(depth_arrays, poses, cam_Ks):
        voxels = backproject_frame_to_voxel_indices(depth_array, cam_K, pose, voxel_size, pixel_stride)
        frame_voxels.append(set(map(tuple, voxels)))

    full_scene_voxels = set().union(*frame_voxels) if frame_voxels else set()

    covered = set()
    remaining = set(range(len(frame_indices)))
    selected = []
    while covered != full_scene_voxels and remaining:
        best_i, best_gain = None, -1
        for i in remaining:
            gain = len(frame_voxels[i] - covered)
            if gain > best_gain:
                best_i, best_gain = i, gain
        if best_gain <= 0:
            break
        selected.append(frame_indices[best_i])
        covered |= frame_voxels[best_i]
        remaining.discard(best_i)

    info = {
        "voxel_size": voxel_size,
        "pixel_stride": pixel_stride,
        "total_scene_voxels": len(full_scene_voxels),
    }
    return selected, info


def detections_to_obj_pcd_and_bbox(
    depth_array,
    masks,
    cam_K, 
    image_rgb=None, 
    trans_pose=None, 
    min_points_threshold=5, 
    spatial_sim_type='axis_aligned', 
    obj_pcd_max_points = None,
    downsample_voxel_size = None,
    dbscan_remove_noise = None,
    dbscan_eps = None,
    dbscan_min_points = None,
    run_dbscan = None,
    device='cuda'
):
    """
    This function processes a batch of objects to create colored point clouds, apply transformations, and compute bounding boxes.

    Args:
        depth_array (numpy.ndarray): Array containing depth values.
        masks (numpy.ndarray): Array containing binary masks for each object.
        cam_K (numpy.ndarray): Camera intrinsic matrix.
        image_rgb (numpy.ndarray, optional): RGB image. Defaults to None.
        trans_pose (numpy.ndarray, optional): Transformation matrix. Defaults to None.
        min_points_threshold (int, optional): Minimum number of points required for an object. Defaults to 5.
        spatial_sim_type (str, optional): Type of spatial similarity. Defaults to 'axis_aligned'.
        device (str, optional): Device to use. Defaults to 'cuda'.

    Returns:
        list: List of dictionaries containing processed objects. Each dictionary contains a point cloud and a bounding box.
    """
    N, H, W = masks.shape

    # Convert inputs to tensors and move to the specified device
    depth_tensor = torch.from_numpy(depth_array).to(device).float()
    masks_tensor = torch.from_numpy(masks).to(device).float()
    cam_K_tensor = torch.from_numpy(cam_K).to(device).float()

    if image_rgb is not None:
        image_rgb_tensor = torch.from_numpy(image_rgb).to(device).float() / 255.0  # Normalize RGB values
    else:
        image_rgb_tensor = None

    points_tensor, colors_tensor = batch_mask_depth_to_points_colors(
        depth_tensor, masks_tensor, cam_K_tensor, image_rgb_tensor, device
    )

    processed_objects = [None] * N  # Initialize with placeholders
    for i in range(N):
        mask_points = points_tensor[i]
        mask_colors = colors_tensor[i] if colors_tensor is not None else None

        valid_points_mask = mask_points[:, :, 2] > 0
        if torch.sum(valid_points_mask) < min_points_threshold:
            continue

        valid_points = mask_points[valid_points_mask]
        valid_colors = mask_colors[valid_points_mask] if mask_colors is not None else None

        downsampled_points, downsampled_colors = dynamic_downsample(valid_points, colors=valid_colors, target=obj_pcd_max_points)

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(downsampled_points.cpu().numpy())
        if downsampled_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(downsampled_colors.cpu().numpy())

        if trans_pose is not None:
            pcd.transform(trans_pose)  # Apply transformation directly to the point cloud
            pass

        bbox = get_bounding_box(spatial_sim_type, pcd)
        if bbox.volume() < 1e-6:
            continue

        processed_objects[i] = {'pcd': pcd, 'bbox': bbox}

    return processed_objects


def processing_needed(
    process_interval, run_on_final_frame, frame_idx, is_final_frame=False
):

    if process_interval > 0 and (frame_idx+1) % process_interval == 0:
        return True
    if run_on_final_frame and is_final_frame:
        return True
    return False


def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)
    
    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        # For dataset whose depth and RGB have different resolutions
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg

def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float=0.025):
    objects_to_save = copy.deepcopy(objects)
            
    # Downsample the point cloud
    for i in range(len(objects_to_save)):
        objects_to_save[i]['pcd'] = objects_to_save[i]['pcd'].voxel_down_sample(downsample_size)

    # Remove unnecessary keys
    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in [
                'pcd', 'bbox', 'clip_ft', 'dino_ft', 'text_ft', 'class_id', 'num_detections', 'inst_color'
            ]:
                del objects_to_save[i][k]
                
    return objects_to_save.to_serializable()

def detect_up_vector(objects, camera_positions):
    '''
    Auto-detects which world-frame axis is "up" and which sign along it is
    "down", since neither is documented anywhere for this Pi3-estimated
    coordinate frame.

    Axis: among X/Y/Z, the up-axis is the one with the smallest 95% spread
    (97.5th minus 2.5th percentile) of the combined point cloud accumulated
    so far -- most rooms are wider/longer than they are tall, and percentiles
    (rather than raw min/max) keep this robust to outlier points.

    Sign: compared to the camera positions' median value on that axis,
    whichever side has more point-cloud mass is "down" -- scene cameras tend
    to tilt down more than up, capturing more content below eye height.

    Called once, from build_final_object_graph(), after all frames are
    processed and `objects` holds the final merged/denoised point clouds --
    not progressively per-frame, so the estimate isn't skewed by a
    partially-observed scene.

    Returns (up_axis: int, up_direction: float) such that
    point[up_axis] * up_direction increases going physically "up".
    '''
    all_points = np.concatenate(
        [np.asarray(o['pcd'].points) for o in objects if len(o['pcd'].points) > 0], axis=0
    )
    spreads = [np.subtract(*np.percentile(all_points[:, ax], [97.5, 2.5])) for ax in range(3)]
    up_axis = int(np.argmin(spreads))

    camera_ref = np.median(camera_positions[:, up_axis])
    below = np.sum(all_points[:, up_axis] < camera_ref)
    above = np.sum(all_points[:, up_axis] > camera_ref)
    up_direction = 1.0 if below >= above else -1.0
    return up_axis, up_direction


def compute_robust_bbox(points, trim_percentile=2.5):
    '''
    Per-axis [trim_percentile, 100-trim_percentile] percentile range -- an
    outlier-robust bounding box (the "95%-confidence" bbox). Reused both as
    the reference center for compute_object_radius() and as the up-axis-
    excluded horizontal footprint for compute_2d_overlap_metrics().
    Returns (lo, hi), each shape (3,).
    '''
    lo = np.percentile(points, trim_percentile, axis=0)
    hi = np.percentile(points, 100 - trim_percentile, axis=0)
    return lo, hi


def compute_object_radius(points, robust_center):
    '''
    Mean distance from `robust_center` to each of the object's points.
    `robust_center` should be a robust bbox center (compute_robust_bbox), not
    the raw point-cloud centroid -- the centroid is biased toward whichever
    side of the object the camera happened to scan more densely (occluded
    faces are never observed), while the bbox center isn't.
    '''
    return float(np.linalg.norm(np.asarray(points) - robust_center, axis=1).mean())


def robust_min_surface_distance(points1, points2, percentile=5):
    '''
    A noise-robust "closest surface" distance between two point clouds.
    Naively taking the single literal closest point pair is easily distorted
    by one stray/noisy point, so instead: for every point in points1, find its
    nearest neighbor in points2 (and vice versa, via KDTree -- O((N+M) log)
    instead of a full O(N*M) pairwise matrix), concatenate both directions'
    distances, and pick the point pair at the `percentile`-th rank of that
    combined, sorted array. Using an actual ranked data point (not an
    interpolated percentile value) means the returned distance always exactly
    matches the returned point pair's separation.
    Returns (distance, point1, point2).
    '''
    tree1, tree2 = cKDTree(points1), cKDTree(points2)
    d_1to2, nn_1to2 = tree2.query(points1)
    d_2to1, nn_2to1 = tree1.query(points2)
    candidates = [(points1[i], points2[nn_1to2[i]], d_1to2[i]) for i in range(len(points1))]
    candidates += [(points1[nn_2to1[j]], points2[j], d_2to1[j]) for j in range(len(points2))]
    dists = np.array([d for _, _, d in candidates])
    rank = int(round(percentile / 100 * (len(dists) - 1)))
    chosen = candidates[np.argsort(dists)[rank]]
    return float(chosen[2]), np.asarray(chosen[0]), np.asarray(chosen[1])


def compute_2d_overlap_metrics(lo1, hi1, lo2, hi2):
    '''
    IoU / GIoU / IoM between two axis-aligned 2D rectangles, each given as
    (lo, hi) corner pairs (e.g. a robust bbox's horizontal-plane axes).
    '''
    inter = np.clip(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0, None)
    inter_area = inter[0] * inter[1]
    area1, area2 = np.prod(hi1 - lo1), np.prod(hi2 - lo2)
    union = area1 + area2 - inter_area
    iou = inter_area / union if union > 0 else 0.0
    iom = inter_area / min(area1, area2) if min(area1, area2) > 0 else 0.0
    enc = np.maximum(hi1, hi2) - np.minimum(lo1, lo2)
    enc_area = enc[0] * enc[1]
    giou = iou - (enc_area - union) / enc_area if enc_area > 0 else iou
    return float(iou), float(giou), float(iom)


def build_final_object_graph(objects, camera_positions, map_edges, frame_idx):
    '''
    Builds the whole scene's object graph once, from the final (fully
    merged/denoised) point clouds, after all frames have been processed --
    replacing the old per-frame incremental edge computation.

    Connectivity: object i and j get an edge iff their robust_min_surface_distance
    is <= n * (radius_i + radius_j), where n is auto-selected as the smallest
    value that leaves no object isolated (n = the largest, over all objects,
    of that object's own smallest distance/radius-sum ratio to any other object).

    Relation label: the taller (up-axis) object is always the subject
    (object_1). "on top of" if the two objects' horizontal-footprint IoU > 0
    (i.e. they overlap at all when looking straight down the up-axis),
    "next to" otherwise.

    Returns (map_edges, up_axis, up_direction) -- the detect_up_vector() result
    is returned alongside map_edges so callers can persist it (e.g. into the
    saved pcd file) for reuse by downstream debug tooling instead of having
    them re-derive a weaker, camera-position-unaware estimate.
    '''
    if len(objects) < 2:
        return map_edges, None, None

    up_axis, up_direction = detect_up_vector(objects, camera_positions)
    horiz_axes = [a for a in range(3) if a != up_axis]

    points_list = [np.asarray(o['pcd'].points) for o in objects]
    robust_bboxes = [compute_robust_bbox(pts) for pts in points_list]
    centers = [(lo + hi) / 2.0 for lo, hi in robust_bboxes]
    radii = [compute_object_radius(pts, c) for pts, c in zip(points_list, centers)]

    n_obj = len(objects)
    pair_data = {}
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            dist, p1, p2 = robust_min_surface_distance(points_list[i], points_list[j])
            ratio = dist / (radii[i] + radii[j]) if (radii[i] + radii[j]) > 0 else float("inf")
            pair_data[(i, j)] = {"distance": dist, "p1": p1, "p2": p2, "ratio": ratio}

    # Smallest n that leaves no object isolated = the largest, over all
    # objects, of that object's own nearest-neighbor ratio.
    per_obj_min_ratio = [float("inf")] * n_obj
    for (i, j), d in pair_data.items():
        per_obj_min_ratio[i] = min(per_obj_min_ratio[i], d["ratio"])
        per_obj_min_ratio[j] = min(per_obj_min_ratio[j], d["ratio"])
    n = max(per_obj_min_ratio) if n_obj > 0 else 0.0
    print(f"Auto-selected edge distance multiplier n={n:.3f} (no isolated objects)")

    up_vec = np.eye(3)[up_axis]
    for (i, j), d in pair_data.items():
        if d["ratio"] > n:
            continue

        height_i = np.dot(centers[i], up_vec) * up_direction
        height_j = np.dot(centers[j], up_vec) * up_direction
        subj, obj_ = (i, j) if height_i >= height_j else (j, i)
        p_subj, p_obj = (d["p1"], d["p2"]) if subj == i else (d["p2"], d["p1"])

        lo_s, hi_s = robust_bboxes[subj]
        lo_o, hi_o = robust_bboxes[obj_]
        iou, giou, iom = compute_2d_overlap_metrics(
            lo_s[horiz_axes], hi_s[horiz_axes], lo_o[horiz_axes], hi_o[horiz_axes]
        )
        rel_type = "on top of" if iou > 0 else "next to"

        # num_detections := number of frames in which both objects were
        # observed together -- equals the number of times scenegraph_viz
        # would actually draw this edge (rerun_realtime_mapping.py's
        # per-frame viz loop only draws an edge when frame_idx is in both
        # endpoints' image_idx).
        frame_overlap = len(set(objects[subj]['image_idx']) & set(objects[obj_]['image_idx']))

        map_edges.add_or_update_edge(
            subj, obj_, rel_type, first_detected=frame_idx,
            num_detections=frame_overlap,
            center_distance=float(np.linalg.norm(centers[subj] - centers[obj_])),
            center_diff=(centers[subj] - centers[obj_]).tolist(),
            surface_min_distance=d["distance"],
            surface_diff=(p_subj - p_obj).tolist(),
            iou=iou, giou=giou, iom=iom,
        )

    return map_edges, up_axis, up_direction