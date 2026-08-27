
import copy
import torch
import numpy as np
import open3d as o3d

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)

class DetectionList(list):
    def get_values(self, key, idx:int=None):
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]
    
    def get_stacked_values_torch(self, key, idx:int=None):
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)
    
    def get_stacked_values_numpy(self, key, idx:int=None):
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)
    
    def __add__(self, other):
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list
    
    def __iadd__(self, other):
        self.extend(other)
        return self

class MapObjectList(DetectionList):
    def to_serializable(self):
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)
            
            s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])
            s_obj_dict['dino_ft'] = to_numpy(s_obj_dict['dino_ft'])
            for attr in ('clip_ft_mean', 'dino_ft_mean'):
                if attr in s_obj_dict:
                    s_obj_dict[attr] = to_numpy(s_obj_dict[attr])
            # s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])
            
            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)

            del s_obj_dict['pcd']
            del s_obj_dict['bbox']

            s_obj_list.append(s_obj_dict)

        return s_obj_list

# not sure if I will use this
class MapEdge():
    def __init__(self, obj1_idx, obj2_idx, rel_type, num_detections=1, first_detected=None,
                 center_distance=None, center_diff=None, surface_min_distance=None,
                 surface_diff=None, iou=None, giou=None, iom=None):
        self.obj1_idx = obj1_idx
        self.obj2_idx = obj2_idx
        self.rel_type = rel_type
        self.num_detections = num_detections
        self.first_detected = first_detected # frame index that the object was first detected
        # Geometry metrics computed by build_final_object_graph (slam/utils.py).
        self.center_distance = center_distance
        self.center_diff = center_diff
        self.surface_min_distance = surface_min_distance
        self.surface_diff = surface_diff
        self.iou = iou
        self.giou = giou
        self.iom = iom

    def __str__(self):
        return f"({self.obj1_idx}, {self.rel_type}, {self.obj2_idx}), num_det: {self.num_detections}"

    def __repr__(self):
        return str(self)

class MapEdgeMapping:
    def __init__(self, objects):
        self.objects = objects  # Reference to the list of existing objects
        self.edges_by_index = {}  # {(obj1_index, obj2_index): MapEdge}
        self.edges_by_uuid = {}  # {(obj1_uuid, obj2_uuid): MapEdge}

    def add_or_update_edge(self, obj1_index, obj2_index, rel_type, first_detected=None, **metrics):
        obj1_uuid, obj2_uuid = self.objects[obj1_index]['id'], self.objects[obj2_index]['id']
        uuid_key = (obj1_uuid, obj2_uuid)

        if obj1_index == obj2_index:
            print(f"LOOOPY EDGE DETECTED: {obj1_index} == {obj2_index}")
            pass

        if (obj1_index, obj2_index) in self.edges_by_index:
            edge = self.edges_by_index[(obj1_index, obj2_index)]
            edge.num_detections += 1
        else:
            edge = MapEdge(obj1_index, obj2_index, rel_type, first_detected=first_detected, **metrics)
            self.edges_by_index[(obj1_index, obj2_index)] = edge
            self.edges_by_uuid[uuid_key] = edge
            
    def delete_edge(self, obj1_index, obj2_index):
        # Check if the edge exists
        if (obj1_index, obj2_index) in self.edges_by_index:
            # Get the UUIDs of the objects
            obj1_uuid = self.objects[obj1_index]['id']
            obj2_uuid = self.objects[obj2_index]['id']
            uuid_key = (obj1_uuid, obj2_uuid)

            # Remove the edge from both index-based and UUID-based dictionaries
            del self.edges_by_index[(obj1_index, obj2_index)]
            if uuid_key in self.edges_by_uuid:
                del self.edges_by_uuid[uuid_key]
            else:
                # If the edge is not found in the UUID-based dictionary, print a warning
                print(f"Edge between {obj1_index} and {obj2_index} not found in UUID-based storage.")
            print(f"Edge between {obj1_index} and {obj2_index} deleted successfully.")
        else:
            print(f"Edge between {obj1_index} and {obj2_index} does not exist.")

    def delete_object_edges(self, obj_index):
        # Remove all edges associated with the object at obj_index
        to_remove = [key for key in self.edges_by_index if obj_index in key]
        for key in to_remove:
            # Remove from both index-based and UUID-based storage
            del self.edges_by_index[key]
            uuid_key = (self.objects[key[0]]['id'], self.objects[key[1]]['id'])
            del self.edges_by_uuid[uuid_key]
            
    def update_indices(self, index_map, new_objects):
        self.objects = new_objects  # Update the objects reference if necessary
        new_edges_by_index = {}
        new_edges_by_uuid = {}

        for (old_obj1_index, old_obj2_index), edge in list(self.edges_by_index.items()):
            new_obj1_index = index_map.get(old_obj1_index)
            new_obj2_index = index_map.get(old_obj2_index)

            if new_obj1_index is not None and new_obj2_index is not None:
                new_key = (new_obj1_index, new_obj2_index)
                new_uuid_key = (self.objects[new_obj1_index]['id'], self.objects[new_obj2_index]['id'])

                if new_key in new_edges_by_index:
                    new_edges_by_index[new_key].num_detections += edge.num_detections
                else:
                    edge.obj1 = new_obj1_index  # Update the edge's internal object index reference
                    edge.obj2 = new_obj2_index
                    new_edges_by_index[new_key] = edge
                    new_edges_by_uuid[new_uuid_key] = edge

        self.edges_by_index = new_edges_by_index
        self.edges_by_uuid = new_edges_by_uuid
    
    def merge_update_indices(self, index_updates):
        """Update all edge indices based on the new mapping after merging objects."""
        updated_edges_by_index = {}
        updated_edges_by_uuid = {}

        # Iterate over current edges to update indices based on index_updates
        for (obj1_index, obj2_index), curr_edge in list(self.edges_by_index.items()):
            new_obj1_index = index_updates[obj1_index]
            new_obj2_index = index_updates[obj2_index]

            # Skip updates if either index is None (meaning the object was merged away)
            if new_obj1_index is None or new_obj2_index is None:
                continue

            # Avoid creating a loop edge where an object points to itself
            if new_obj1_index == new_obj2_index:
                print(f"LOOOPY EDGE DETECTED: {new_obj1_index} == {new_obj2_index}")
                continue

            new_key = (new_obj1_index, new_obj2_index)
            new_obj1_uuid, new_obj2_uuid = self.objects[new_obj1_index]['id'], self.objects[new_obj2_index]['id']
            new_uuid_key = (new_obj1_uuid, new_obj2_uuid)
            
            # If the edge already exists after merge, update num_detections
            if new_key in updated_edges_by_index:
                updated_edges_by_index[new_key].num_detections += curr_edge.num_detections
            else:
                # Update the edge with new indices
                curr_edge.obj1_idx = new_obj1_index
                curr_edge.obj2_idx = new_obj2_index
                updated_edges_by_index[new_key] = curr_edge
                updated_edges_by_uuid[new_uuid_key] = curr_edge

        # Update the class attributes with the modified edges
        self.edges_by_index = updated_edges_by_index
        self.edges_by_uuid = updated_edges_by_uuid
        
    def update_objects_list(self, new_objects):
        self.objects = new_objects

    def __str__(self):
        return '\n'.join([str(edge) for edge in self.edges_by_index.values()])

    def __repr__(self):
        return self.__str__()
    
    def to_serializable(self):
        s_edges = []
        for (obj1_index, obj2_index), edge in self.edges_by_index.items():
            s_edges.append({
                'obj1_index': obj1_index,
                'obj2_index': obj2_index,
                'rel_type': edge.rel_type,
                'num_detections': edge.num_detections,
                'center_distance': edge.center_distance,
                'center_diff': edge.center_diff,
                'surface_min_distance': edge.surface_min_distance,
                'surface_diff': edge.surface_diff,
                'iou': edge.iou,
                'giou': edge.giou,
                'iom': edge.iom,
            })
        
        # Serialize the object list using its existing method
        s_objects = self.objects.to_serializable()
        
        return {
            'edges': s_edges,
            'objects': s_objects
        }
        
