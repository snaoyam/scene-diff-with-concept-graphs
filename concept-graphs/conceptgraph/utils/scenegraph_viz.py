'''
Renders a per-frame visualization of the online scene graph being built by
rerun_realtime_mapping.py: the current RGB frame with object masks overlaid
on the left ("before"), and the same frame with the object graph (nodes at
mask centroids, edges as lines between them, labeled with the relation type)
drawn directly on top on the right -- similar in style to the
"*_annotated_for_vlm_w_edges.jpg" debug images produced at detection time,
but using the final, cross-frame-merged objects/edges instead of a single
frame's raw VLM output.
'''
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi


def get_object_color(obj_num: int, color_cache: dict) -> tuple:
    '''
    Returns a persistent RGB (0-1 float) color for a given object id (curr_obj_num).
    The same obj_num always maps to the same color, across frames and across calls,
    as long as the same color_cache dict is reused.
    '''
    if obj_num not in color_cache:
        cmap = plt.get_cmap("tab20")
        color_cache[obj_num] = cmap(len(color_cache) % 20)[:3]
    return color_cache[obj_num]


def _mask_centroid(mask, xyxy):
    ys, xs = np.nonzero(mask)
    if ys.size > 0 and xs.size > 0:
        y, x = ndi.center_of_mass(mask)
        return float(x), float(y)
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _blend_masks(image_rgb, frame_objects, color_cache, alpha=0.45):
    overlay = image_rgb.astype(np.float32).copy()
    for obj in frame_objects:
        color = np.array(get_object_color(obj["obj_num"], color_cache)) * 255.0
        overlay[obj["mask"]] = overlay[obj["mask"]] * (1 - alpha) + color * alpha
    return overlay.astype(np.uint8)


def _draw_before_panel(ax, image_rgb, frame_objects, color_cache):
    ax.imshow(_blend_masks(image_rgb, frame_objects, color_cache))
    ax.axis("off")
    for obj in frame_objects:
        x1, y1, _, _ = obj["xyxy"]
        label = f"#{obj['obj_num']} {obj['class_name']}"
        ax.text(
            x1, max(y1 - 5, 0), label,
            fontsize=7, color="white",
            bbox=dict(facecolor=get_object_color(obj["obj_num"], color_cache), alpha=0.85, pad=1.5, edgecolor="none"),
        )
    ax.set_title(f"frame objects: {len(frame_objects)}", fontsize=10)


def _draw_graph_overlay_panel(ax, image_rgb, frame_objects, frame_edges, color_cache):
    ax.imshow(_blend_masks(image_rgb, frame_objects, color_cache))
    ax.axis("off")

    centroids = {obj["obj_num"]: _mask_centroid(obj["mask"], obj["xyxy"]) for obj in frame_objects}

    for obj1_num, rel_type, obj2_num in frame_edges:
        if obj1_num not in centroids or obj2_num not in centroids:
            continue
        x1, y1 = centroids[obj1_num]
        x2, y2 = centroids[obj2_num]
        # white halo underneath a black line so it stays visible on any background
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=3, alpha=0.9, zorder=4)
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.2, alpha=0.9, zorder=5)
        ax.text(
            (x1 + x2) / 2.0, (y1 + y2) / 2.0, rel_type,
            fontsize=6, color="black", ha="center", va="center", zorder=6,
            bbox=dict(facecolor="white", alpha=0.85, pad=1, edgecolor="none"),
        )

    for obj in frame_objects:
        cx, cy = centroids[obj["obj_num"]]
        color = get_object_color(obj["obj_num"], color_cache)
        ax.scatter([cx], [cy], s=120, color=[color], edgecolors="black", linewidths=1, zorder=7)
        ax.annotate(
            f"#{obj['obj_num']} {obj['class_name']}", (cx, cy),
            textcoords="offset points", xytext=(5, 5),
            fontsize=6, color="black", zorder=8,
            bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none"),
        )

    ax.set_title(f"frame edges: {len(frame_edges)}", fontsize=10)


def render_frame_scenegraph(image_rgb, frame_objects, frame_edges, save_path, color_cache):
    '''
    Saves a side-by-side PNG for one frame: left is the image with mask
    overlay only ("before"), right is the same image with the object graph
    (nodes at mask centroids, edges between them) drawn on top.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image for this frame.
        frame_objects: list of dicts with keys obj_num, class_name, caption,
            mask (H, W bool ndarray), xyxy (4,) ndarray/list.
        frame_edges: list of (obj_num1, rel_type, obj_num2) tuples between
            objects that both appear in frame_objects.
        save_path: output PNG path.
        color_cache: dict persisted across frames/calls so the same obj_num
            always gets the same color.
    '''
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_before, ax_graph) = plt.subplots(1, 2, figsize=(16, 8))
    _draw_before_panel(ax_before, image_rgb, frame_objects, color_cache)
    _draw_graph_overlay_panel(ax_graph, image_rgb, frame_objects, frame_edges, color_cache)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
