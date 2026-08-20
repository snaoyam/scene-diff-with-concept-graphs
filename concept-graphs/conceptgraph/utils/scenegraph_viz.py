'''
Renders two per-frame visualizations of the online scene graph being built by
rerun_realtime_mapping.py: scenegraph_viz/ has the current RGB frame with
object masks overlaid only, scenegraph_viz_with_edges/ has the same frame
with the object graph (nodes at mask centroids, edges as lines between them,
labeled with the relation type) drawn on top -- similar in style to the
"*_annotated_for_vlm_w_edges.jpg" debug images produced at detection time,
but using the final, cross-frame-merged objects/edges instead of a single
frame's raw VLM output. Each PNG is saved full-bleed at the source frame's
own resolution -- no title/margin, so no white padding around the image.
Labels near the frame border are nudged back inside the image bounds (see
_clamp_texts_to_image) so that same lack of margin doesn't cut them off.
'''
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

_DPI = 150


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


def _clamp_texts_to_image(ax, texts, w, h):
    '''Label positions are derived from mask/edge geometry (e.g. just above an
    object's bbox), which routinely puts a label right at the frame border for
    objects near an edge. With the padding-free full-bleed axes this module
    uses (see _new_fullbleed_axes), there is no surrounding margin left to
    absorb that overflow -- text drawn past (0,0)-(w,h) simply falls outside
    the saved canvas and is not rendered ("잘려서 안보이는" -- clipped/missing,
    not just visually clipped). Redraws once to get each label's actual
    rendered extent, then nudges (never resizes) any that overflow back
    inside the image bounds.'''
    if not texts:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in texts:
        bbox = t.get_window_extent(renderer)
        (dx0, dy0), (dx1, dy1) = ax.transData.inverted().transform(
            [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        x_lo, x_hi = sorted((dx0, dx1))
        y_lo, y_hi = sorted((dy0, dy1))
        dx = -x_lo if x_lo < 0 else (w - x_hi if x_hi > w else 0.0)
        dy = -y_lo if y_lo < 0 else (h - y_hi if y_hi > h else 0.0)
        if dx or dy:
            px, py = t.get_position()
            t.set_position((px + dx, py + dy))


def _new_fullbleed_axes(image_rgb):
    '''A figure/axes pair sized in inches to exactly match image_rgb's pixel
    size at _DPI, with the axes filling the whole figure -- so the saved PNG
    is the image itself with no title/margin/white padding around it. Leaves
    the x/y limits for imshow to set from the image itself (rather than
    pinning them to (0, w)/(h, 0) here) -- imshow's own extent already lands
    exactly on the pixel grid, whereas those round-number limits are half a
    pixel off from it and rasterize as a stray white edge row/column.'''
    h, w = image_rgb.shape[:2]
    fig = plt.figure(figsize=(w / _DPI, h / _DPI), dpi=_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def _draw_masks_only(ax, image_rgb, frame_objects, color_cache):
    h, w = image_rgb.shape[:2]
    ax.imshow(_blend_masks(image_rgb, frame_objects, color_cache))
    texts = []
    for obj in frame_objects:
        x1, y1, _, _ = obj["xyxy"]
        label = f"#{obj['obj_num']} {obj['class_name']}"
        texts.append(ax.text(
            x1, max(y1 - 5, 0), label,
            fontsize=7, color="white",
            bbox=dict(facecolor=get_object_color(obj["obj_num"], color_cache), alpha=0.85, pad=1.5, edgecolor="none"),
        ))
    _clamp_texts_to_image(ax, texts, w, h)


def _draw_with_edges(ax, image_rgb, frame_objects, frame_edges, color_cache):
    h, w = image_rgb.shape[:2]
    ax.imshow(_blend_masks(image_rgb, frame_objects, color_cache))

    centroids = {obj["obj_num"]: _mask_centroid(obj["mask"], obj["xyxy"]) for obj in frame_objects}
    texts = []

    for obj1_num, rel_type, obj2_num in frame_edges:
        if obj1_num not in centroids or obj2_num not in centroids:
            continue
        x1, y1 = centroids[obj1_num]
        x2, y2 = centroids[obj2_num]
        # white halo underneath a black line so it stays visible on any background
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=3, alpha=0.9, zorder=4)
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.2, alpha=0.9, zorder=5)
        texts.append(ax.text(
            (x1 + x2) / 2.0, (y1 + y2) / 2.0, rel_type,
            fontsize=6, color="black", ha="center", va="center", zorder=6,
            bbox=dict(facecolor="white", alpha=0.85, pad=1, edgecolor="none"),
        ))

    for obj in frame_objects:
        cx, cy = centroids[obj["obj_num"]]
        color = get_object_color(obj["obj_num"], color_cache)
        ax.scatter([cx], [cy], s=120, color=[color], edgecolors="black", linewidths=1, zorder=7)
        # Data-space offset (not textcoords="offset points") so the position
        # _clamp_texts_to_image reads/writes via get_position()/set_position()
        # is in the same image-pixel units as its overflow computation --
        # offset-points positions are stored in points and would be nudged by
        # the wrong amount.
        texts.append(ax.text(
            cx + 8, cy - 8, f"#{obj['obj_num']} {obj['class_name']}",
            fontsize=6, color="black", zorder=8,
            bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none"),
        ))

    _clamp_texts_to_image(ax, texts, w, h)


def render_frame_scenegraph(image_rgb, frame_objects, frame_edges, masks_save_path, edges_save_path, color_cache):
    '''
    Saves two full-bleed PNGs for one frame, no title/margin padding:
    masks_save_path gets the image with mask overlay only, edges_save_path
    gets the same image plus the object graph (nodes at mask centroids,
    edges between them) drawn on top.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image for this frame.
        frame_objects: list of dicts with keys obj_num, class_name, caption,
            mask (H, W bool ndarray), xyxy (4,) ndarray/list.
        frame_edges: list of (obj_num1, rel_type, obj_num2) tuples between
            objects that both appear in frame_objects.
        masks_save_path: output PNG path for the mask-only image.
        edges_save_path: output PNG path for the mask+graph image.
        color_cache: dict persisted across frames/calls so the same obj_num
            always gets the same color.
    '''
    masks_save_path = Path(masks_save_path)
    edges_save_path = Path(edges_save_path)
    masks_save_path.parent.mkdir(parents=True, exist_ok=True)
    edges_save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _new_fullbleed_axes(image_rgb)
    _draw_masks_only(ax, image_rgb, frame_objects, color_cache)
    fig.savefig(masks_save_path, dpi=_DPI)
    plt.close(fig)

    fig, ax = _new_fullbleed_axes(image_rgb)
    _draw_with_edges(ax, image_rgb, frame_objects, frame_edges, color_cache)
    fig.savefig(edges_save_path, dpi=_DPI)
    plt.close(fig)
