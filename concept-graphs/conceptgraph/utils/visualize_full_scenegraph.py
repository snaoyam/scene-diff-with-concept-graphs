'''
Renders the final, whole-scene object graph from a completed mapping run's
obj_json/edge_json output (as opposed to scenegraph_viz.py, which renders one
frame at a time during the live run). Node colors match get_object_color() so
ids line up visually with the per-frame overlays.
'''
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from conceptgraph.utils.scenegraph_viz import get_object_color


def load_scene_graph(obj_json_path, edge_json_path):
    objects = json.load(open(obj_json_path))
    edges = json.load(open(edge_json_path))

    graph = nx.Graph()
    for obj in objects.values():
        graph.add_node(obj["id"], tag=obj["object_tag"], caption=obj.get("object_caption", ""),
                       # Objects whose recognition evidence didn't clear the thresholds are
                       # still full members of the graph -- render_full_scenegraph just
                       # outlines them differently. Absent (older obj_json) means trusted.
                       trusted=bool(obj.get("recognition_trusted", True)))
    for edge in edges.values():
        obj1_id, obj2_id = edge["object_1_id"], edge["object_2_id"]
        if graph.has_edge(obj1_id, obj2_id):
            # Same object pair reported with more than one relation type (e.g.
            # both "on top of" and "next to") -- merge instead of overwriting.
            existing = graph.edges[obj1_id, obj2_id]
            if edge["relationship"] not in existing["relationship"].split(" / "):
                existing["relationship"] += f" / {edge['relationship']}"
            existing["num_detections"] += edge["num_detections"]
        else:
            graph.add_edge(
                obj1_id, obj2_id,
                relationship=edge["relationship"], num_detections=edge["num_detections"],
            )
    return graph


def _layout_component(sub, seed, min_sep=1.5):
    '''
    Spring-layout for one connected component, then rescaled so the two
    closest nodes are always >= min_sep apart -- unlike spring_layout's
    `scale` (which fixes the *overall diagram diameter*, so per-node spacing
    shrinks as the component grows), this keeps label/marker spacing roughly
    constant no matter how many nodes are in the component.
    '''
    n = sub.number_of_nodes()
    if n == 1:
        node = next(iter(sub.nodes()))
        return {node: (0.0, 0.0)}, 1.0, 1.0

    local_pos = nx.spring_layout(sub, seed=seed, k=2.0 / n ** 0.5, iterations=500)
    nodes = list(local_pos.keys())
    coords = np.array([local_pos[n] for n in nodes])

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist = max(dists.min(), 1e-6)
    scale_factor = min(min_sep / min_dist, 10.0)  # cap to avoid blowup on near-duplicate positions
    coords = coords * scale_factor
    coords -= coords.mean(axis=0)

    w = coords[:, 0].max() - coords[:, 0].min()
    h = coords[:, 1].max() - coords[:, 1].min()
    return {node: tuple(coords[i]) for i, node in enumerate(nodes)}, w, h


def _pack_components(footprints, gap=1.0, target_aspect=1.4):
    '''
    Packs components into a grid of `n_cols` items per row (n_cols chosen so
    the overall packed area roughly matches target_aspect, e.g. a 14x10
    figure), where each row/column is sized by the actual footprints in it
    rather than a uniform cell -- so a single isolated node takes up barely
    any room while a big connected cluster gets the space it needs, and
    isolated nodes stay packed close together instead of one-per-row.
    '''
    n = len(footprints)
    n_cols = max(1, round(math.sqrt(n * target_aspect)))

    pos = {}
    y_cursor = 0.0
    for row_start in range(0, n, n_cols):
        row = footprints[row_start:row_start + n_cols]
        row_height = max(h for _, w, h in row)
        x_cursor = 0.0
        for local_pos, w, h in row:
            cx = x_cursor + w / 2.0
            for node, (x, y) in local_pos.items():
                pos[node] = (cx + x, y_cursor + y)
            x_cursor += w + gap
        y_cursor -= row_height + gap
    return pos


def _layout_by_component(graph, seed=0):
    '''
    Lays out each connected component independently and packs them together
    by actual size. Prevents a tightly interconnected cluster from collapsing
    to an unreadable point relative to far-flung isolated nodes, which plain
    global spring_layout does when node degrees are very uneven.
    '''
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    footprints = [_layout_component(graph.subgraph(comp), seed) for comp in components]
    return _pack_components(footprints)


def render_full_scenegraph(graph, save_path, title=None, seed=0):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    color_cache = {}
    node_colors = [get_object_color(n, color_cache) for n in graph.nodes()]
    # Weakly-recognized objects stay on the graph at full size and colour; only their
    # outline goes pale, so they read as "present but not something to assert a change
    # from" without disappearing (see annotate_recognition_trust).
    node_edge_colors = ["black" if graph.nodes[n].get("trusted", True) else "#cccccc"
                        for n in graph.nodes()]
    node_labels = {n: f"#{n}\n{graph.nodes[n]['tag']}" for n in graph.nodes()}
    edge_widths = [1 + 0.4 * graph.edges[e]["num_detections"] for e in graph.edges()]
    edge_labels = {e: graph.edges[e]["relationship"] for e in graph.edges()}

    pos = _layout_by_component(graph, seed=seed)

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, edge_color="gray", alpha=0.7)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=7,
                                  bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.5))
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=1000,
                            edgecolors=node_edge_colors, linewidths=1)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax, font_size=7)

    n_isolated = sum(1 for n in graph.nodes() if graph.degree(n) == 0)
    ax.set_title(title or f"scene graph: {graph.number_of_nodes()} objects, "
                           f"{graph.number_of_edges()} edges ({n_isolated} isolated)", fontsize=12)
    ax.margins(0.15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

