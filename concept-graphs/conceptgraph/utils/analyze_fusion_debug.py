'''
Counts what actually happened in a geometry-only mapping run, from the jsonl that
slam/geometric_fusion.py's FusionDebugWriter leaves under <exp_out_path>/fusion_debug/.

Two things it is for. First, how the two point-overlap directions divide the work --
how often D->O carried a merge on its own, and how often O_vis->D was the only thing
that matched a detection to an existing node. Second, and the reason to run it: which
frames show a detection swallowing an existing node (a weak match where almost none of
the detection lies on that node). Fusion merges those, correctly given the mask it was
handed, so they are the trail back to the frames where YOLO/SAM missed a small object
sitting on a large one.

Usage:
    python -m conceptgraph.utils.analyze_fusion_debug <exp_out_path> [<exp_out_path> ...]
    python -m conceptgraph.utils.analyze_fusion_debug outputs/*/concept_graphs/*/exps/*/
'''

import json
import sys
from collections import Counter
from pathlib import Path


def _load(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _pct(n, total):
    return f"{n:6d} ({100.0 * n / total:5.1f}%)" if total else f"{n:6d}     --"


def _containment(record):
    '''The weak ("how much of the node does this detection cover?") score -- the 3D
    nearest-neighbour ratio restricted to the object's points this frame's camera should
    have seen (O_vis->D). None when there was no usable view for this pair.'''
    ratio = record.get("overlap_visible_obj_to_det")
    return "n/a" if ratio is None else f"O_vis->D {ratio:.3f}"


def _strength(record):
    '''The strong ("how much of the detection is this node?") score -- the 3D
    nearest-neighbour ratio D->O.'''
    return record.get("overlap_det_to_obj", 0.0)


def _quantiles(values):
    if not values:
        return "n/a"
    s = sorted(values)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return f"min {s[0]:.3f} | p25 {q(.25):.3f} | median {q(.5):.3f} | p75 {q(.75):.3f} | max {s[-1]:.3f}"


def analyze(exp_out_path: Path):
    debug_dir = Path(exp_out_path) / "fusion_debug"
    online = _load(debug_dir / "online_fusion.jsonl")
    consolidation = _load(debug_dir / "global_consolidation.jsonl")
    if not online and not consolidation:
        print(f"\n=== {exp_out_path} ===\n  no fusion_debug logs found")
        return

    print(f"\n=== {exp_out_path} ===")

    summaries = [r for r in online if r.get("summary")]
    candidates = [r for r in online if not r.get("summary")]

    # ---- per-detection outcomes -------------------------------------------
    if summaries:
        actions = Counter(r["action"] for r in summaries)
        total = len(summaries)
        print(f"\n  detections processed: {total}")
        for action, n in actions.most_common():
            print(f"    {action:<22} {_pct(n, total)}")
        bridged = [r for r in summaries if r["n_gate_pass"] > 1]
        print(f"    {'multi-node fusion':<22} {_pct(len(bridged), total)}"
              f"  (a detection bridging {sum(r['n_gate_pass'] for r in bridged)} nodes in total)")

        # The case the containment direction was added for: nothing matched strongly,
        # so without it this detection would have spawned a duplicate node.
        weak_only = [r for r in summaries if r.get("n_strong") == 0 and r["n_gate_pass"] > 0]
        print(f"    {'saved by containment':<22} {_pct(len(weak_only), total)}"
              f"  (no strong match existed)")

        # Only the closest weak match is allowed to merge; the rest are held back. A
        # detection dropping several of them is one whose silhouette enclosed a pile of
        # separate nodes -- exactly what the rule exists to stop from collapsing.
        dropped = sum(r.get("n_weak_dropped", 0) for r in summaries)
        if dropped:
            held = [r for r in summaries if r.get("n_weak_dropped", 0) > 0]
            print(f"    {'weak matches held back':<22} {dropped} node(s) across"
                  f" {len(held)} detection(s), leaving them separate")

    # ---- the ambiguous population -----------------------------------------
    passed = [r for r in candidates if r.get("gate_class")]
    weak = [r for r in passed if r["gate_class"] == "weak"]
    merged = [r for r in candidates if r.get("merged")]
    print(f"\n  candidate matches: {len(candidates)}, cleared both gates: {len(passed)}"
          f"  (strong: {len(passed) - len(weak)}, weak only: {len(weak)})")
    print(f"  actually merged: {len(merged)}"
          f"  ({len(passed) - len(merged)} weak match(es) lost the closest-only contest)")

    # A node merged on the weak direction alone, with almost none of the detection
    # lying on it, is a detection that swallowed it -- which for a small object on a
    # large one means the detector missed the small object in that frame. Merging is
    # geometrically correct given that mask, so it is not prevented here; these rows
    # are the pointer to the frames worth fixing upstream in YOLO/SAM.
    swallowed = sorted(
        (r for r in weak if r.get("merged") and _strength(r) < 0.25),
        key=_strength,
    )
    if swallowed:
        print(f"\n  likely detector misses (weak match, <25% of the detection on the node):"
              f" {len(swallowed)}")
        print(f"    strong ratio: {_quantiles([_strength(r) for r in swallowed])}")
        print("    worst frames (a detection almost entirely NOT this node still absorbed it).")
        print("    'mask #N' is the badge in that frame's filtered_masks/ overlay,")
        print("    'obj #N' the badge in its fused_masks/ overlay:")
        for r in swallowed[:10]:
            print(f"      frame {r['frame_idx']:>5}  mask #{r.get('mask_idx')}"
                  f"  obj #{r.get('obj_num', r['obj_idx'])}"
                  f"  strong {_strength(r):.3f}"
                  f"  containment {_containment(r)}")

    reasons = Counter(r["reason"] for r in candidates)
    if reasons:
        print(f"\n  rejection reasons over all candidates:")
        for reason, n in reasons.most_common():
            print(f"    {reason:<26} {_pct(n, len(candidates))}")

    # ---- final consolidation ----------------------------------------------
    if consolidation:
        sweeps = sorted({r["sweep"] for r in consolidation})
        merged = [r for r in consolidation if r["merged"]]
        print(f"\n  final consolidation: {len(sweeps)} sweeps, "
              f"{len(consolidation)} pair evaluations, {len(merged)} merges")
        for sweep in sweeps:
            rows = [r for r in consolidation if r["sweep"] == sweep]
            print(f"    sweep {sweep}: {len(rows):5d} pairs, "
                  f"{sum(1 for r in rows if r['merged']):4d} merged, "
                  f"{sum(1 for r in rows if r['reason'] == 'deferred_to_next_sweep'):4d} deferred")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for path in argv[1:]:
        analyze(Path(path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
