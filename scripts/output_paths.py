"""
Shared helper for scripts/convert_concept_graphs_to_scene_diff_benchmark_data.py and
scripts/run_scene_diff_benchmark.py: reads output_root/exp_suffix from
conceptgraph/hydra_configs/rerun_realtime_mapping.yaml -- the single source of truth
for where pipeline outputs live, so step 2/3 never need their own --*_root CLI flags --
and builds the outputs/<pair>/{concept_graphs,benchmark_data,benchmark_result} layout
conceptgraph/slam/rerun_realtime_mapping.py's run_mapping_for_scene() writes into.

These scripts run under the scene_diff conda env, which has pyyaml but not omegaconf.
The yaml only uses flat top-level ${key} string interpolation (no nested/list
interpolation, no OmegaConf-specific features), so a small hand-rolled resolver here is
enough -- it isn't a general OmegaConf replacement.
"""
import re
from pathlib import Path

import yaml

RERUN_MAPPING_CONFIG_PATH = Path(
    "/node_data/urp26su_dongwoo/concept-graphs-project/concept-graphs/"
    "conceptgraph/hydra_configs/rerun_realtime_mapping.yaml"
)
_INTERP_RE = re.compile(r"\$\{(\w+)\}")


def _resolve(key: str, raw: dict, cache: dict, stack=()):
    if key in cache:
        return cache[key]
    if key in stack:
        raise ValueError(
            f"Circular interpolation in {RERUN_MAPPING_CONFIG_PATH}: "
            f"{' -> '.join(stack + (key,))}"
        )
    if key not in raw:
        raise KeyError(f"'{key}' not found in {RERUN_MAPPING_CONFIG_PATH}")

    value = raw[key]
    if isinstance(value, str):
        def sub(match):
            return str(_resolve(match.group(1), raw, cache, stack + (key,)))
        value = _INTERP_RE.sub(sub, value)
    cache[key] = value
    return value


def load_config() -> dict:
    """rerun_realtime_mapping.yaml's top-level scalar values, with ${key} interpolations
    resolved. Non-scalar entries (defaults:, hydra:, bg_classes list, ...) are skipped --
    callers of this module only need plain path/string values."""
    with open(RERUN_MAPPING_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    cache = {}
    return {
        key: _resolve(key, raw, cache)
        for key, value in raw.items()
        if not isinstance(value, (dict, list))
    }


def get_output_root() -> Path:
    return Path(load_config()["output_root"]).resolve()


def get_exp_suffix() -> str:
    return load_config()["exp_suffix"]


def scene_root(pair_name: str, output_root: Path = None) -> Path:
    return (output_root or get_output_root()) / pair_name


def concept_graphs_dir(pair_name: str, output_root: Path = None) -> Path:
    return scene_root(pair_name, output_root) / "concept_graphs"


def benchmark_data_dir(pair_name: str, output_root: Path = None) -> Path:
    return scene_root(pair_name, output_root) / "benchmark_data"


def benchmark_result_dir(pair_name: str, output_root: Path = None) -> Path:
    return scene_root(pair_name, output_root) / "benchmark_result"
