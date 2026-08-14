# Scene pairs processed by run_construct_concept_graphs.sh (ConceptGraph mapping) and, by
# default, run_scenediff_benchmark_on_concept_graphs.sh's no-argument batch mode (SceneDiff
# benchmark scoring). Kept in one shared file, sourced by both, so the two scripts can never drift
# apart on which scenes they cover -- the same class of silent mismatch as the
# conv_resample_rate bug this pipeline used to have, just for scene selection instead of frame
# rate. Meant to be sourced, not executed.
JSON_FILE="/node_data/urp26su_dongwoo/concept-graphs-project/scene_diff/data/scenediff_benchmark/splits/val_split.json"

SCENE_PAIRS=($(python -c "import json; d=json.load(open('$JSON_FILE')); print('\n'.join(d.get('varied', []) + d.get('kitchen', [])))" | sort))
# SCENE_PAIRS=(
#     living_room_1_living_room_2
#     lounge_3_lounge_4
#     lounge_17_lounge_18
#     P09-20240621-153208_0006_P09-20240621-153208_0010
#     P09-20240621-153208_0013_P09-20240621-153208_0015
#     P09-20240621-153208_0026_P09-20240621-153208_0030
# )
# SCENE_PAIRS=($(find /node_data/urp26su_dongwoo/concept-graphs-project/Datasets/scenediff -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort))

# REVERSED_SCENE_PAIRS=()
# for (( i=${#SCENE_PAIRS[@]}-1; i>=0; i-- )); do
#     REVERSED_SCENE_PAIRS+=("${SCENE_PAIRS[$i]}")
# done

echo "SCENE_PAIRS의 전체 개수: ${#SCENE_PAIRS[@]}개"