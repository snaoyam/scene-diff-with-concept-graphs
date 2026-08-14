JSON_FILE="/node_data/urp26su_dongwoo/concept-graphs-project/scene_diff/data/scenediff_benchmark/splits/val_split.json"

SCENE_PAIRS=($(python -c "import json; d=json.load(open('$JSON_FILE')); print('\n'.join(d.get('varied', []) + d.get('kitchen', [])))" | sort))

# SCENE_PAIRS=(
#   living_room_17_living_room_18
# )

# SCENE_PAIRS=("${SCENE_PAIRS[@]:0}")

SUB_SCENE_PAIRS=()
for (( i=0; i<${#SCENE_PAIRS[@]}; i+=5 )); do
    SUB_SCENE_PAIRS+=("${SCENE_PAIRS[i]}")
done
SCENE_PAIRS=("${SUB_SCENE_PAIRS[@]}")

echo "SCENE_PAIRS의 전체 개수: ${#SCENE_PAIRS[@]}개"