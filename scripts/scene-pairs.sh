JSON_FILE="/node_data/urp26su_dongwoo/concept-graphs-project/Datasets/scenediff/val_split.json"

SCENE_PAIRS=($(python -c "import json; d=json.load(open('$JSON_FILE')); print('\n'.join(d.get('varied', []) + d.get('kitchen', [])))" | sort))

SCENE_PAIRS=(
  lounge_3_lounge_4

  # P09-20240621-093545_0041_P09-20240621-093545_0044
  # living_room_17_living_room_18
  # table_1_table_2
  # bathroom_1_bathroom_2
  # bedroom_1_bedroom_2
  # kitchen_6_kitchen_7
  # office_13_office_14
  # hallway_1_hallway_2
  # airhockeytable_1_airhockeytable_2
  # # gas_station_3_gas_station_4
  # gym_9_gym_10
  # laundry_room_3_laundry_room_4
  # lounge_17_lounge_18
  # mailroom_1_mailroom_2
  # P09-20240621-093545_0001_P09-20240621-093545_0004
  # P09-20240621-093545_0088_P09-20240621-093545_0090
  # P09-20240622-150155_0028_P09-20240622-150155_0034
  # room_entrance_3_room_entrance_4
  # storage_1_storage_2
  # store_11_store_12
  # street_3_street_4
)

SCENE_PAIRS=( "${SCENE_PAIRS[@]}" )

SUB_SCENE_PAIRS=()
for offset in 0 1; do
    for (( i=offset; i<${#SCENE_PAIRS[@]}; i+=2 )); do
        SUB_SCENE_PAIRS+=("${SCENE_PAIRS[i]}")
    done
done
SCENE_PAIRS=("${SUB_SCENE_PAIRS[@]}")

echo "SCENE_PAIRS의 전체 개수: ${#SCENE_PAIRS[@]}개"