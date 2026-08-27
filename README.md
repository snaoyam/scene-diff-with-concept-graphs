# preparing Datasets

## 1. download scene_diff benchmark dataset
Download `https://huggingface.co/datasets/yuqun/SceneDiff/resolve/main/scenediff_benchmark.zip` and store at `~/concept-graphs-project/scene_diff/data/scenediff_benchmark`

## 2. convert to image frames(RGB-D) + camera pose Dataset
Run script `~/concept-graphs-project/scripts/datasets/scenediff_to_conceptgraph-dataset.sh`

## 3. prepare ground-truth frames
Run script `~/concept-graphs-project/scene_diff/scripts/visualize_gt_masks.py` and store output to `~/concept-graphs-project/ground-truth`

# run pipeline
Run scene change detection via ConceptGraph pipeline by running script `~/concept-graphs-project/scripts/run.sh`.
It will output in `~/concept-graphs-project/outputs`
