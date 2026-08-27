# Setup environment
```bash
# conda environment 생성
conda create -n conceptgraph python=3.10
conda activate conceptgraph

# Pytorch 패키지 설치
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# pointclouds indexing 및 merging 작업을 위한 Faiss library 설치
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

# Pytorch3D 빌드 URL 설치
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2

# cuda development toolkit 설치
conda install -c conda-forge cudatoolkit-dev

# 기타 필수 PyPI library 및 CLIP 설치
pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy ultralytics dill supervision open3d imageio natsort kornia rerun-sdk pyliblzfse pypng git+https://github.com/ultralytics/CLIP.git

# CUDA_HOME environment variable 설정
export CUDA_HOME=/path/to/anaconda3/envs/conceptgraph

# concept-graphs 설치
cd concept-graphs
pip install -e .
cd ..

# ram 설치
cd Grounded-Segment-Anything/recognize-anything/
pip install -e .
cd ../..

# segment-anything 설치
cd Grounded-Segment-Anything/segment_anything_gsa/
pip install -e .
cd ../..

# gradslam 설치
cd gradslam/
pip install -e .
cd ..

# llava 설치
cd LLaVA/
pip install -e .
cd ..

# groundingdino 설치
cd Grounded-Segment-Anything/GroundingDINO/
pip install -e .
cd ../..

# chamferdist 설치
cd chamferdist/
pip install -e .
cd ..

# flash-attn 설치
pip install flash-attn --no-build-isolation
```

# Preparing datasets

## 1. download scene_diff benchmark dataset
Download `https://huggingface.co/datasets/yuqun/SceneDiff/resolve/main/scenediff_benchmark.zip` and store at `~/concept-graphs-project/scene_diff/data/scenediff_benchmark`

## 2. convert to image frames(RGB-D) + camera pose Dataset
Run script `~/concept-graphs-project/scripts/datasets/scenediff_to_conceptgraph-dataset.sh`

## 3. prepare ground-truth frames
Run script `~/concept-graphs-project/scene_diff/scripts/visualize_gt_masks.py` and store output to `~/concept-graphs-project/ground-truth`

# Run pipeline
Run scene change detection via ConceptGraph pipeline by running script `~/concept-graphs-project/scripts/run.sh`.
It will output in `~/concept-graphs-project/outputs`
