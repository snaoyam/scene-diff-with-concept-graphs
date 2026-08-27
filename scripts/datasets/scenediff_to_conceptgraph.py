'''
Converts SceneDiff benchmark video pairs (github.com/yuqunw/scene_diff,
huggingface.co/datasets/yuqun/SceneDiff) into ConceptGraph-ready RGB-D
sequences.

SceneDiff ships RGB video only (no depth/camera poses), so this script
reproduces the geometry-estimation stage of the SceneDiff paper's own
pipeline (modules/geometry_model.py) using Pi3 (github.com/yyfz/Pi3) to get
per-frame camera poses and depth, then writes them out in the ScanNet-style
color/depth/pose folder layout that conceptgraph.dataset.datasets_common
already knows how to load.

For each <pair_name>/original_video{1,2}.mp4 under --raw_dir, both videos
("before" / "after") are run through Pi3 *jointly* in a single forward pass,
matching how the SceneDiff paper itself estimates geometry -- this puts the
before/after reconstructions in one shared coordinate frame. Two scenes'
worth of color/depth/pose/ply are written per pair, nested under one pair
folder -- <pair_name>/before/ and <pair_name>/after/ -- each re-centered on
its own first frame at load time by ConceptGraph. See
transform_after_first_in_before_frame.txt saved alongside <pair_name>/before/
for the relative pose between the two scenes' (pre-normalization) coordinate
frames.

A dataconfig-only entry, <pair_name>.yaml (bare pair name, no before/after/
combined suffix), is also written -- just a tiny yaml, no duplicated
color/depth/pose/ply files. Pointing hydra_configs/scenediff.yaml's scene_id
at the bare "<pair_name>" makes conceptgraph.dataset.datasets_common
.SceneDiffDataset detect the sibling before/ and after/ subfolders and read
before frames immediately followed by after frames -- concatenated directly
from those two folders, no physical combined copy -- as one continuous
sequence for ConceptGraph mapping. That lets a single object tracker persist
across the before->after boundary, so an object present in both halves
naturally keeps the same curr_obj_num, instead of before/after graphs having
independently-numbered, unrelated object IDs. See
SceneDiffDataset.after_start_idx for where "after" begins in that sequence.

Usage:
    python scenediff_to_conceptgraph.py \
      --raw_dir scenediff_benchmark/data \
      --out_dataset_root /node_data/urp26su_dongwoo/concept-graphs-project/Datasets/scenediff \
      --dataconfig_root /node_data/urp26su_dongwoo/concept-graphs-project/concept-graphs/conceptgraph/dataset/dataconfigs/scenediff/ \
      --pairs coffee_table_1_coffee_table_2 \
      --extract_fps 30 \
      --resample_rate 2 \
      --save_ply
'''
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

THIRD_PARTY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THIRD_PARTY_DIR / "Pi3"))
sys.path.insert(0, str(THIRD_PARTY_DIR / "scene_diff"))

from pi3.models.pi3 import Pi3  # noqa: E402
from utils import recover_focal_shift, get_robust_voxel_size  # noqa: E402

DEPTH_SCALE = 1000.0  # matches ScanNet's png_depth_scale convention (mm stored as uint16)


def extract_frames(video_path: Path, out_dir: Path, fps: int = 30):
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.jpg")):
        return
    cmd = [
        "ffmpeg", "-loglevel", "quiet", "-i", str(video_path),
        "-q:v", "2", "-r", str(fps), "-start_number", "0",
        str(out_dir / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def load_images_as_tensor(file_list, max_size=518):
    '''Same resize convention as scene_diff/utils.py: longer side -> max_size, multiple of 14.'''
    sources = [Image.open(p).convert("RGB") for p in file_list]
    w0, h0 = sources[0].size
    if h0 > w0:
        new_h = max_size
        new_w = round(round(w0 * (max_size / h0)) / 14) * 14
    else:
        new_w = max_size
        new_h = round(round(h0 * (max_size / w0)) / 14) * 14

    tensors = []
    for im in sources:
        im = im.resize((new_w, new_h), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors, dim=0), new_h, new_w


def run_pi3(model, images, device):
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            res = model(images[None].to(device))
    return res


def compute_intrinsics(res, H, W):
    '''Per-frame focal recovery (assumes principal point at image center), then
    the caller takes the median across frames since ConceptGraph expects one
    fixed intrinsics matrix per scene.'''
    fx_list, fy_list = [], []
    N = res["local_points"].shape[1]
    for i in range(N):
        local_point_map = res["local_points"][0, i]
        mask = torch.sigmoid(res["conf"][0, i]) > 0.1
        oh, ow = local_point_map.shape[-3], local_point_map.shape[-2]
        aspect_ratio = ow / oh
        focal, _shift = recover_focal_shift(local_point_map, mask)
        fx = float(focal) * W / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = float(focal) * H / 2 * (1 + aspect_ratio ** 2) ** 0.5
        fx_list.append(fx)
        fy_list.append(fy)
    return fx_list, fy_list


def normalize_scene_scale(point_map_np, depth_map, poses, subsample_size=1_000_000, total_voxel_number=200):
    '''Same heuristic as scene_diff/modules/geometry_model.py: pick a scale so the
    scene's point cloud spans a canonical voxel grid. Pi3's raw output is only
    consistent up to an unknown scale, so this is what turns it into
    ConceptGraph-usable "metric-ish" units.'''
    voxel_size = get_robust_voxel_size(
        point_map_np.reshape(-1, 3),
        subsample_size=subsample_size,
        scale_factor=total_voxel_number,
    )
    scale = 1.0 / (voxel_size * total_voxel_number)
    depth_scaled = depth_map * scale
    poses_scaled = poses.clone()
    poses_scaled[:, :, :3, 3] = poses_scaled[:, :, :3, 3] * scale
    return depth_scaled, poses_scaled, float(scale)


def save_pointcloud_ply(ply_path: Path, points_world, colors_u8, valid_mask, frame_indices,
                         voxel_size=0.005, max_points=None):
    '''
    Dumps the Pi3 world-frame reconstruction for one scene (before or after
    split) as a single merged, colored PLY point cloud -- handy for eyeballing
    the raw geometry quality independently of the ConceptGraph mapping run.

    voxel_size default of 0.005 matches this scale by construction: normalize_scene_scale()
    picks `scale` so that the scene's own robust voxel size becomes exactly 1/200 = 0.005
    in these units (see get_robust_voxel_size(..., scale_factor=200) above), i.e. 0.005 is
    the native point-detail resolution of the reconstruction -- going coarser than that
    throws away real detail, not just noise.
    '''
    import open3d as o3d

    idx = list(frame_indices)
    pts = points_world[idx][valid_mask[idx]]
    cols = colors_u8[idx][valid_mask[idx]].astype(np.float64) / 255.0
    if len(pts) == 0:
        print(f"  [ply] no valid points for {ply_path}, skipping")
        return 0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    if max_points and len(pcd.points) > max_points:
        keep = np.random.choice(len(pcd.points), max_points, replace=False)
        pcd = pcd.select_by_index(keep)

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    return len(pcd.points)


def save_dataconfig(dataconfig_path: Path, fx, fy, H, W):
    dataconfig_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataconfig_path, "w") as f:
        yaml.safe_dump({
            "dataset_name": "scenediff",
            "camera_params": {
                "image_height": int(H),
                "image_width": int(W),
                "fx": float(fx),
                "fy": float(fy),
                "cx": W / 2.0,
                "cy": H / 2.0,
                "png_depth_scale": DEPTH_SCALE,
            },
        }, f, default_flow_style=False)


def save_scene(scene_dir: Path, dataconfig_path: Path, frame_indices, images_hwc_u8, depth_m, poses_c2w, fx, fy, H, W,
                point_map_world=None, valid_mask=None, save_ply=True, ply_voxel_size=0.005, ply_max_points=None):
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    for d in (color_dir, depth_dir, pose_dir):
        d.mkdir(parents=True, exist_ok=True)

    for out_idx, src_idx in enumerate(frame_indices):
        Image.fromarray(images_hwc_u8[src_idx]).save(color_dir / f"{out_idx}.jpg", quality=95)
        depth_mm = np.clip(np.nan_to_num(depth_m[src_idx]) * DEPTH_SCALE, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(depth_dir / f"{out_idx}.png")
        np.savetxt(pose_dir / f"{out_idx}.txt", poses_c2w[src_idx])

    if save_ply and point_map_world is not None:
        n_pts = save_pointcloud_ply(
            scene_dir / "pointcloud_pi3.ply", point_map_world, images_hwc_u8, valid_mask, frame_indices,
            voxel_size=ply_voxel_size, max_points=ply_max_points,
        )
        print(f"  [ply] saved {n_pts} points to {scene_dir / 'pointcloud_pi3.ply'}")

    save_dataconfig(dataconfig_path, fx, fy, H, W)


def process_pair(pair_dir: Path, out_dataset_root: Path, dataconfig_root: Path, model, device, resample_rate: int, extract_fps: int,
                  save_ply: bool = True, ply_voxel_size: float = 0.005, ply_max_points: int = None):
    pair_name = pair_dir.name
    print(f"\n=== Processing pair: {pair_name} ===")

    video1 = next(pair_dir.glob("original_video1.*"))
    video2 = next(pair_dir.glob("original_video2.*"))

    frames1_dir = pair_dir / "video1_frames"
    frames2_dir = pair_dir / "video2_frames"
    extract_frames(video1, frames1_dir, fps=extract_fps)
    extract_frames(video2, frames2_dir, fps=extract_fps)

    frames1 = sorted(frames1_dir.glob("*.jpg"))[::resample_rate]
    frames2 = sorted(frames2_dir.glob("*.jpg"))[::resample_rate]
    if not frames1 or not frames2:
        print(f"  Skipping {pair_name}: no frames extracted")
        return
    print(f"  before: {len(frames1)} frames, after: {len(frames2)} frames (resample_rate={resample_rate})")

    file_list = frames1 + frames2
    images, H, W = load_images_as_tensor(file_list)

    res = run_pi3(model, images, device)

    fx_list, fy_list = compute_intrinsics(res, H, W)
    fx, fy = float(np.median(fx_list)), float(np.median(fy_list))

    depth_map = res["local_points"][..., 2][..., None]  # (1, N, H, W, 1), camera-local z
    point_map_np = res["points"][0].detach().cpu().float().numpy()  # (N, H, W, 3), world frame
    depth_scaled, poses_scaled, scale = normalize_scene_scale(point_map_np, depth_map, res["camera_poses"])

    conf_mask = (torch.sigmoid(res["conf"][..., 0]) > 0.1)[0]  # (N, H, W)
    depth_np = depth_scaled[0, ..., 0].detach().cpu().float().numpy()  # (N, H, W)
    depth_np = np.where(conf_mask.cpu().numpy(), depth_np, 0.0)
    depth_np = np.where(depth_np > 0, depth_np, 0.0)  # drop negative/behind-camera depth
    valid_mask = depth_np > 0  # (N, H, W), same validity used for the saved depth images

    poses_np = poses_scaled[0].detach().cpu().float().numpy()  # (N, 4, 4), camera-to-world
    images_u8 = (images.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)  # (N, H, W, 3)
    point_map_scaled = point_map_np * scale  # world-frame points, same scale as depth_np/poses_np

    n1 = len(frames1)
    pair_dataset_dir = out_dataset_root / pair_name
    before_dir = pair_dataset_dir / "before"
    after_dir = pair_dataset_dir / "after"
    save_scene(
        before_dir, dataconfig_root / pair_name / "before.yaml",
        range(0, n1), images_u8, depth_np, poses_np, fx, fy, H, W,
        point_map_world=point_map_scaled, valid_mask=valid_mask, save_ply=save_ply, ply_voxel_size=ply_voxel_size,
        ply_max_points=ply_max_points,
    )
    save_scene(
        after_dir, dataconfig_root / pair_name / "after.yaml",
        range(n1, n1 + len(frames2)), images_u8, depth_np, poses_np, fx, fy, H, W,
        point_map_world=point_map_scaled, valid_mask=valid_mask, save_ply=save_ply, ply_voxel_size=ply_voxel_size,
        ply_max_points=ply_max_points,
    )

    # The bare "<pair>" scene_id (combined) needs a dataconfig yaml too
    # (camera_params are identical to before/after -- computed once from the
    # joint Pi3 run over all frames), but NOT its own color/depth/pose/ply
    # files: conceptgraph.dataset.datasets_common.SceneDiffDataset detects a
    # bare scene_id by finding the sibling before/after subfolders and
    # concatenates them directly, so nothing is duplicated on disk.
    save_dataconfig(dataconfig_root / f"{pair_name}.yaml", fx, fy, H, W)

    # Both scenes share this joint Pi3 run's coordinate frame before ConceptGraph
    # re-centers each on its own first frame -- save the relative transform so the
    # before/after-only scenes can be re-aligned later for change detection.
    world_to_before_first = np.linalg.inv(poses_np[0])
    after_first_in_before_frame = world_to_before_first @ poses_np[n1]
    np.savetxt(before_dir / "transform_after_first_in_before_frame.txt", after_first_in_before_frame)

    print(f"  saved: {before_dir}")
    print(f"  saved: {after_dir}")
    print(f"  combined scene_id={pair_name} uses the two folders above directly")
    print(f"  scale factor applied: {scale:.4f}, fx={fx:.1f}, fy={fy:.1f}, size={W}x{H}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Directory containing <pair_name>/original_video{1,2}.* subfolders")
    parser.add_argument("--out_dataset_root", type=str, required=True,
                         help="Where to write <pair_name>/before/, <pair_name>/after/ (color/depth/pose)")
    parser.add_argument("--dataconfig_root", type=str, required=True,
                         help="Where to write per-scene ConceptGraph dataconfig yamls")
    parser.add_argument("--pairs", type=str, nargs="*", default=None,
                         help="Subset of pair names to process (default: all subfolders of --raw_dir)")
    parser.add_argument("--resample_rate", type=int, default=10,
                         help="Take every Nth extracted frame (SceneDiff's own default is 30; lower gives ConceptGraph more views)")
    parser.add_argument("--extract_fps", type=int, default=30, help="ffmpeg frame extraction rate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_ply", action="store_true", default=True,
                         help="Also dump each scene's Pi3 reconstruction as a merged colored PLY point cloud")
    parser.add_argument("--no_save_ply", dest="save_ply", action="store_false")
    parser.add_argument("--ply_voxel_size", type=float, default=0.005,
                         help="Voxel size (in scaled scene units) used to downsample the saved PLY; "
                              "0.005 matches the reconstruction's native voxel resolution by construction "
                              "(see normalize_scene_scale/get_robust_voxel_size), 0 disables downsampling "
                              "and keeps every valid pixel from every frame (much denser, no multiview merging)")
    parser.add_argument("--ply_max_points", type=int, default=None,
                         help="Cap on saved PLY point count (random subsample if exceeded); default is unlimited")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dataset_root = Path(args.out_dataset_root)
    dataconfig_root = Path(args.dataconfig_root)

    pair_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if args.pairs:
        pair_dirs = [p for p in pair_dirs if p.name in args.pairs]

    print(f"Loading Pi3 model on {args.device}...")
    model = Pi3.from_pretrained("yyfz233/Pi3").to(args.device).eval()

    for pair_dir in pair_dirs:
        process_pair(pair_dir, out_dataset_root, dataconfig_root, model, args.device, args.resample_rate, args.extract_fps,
                     save_ply=args.save_ply, ply_voxel_size=args.ply_voxel_size, ply_max_points=args.ply_max_points)


if __name__ == "__main__":
    main()
