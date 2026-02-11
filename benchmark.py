import os
from argparse import ArgumentParser
import json
import pandas as pd

cap_max = {
    "bicycle": 6_000_000,
    "flowers": 3_000_000,
    "garden": 5_000_000,
    "stump": 4_500_000,
    "treehill": 3_500_000,
    "room": 1_500_000,
    "counter": 1_500_000,
    "kitchen": 1_500_000,
    "bonsai": 1_500_000,
    "train": 1_000_000,
    "truck": 2_500_000,
    "drjohnson": 3_500_000,
    "playroom": 2_500_000,
}

# Lists of scenes for each dataset
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
nerf_synthetic_scenes = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]
dnerf_scenes = [
    "bouncingballs",
    "hellwarrior",
    "hook",
    "jumpingjacks",
    "mutant",
    "standup",
    "trex",
]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--output_path", default="./eval")
parser.add_argument(
    "--mipnerf360", "-m360", type=str, help="Path to Mip-NeRF360 dataset"
)
parser.add_argument(
    "--nerfsynthetic", "-ns", type=str, help="Path to NeRF Synthetic dataset"
)
parser.add_argument("--dnerf", type=str, help="Path to Dynamic NeRF Synthetic dataset")
args = parser.parse_args()


# Helper function to create markdown tables and compute mean
def create_markdown_table(metrics, dataset_name):
    df = pd.DataFrame(metrics)
    cols = ["Scene", "PSNR", "SSIM", "LPIPS"]
    df = df[cols]

    # Compute mean row
    mean_row = df.drop("Scene", axis=1).mean()
    mean_row["Scene"] = "Mean"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Generate markdown table
    md_table = (
        f"## Metrics for {dataset_name}\n"
        + df.to_markdown(index=False, floatfmt=".4f")
        + "\n\n"
    )
    return md_table


# Process Deep Blending dataset if provided
if args.nerfsynthetic:
    nf_metrics = []
    for scene in nerf_synthetic_scenes:
        source = os.path.join(args.nerfsynthetic, scene)
        os.system(
            f"python train.py -s {source} -m {args.output_path}/{scene} --noise_lr 1000000 --cap_max 300000 --eval --disable_viewer --quiet"
        )

    for scene in nerf_synthetic_scenes:
        scene_path = os.path.join(args.output_path, scene)
        results_file = os.path.join(
            scene_path, "point_cloud/iteration_30000/metrics.json"
        )
        with open(results_file, "r") as f:
            scene_metrics = json.load(f)
        scene_metrics["Scene"] = scene
        nf_metrics.append(scene_metrics)

    output_text = create_markdown_table(nf_metrics, "NeRF Synthetic")
    with open(os.path.join(args.output_path, "nerf_synthetic_metrics.txt"), "w") as f:
        f.write(output_text)


# Processing Dynamic NeRF dataset if provided
if args.dnerf:
    dnerf_metrics = []
    for scene in dnerf_scenes:
        source = os.path.join(args.dnerf, scene)
        os.system(
            f"python train.py -s {source} -m {args.output_path}/{scene} -r 2 --noise_lr 1.0 --cap_max 150000 --eval --input_dim 7 --batch_size 1 --disable_viewer"
        )
    for scene in dnerf_scenes:
        scene_path = os.path.join(args.output_path, scene)
        results_file = os.path.join(
            scene_path, "point_cloud/iteration_30000/metrics.json"
        )
        with open(results_file, "r") as f:
            scene_metrics = json.load(f)
        scene_metrics["Scene"] = scene
        dnerf_metrics.append(scene_metrics)

    output_text = create_markdown_table(dnerf_metrics, "Dynamic NeRF")
    with open(os.path.join(args.output_path, "dnerf_metrics.txt"), "w") as f:
        f.write(output_text)


# Process Mip-NeRF360 dataset if provided
if args.mipnerf360:
    mip_metrics = []
    # Process outdoor scenes
    for scene in mipnerf360_outdoor_scenes:
        source = os.path.join(args.mipnerf360, scene)
        os.system(
            f"python train.py -s {source} -r 4 -m {args.output_path}/{scene} --noise_lr 1000000 --cap_max {cap_max[scene]} --eval --disable_viewer --quiet"
        )
    # Process indoor scenes
    for scene in mipnerf360_indoor_scenes:
        source = os.path.join(args.mipnerf360, scene)
        os.system(
            f"python train.py -s {source} -r 2 -m {args.output_path}/{scene} --noise_lr 1000000 --cap_max {cap_max[scene]} --eval --disable_viewer --quiet"
        )

    # Collect metrics for Mip-NeRF360 scenes
    all_mip_scenes = mipnerf360_outdoor_scenes + mipnerf360_indoor_scenes
    for scene in all_mip_scenes:
        scene_path = os.path.join(args.output_path, scene)
        results_file = os.path.join(
            scene_path, "point_cloud/iteration_30000/metrics.json"
        )
        with open(results_file, "r") as f:
            scene_metrics = json.load(f)
        scene_metrics["Scene"] = scene
        mip_metrics.append(scene_metrics)

    # Create markdown table and save to file
    output_text = create_markdown_table(mip_metrics, "Mip-NeRF 360")
    with open(os.path.join(args.output_path, "mipnerf360_metrics.txt"), "w") as f:
        f.write(output_text)
