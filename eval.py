import os
import torch
import sys
from scene import Scene, BetaModel
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, ViewerParams
import json


def training(args):
    beta_model = BetaModel(args.input_dim)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scene = Scene(args, beta_model)
    ply_path = os.path.join(
        args.model_path, "point_cloud", "iteration_" + args.iteration, "point_cloud.ply"
    )
    if os.path.exists(ply_path):
        print("Evaluating " + ply_path)
        beta_model.load_ply(ply_path)
        result = scene.eval()
        with open(
            os.path.join(
                scene.model_path,
                "point_cloud",
                "iteration_" + args.iteration,
                "metrics.json",
            ),
            "w",
        ) as f:
            json.dump(result, f, indent=True)
    else:
        print("No point cloud found at " + ply_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluating script parameters")
    ModelParams(parser), OptimizationParams(parser), ViewerParams(parser)
    parser.add_argument(
        "--iteration", default="30000", type=str, help="Iteration to evaluate"
    )
    args = parser.parse_args(sys.argv[1:])
    args.eval = True

    print("Evaluating " + args.model_path)

    training(args)
