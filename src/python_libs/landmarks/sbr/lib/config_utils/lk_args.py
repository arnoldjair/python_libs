# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import random
import sys
import time


def obtain_args(colab=False):
    parser = argparse.ArgumentParser(
        description="Train facial landmark detectors on 300-W, AFLW or Mugsy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_lists",
        type=str,
        nargs="+",
        help="The list file path to the video training dataset.",
    )
    parser.add_argument(
        "--eval_vlists",
        type=str,
        nargs="+",
        help="The list file path to the video testing dataset.",
    )
    parser.add_argument(
        "--eval_ilists",
        type=str,
        nargs="+",
        help="The list file path to the image testing dataset.",
    )
    parser.add_argument("--num_pts", type=int, help="Number of point.")
    parser.add_argument(
        "--model_config", type=str, help="The path to the model configuration"
    )
    parser.add_argument(
        "--opt_config", type=str, help="The path to the optimizer configuration"
    )
    parser.add_argument(
        "--lk_config", type=str, help="The path to the LK configuration"
    )
    # Data Generation
    parser.add_argument(
        "--heatmap_type",
        type=str,
        choices=["gaussian", "laplacian"],
        help="The method for generating the heatmap.",
    )
    parser.add_argument(
        "--data_indicator", type=str, default="300W-68", help="The dataset indicator."
    )
    parser.add_argument("--video_parser", type=str, help="The video-parser indicator.")
    # Data Transform
    parser.add_argument(
        "--pre_crop_expand", type=float, help="parameters for pre-crop expand ratio"
    )
    parser.add_argument("--sigma", type=float, help="sigma distance for CPM.")
    parser.add_argument("--scale_prob", type=float, help="argument scale probability.")
    parser.add_argument(
        "--scale_min", type=float, help="argument scale : minimum scale factor."
    )
    parser.add_argument(
        "--scale_max", type=float, help="argument scale : maximum scale factor."
    )
    parser.add_argument(
        "--scale_eval", type=float, help="argument scale : maximum scale factor."
    )
    parser.add_argument(
        "--rotate_max", type=int, help="argument rotate : maximum rotate degree."
    )
    parser.add_argument(
        "--crop_height", type=int, default=256, help="argument crop : crop height."
    )
    parser.add_argument(
        "--crop_width", type=int, default=256, help="argument crop : crop width."
    )
    parser.add_argument(
        "--crop_perturb_max",
        type=int,
        help="argument crop : center of maximum perturb distance.",
    )
    parser.add_argument(
        "--arg_flip", action="store_true", help="Using flip data argumentation or not "
    )
    # Optimization options
    parser.add_argument(
        "--eval_once", action="store_true", help="evaluation only once for evaluation "
    )
    parser.add_argument(
        "--error_bar",
        type=float,
        help="For drawing the image with large distance error.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training."
    )
    # Checkpoints
    parser.add_argument(
        "--print_freq", type=int, default=100, help="print frequency (default: 200)"
    )
    parser.add_argument(
        "--init_model", type=str, help="The detector model to be initalized."
    )
    parser.add_argument(
        "--save_path", type=str, help="Folder to save checkpoints and log."
    )
    # Acceleration
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of data loading workers (default: 2)",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, help="manual seed")

    list_base = "/content/drive/MyDrive/Datasets/landmark-datasets/300W-Style/lists/300W/Original/"

    custom_args = [
        "--train_list",
        list_base + "300w.train.GTB",
        "--eval_ilists",
        list_base + "300w.test.common.GTB",
        list_base + "300w.test.challenge.GTB",
        list_base + "300w.test.full.GTB",
        "--num_pts",
        "68",
        "--model_config",
        "/content/code/SBR/configs/Detector.config",
        "--opt_config",
        "/content/code/SBR/configs/LK.SGD.config",
        "--lk_config",
        "/content/code/SBR/configs/mix.lk.config",
        "--video_parser",
        "x-1-1",
        "--save_path",
        "/content/drive/MyDrive/evals/SBR",
        "--init_model",
        "/content/drive/MyDrive/models/SBR/cpm_vgg16-epoch-049-050.pth",
        "--pre_crop_expand",
        "0.2",
        "--sigma",
        "4",
        "--batch_size",
        "8",
        "--crop_perturb_max",
        "5",
        "--scale_prob",
        "1",
        "--scale_min",
        "1",
        "--scale_max",
        "1",
        "--scale_eval",
        "1",
        "--heatmap_type",
        "gaussian",
        "--print_freq",
        "10",
        "--eval_once",
    ]

    if colab == True:
        args = parser.parse_args(custom_args)
    else:
        args = parser.parse_args()

    if args.rand_seed is None:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_path is not None, "save-path argument can not be None"

    # state = {k: v for k, v in args._get_kwargs()}
    # Arguments = namedtuple('Arguments', ' '.join(state.keys()))
    # arguments = Arguments(**state)
    return args
