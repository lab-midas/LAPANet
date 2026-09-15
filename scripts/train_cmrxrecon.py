#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Aya

script for Local All Pass Attention Network (LAPANet)
Developed at the University Hospital of Tübingen.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import yaml

# Resolve project root dynamically based on script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent#.parent
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from src.train.trainer import LAPANet2DTrainer


def parse_args():
    default_config = PROJECT_ROOT / "configs" / "train_cmrxrecon.yaml"

    parser = argparse.ArgumentParser(description="Train LAPANet on CMRxRecon")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data_dir", type=str, help="Override data directory path"
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "debug"], help="Training mode"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Override batch size"
    )
    parser.add_argument(
        "--num_gpu", type=str, help="GPU device ID for debug mode"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration Setup
# ---------------------------------------------------------------------------

cli_args = parse_args()
config_path = Path(cli_args.config).resolve()

with open(config_path, "r") as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

# Override YAML values with CLI arguments if passed
for key, value in vars(cli_args).items():
    if value is not None and key != "config":
        config_dict[key] = value

args = json.loads(
    json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d)
)


# ---------------------------------------------------------------------------
# Reproducibility & Execution
# ---------------------------------------------------------------------------

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed_all(1234)
torch.cuda.empty_cache()

if getattr(args, "mode", None) == "debug":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)

trainer = LAPANet2DTrainer(args)
trainer.run()