#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Aya

script for Local All Pass Attention Network (LAPANet)

Developed at the University Hospital of Tübingen.
Copyright © 2026 University Hospital of Tübingen.

If you'd like to use or share this code, please get in touch with
Aya Ghoul <aya.ghoul@med.uni-tuebingen.de>.
"""

from pathlib import Path
import os
import json
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from types import SimpleNamespace
import yaml

from train.trainer import LAPANet2DTrainer



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config_files" / "train_cmrxrecon.yaml"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed_all(1234)
torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

args = json.loads(json.dumps(config), object_hook=lambda d: SimpleNamespace(**d))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if args.mode == "debug":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu

trainer = LAPANet2DTrainer(args)
trainer.run()