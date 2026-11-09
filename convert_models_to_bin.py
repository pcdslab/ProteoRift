#!/usr/bin/env python3
from pathlib import Path
import torch

for p in Path(__file__).parent.joinpath("models").glob("*.pt"):
    print("Converting", p.name)
    obj = torch.load(p, map_location="cpu")
    torch.save(obj, p.with_suffix(".bin"))