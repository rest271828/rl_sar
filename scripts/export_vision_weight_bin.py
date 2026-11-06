#!/usr/bin/env python3
"""Export go2_ep_student vision_weight.pt tensors to a C++-readable binary.

Format (little-endian):
  magic b'EPWT' | version u32=1 | n_tensors u32
  repeated:
    name_len u32 | name utf8 | ndim u32 | dims[ndim] u32 | float32 data[product(dims)]

Keys match the C++ DepthEncoder (base_backbone.image_compression.* etc.).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch


NEEDED = [
    "base_backbone.image_compression.0.weight",
    "base_backbone.image_compression.0.bias",
    "base_backbone.image_compression.3.weight",
    "base_backbone.image_compression.3.bias",
    "base_backbone.image_compression.6.weight",
    "base_backbone.image_compression.6.bias",
    "base_backbone.image_compression.8.weight",
    "base_backbone.image_compression.8.bias",
    "combination_mlp.0.weight",
    "combination_mlp.0.bias",
    "combination_mlp.2.weight",
    "combination_mlp.2.bias",
    "rnn.weight_ih_l0",
    "rnn.weight_hh_l0",
    "rnn.bias_ih_l0",
    "rnn.bias_hh_l0",
    "output_mlp.0.weight",
    "output_mlp.0.bias",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default=str(
            Path(__file__).resolve().parents[1]
            / "policy/go2/go2_ep_student/vision_weight.pt"
        ),
    )
    ap.add_argument(
        "--dst",
        default=str(
            Path(__file__).resolve().parents[1]
            / "policy/go2/go2_ep_student/vision_weight.bin"
        ),
    )
    args = ap.parse_args()

    blob = torch.load(args.src, map_location="cpu")
    if isinstance(blob, dict) and "depth_encoder_state_dict" in blob:
        sd = blob["depth_encoder_state_dict"]
    else:
        sd = blob

    missing = [k for k in NEEDED if k not in sd]
    if missing:
        raise SystemExit(f"missing keys: {missing}")

    with open(args.dst, "wb") as f:
        f.write(b"EPWT")
        f.write(struct.pack("<II", 1, len(NEEDED)))
        for name in NEEDED:
            t = sd[name].detach().to(torch.float32).contiguous().cpu()
            name_b = name.encode("utf-8")
            dims = list(t.shape)
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", len(dims)))
            f.write(struct.pack("<" + "I" * len(dims), *dims))
            f.write(t.numpy().tobytes(order="C"))
            print(f"wrote {name} {tuple(dims)}")

    print(f"OK -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
