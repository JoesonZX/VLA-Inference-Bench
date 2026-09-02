"""Build data/fixtures_pusht.pt: 8 fixed frames from episode 0 of lerobot/pusht.

Run on the server (smolvla env) with HF_HOME=/storage/xuan/vlabench/hf_cache.
The fixture is committed to the repo so every latency/deviation number points
at bit-identical inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lerobot/pusht")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--out", default="data/fixtures_pusht.pt")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.dataset)
    ep_index = ds.episode_data_index
    frm, to = int(ep_index["from"][args.episode]), int(ep_index["to"][args.episode])
    idxs = np.linspace(frm, to - 1, args.n_frames).round().astype(int).tolist()

    img_key = next(k for k in ds.meta.features if "image" in k)
    state_key = next(k for k in ds.meta.features if "state" in k)

    images, states = [], []
    for i in idxs:
        item = ds[i]
        img = item[img_key]  # (C, H, W) float in [0, 1]
        images.append((img.numpy() * 255).round().clip(0, 255).astype(np.uint8))
        states.append(item[state_key].numpy().astype(np.float32))

    tasks = list(ds.meta.tasks.keys()) if isinstance(ds.meta.tasks, dict) else None
    task = ds[idxs[0]]["task"] if "task" in ds[idxs[0]] else "push the t onto the cube"

    fixture = {
        "images": torch.from_numpy(np.stack(images)),
        "states": torch.from_numpy(np.stack(states)),
        "task": str(task),
        "meta": {
            "dataset": args.dataset,
            "episode": args.episode,
            "frame_indices": idxs,
            "image_key": img_key,
            "state_key": state_key,
            "image_shape": list(images[0].shape),
            "tasks_vocab": tasks,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.out)
    print(json.dumps({k: (v if not isinstance(v, torch.Tensor) else list(v.shape)) for k, v in fixture.items()}, indent=1, default=str))
    print("task:", repr(task))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
