"""
train.py - optional TuSimple fine-tune for mode='dl'.

Usage:
    python algorithms/lane_detection/train.py --tusimple-root datasets/tusimple

Requires the TuSimple dataset in the standard layout:
    datasets/tusimple/
        label_data_0313.json     (train)
        label_data_0531.json
        label_data_0601.json
        label_data_*.json        (val / test)
        clips/{some_clip}/{some_frame}.jpg

If you don't have TuSimple this script will simply error out with a clear
message - the classical mode works without any dataset.
"""

import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import config
from .model import UNetLane
from .tusimple import TuSimpleDataset


def main():
    p = argparse.ArgumentParser(description="Train DL lane model on TuSimple")
    p.add_argument("--tusimple-root", type=Path, default=config.TUSIMPLE_DIR)
    p.add_argument("--epochs",   type=int, default=10)
    p.add_argument("--batch",    type=int, default=8)
    p.add_argument("--lr",       type=float, default=1e-3)
    args = p.parse_args()

    train_json = args.tusimple_root / "label_data_0313.json"
    clips_root = args.tusimple_root / "clips"
    if not train_json.exists():
        raise SystemExit(f"TuSimple labels not found at {train_json}. "
                          "See module README for the expected layout.")
    config.MODEL_OUT.mkdir(parents=True, exist_ok=True)
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    ds = TuSimpleDataset(train_json, clips_root)
    print(f"TuSimple: {len(ds)} frames")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    device = config.get_device()
    model  = UNetLane().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    for ep in range(args.epochs):
        t0 = time.perf_counter(); loss_sum = 0; n_batches = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward(); opt.step()
            loss_sum += loss.item(); n_batches += 1
        print(f"epoch {ep+1}/{args.epochs}  mean loss {loss_sum/max(n_batches,1):.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")
    torch.save(model.state_dict(), config.MODEL_OUT / "lane_unet.pt")
    print("Saved weights ->", config.MODEL_OUT / "lane_unet.pt")


if __name__ == "__main__":
    main()
