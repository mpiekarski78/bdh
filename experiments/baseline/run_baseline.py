"""Phase 1 baseline: train BDH like upstream train.py, save checkpoint + logs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import requests
import torch

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import bdh
from experiments.common.checkpoint import save_checkpoint
from experiments.common.run_io import collect_environment, init_run, write_json, write_summary
from experiments.common.seed import set_training_seed


def fetch_data(path: Path) -> None:
    if path.exists():
        return
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path.write_text(requests.get(url, timeout=60).text)


def get_batch(data_path: Path, split: str, block_size: int, batch_size: int, device: torch.device):
    data = np.memmap(data_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    if device.type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline BDH and save checkpoint")
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda|cpu (default: cuda if available)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "base.pt"),
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dtype = (
        "bfloat16"
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else ("float32" if device.type == "cpu" else "float16")
    )
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=ptdtype)
        if device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))

    set_training_seed(args.seed)

    config = bdh.BDHConfig()
    run_dir = init_run(
        {
            "experiment": "baseline",
            "seed": args.seed,
            "max_iters": args.max_iters,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dtype": dtype,
            "bdh_config": config.__dict__,
            "compile": not args.no_compile,
        },
        prefix="baseline",
    )

    data_path = REPO_ROOT / "input.txt"
    fetch_data(data_path)

    model = bdh.BDH(config).to(device)
    compile_ok = False
    compile_error = None
    if not args.no_compile:
        try:
            model = torch.compile(model)
            compile_ok = True
        except Exception as e:
            compile_error = str(e)
            print(f"torch.compile failed, continuing uncompiled: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_log_path = run_dir / "loss.csv"
    with loss_log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])

        x, y = get_batch(data_path, "train", args.block_size, args.batch_size, device)
        loss_acc = 0.0
        loss_steps = 0
        last_loss = None
        for step in range(args.max_iters):
            with ctx:
                logits, loss = model(x, y)
            x, y = get_batch(data_path, "train", args.block_size, args.batch_size, device)
            loss_acc += float(loss.item())
            loss_steps += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if step % args.log_freq == 0 or step == args.max_iters - 1:
                avg = loss_acc / max(1, loss_steps)
                last_loss = avg
                print(f"Step: {step}/{args.max_iters} loss {avg:.4f}")
                writer.writerow([step, avg])
                f.flush()
                loss_acc = 0.0
                loss_steps = 0

    # Unwrap compiled module for clean state_dict if needed
    raw_model = model
    if hasattr(model, "_orig_mod"):
        raw_model = model._orig_mod

    ckpt_path = Path(args.checkpoint)
    save_checkpoint(
        ckpt_path,
        raw_model,
        config=config,
        meta={
            "seed": args.seed,
            "max_iters": args.max_iters,
            "final_logged_loss": last_loss,
            "dtype": dtype,
            "compile_ok": compile_ok,
        },
    )
    # Also copy reference into run dir
    save_checkpoint(
        run_dir / "checkpoints" / "base.pt",
        raw_model,
        config=config,
        meta={"seed": args.seed, "final_logged_loss": last_loss},
    )

    # Sample generation (match train.py)
    raw_model.eval()
    prompt = torch.tensor(bytearray("To be or ", "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with ctx:
            ret = raw_model.generate(prompt, max_new_tokens=100, top_k=3)
    sample = bytes(ret.to(torch.uint8).to("cpu").squeeze(0).tolist()).decode(errors="backslashreplace")

    env = collect_environment(
        {
            "dtype": dtype,
            "compile_ok": compile_ok,
            "compile_error": compile_error,
            "checkpoint": str(ckpt_path),
            "final_logged_loss": last_loss,
        }
    )
    write_json(run_dir / "environment.json", env)
    write_json(
        run_dir / "metrics.json",
        {"final_logged_loss": last_loss, "sample": sample, "checkpoint": str(ckpt_path)},
    )
    write_summary(
        run_dir,
        f"""# Baseline training

- device: {env['device']}
- torch: {env['torch']}
- commit: {env['commit']}
- seed: {args.seed}
- max_iters: {args.max_iters}
- final_logged_loss: {last_loss}
- compile_ok: {compile_ok}
- checkpoint: `{ckpt_path}`

## Sample

```
{sample}
```
""",
    )
    print("Saved checkpoint to", ckpt_path)
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()
