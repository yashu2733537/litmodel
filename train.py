"""Train the literary GPT on CPU.

  python train.py --preset 5m            # ~5.2M params (default)
  python train.py --preset 20m           # ~18.9M params
  python train.py --resume               # continue from out/ckpt.pt
"""
import argparse
import json
import math
import os
import pathlib
import time

import numpy as np
import torch

from config import load_config
from model import GPT

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"
CKPT = OUT / "ckpt.pt"
CKPT_BEST = OUT / "ckpt.best.pt"

torch.manual_seed(1337)

PROMPTS = [
    # The benchmark: tests question-answering shape, grammar, and literary
    # register in one frame. Same prompt every sample -> clean time-lapse.
    "OPHELIA:\nWhat is the difference between a true word and a fair one?\n\nHAMLET:\n",
]


def sample_snippet(model, tok, prompt: str, n_tokens: int = 60) -> str:
    """Generate a short continuation from a fixed prompt (deterministic seed).

    RNG discipline (2026-10-08): torch.manual_seed(42) used to reset the GLOBAL
    rng after every eval, which restarted the training batch stream identically
    each cycle -> the model trained on the same ~32 windows (~33k tokens, 0.4%
    of the corpus) for 950 iters. Population loss climbed 6.36 -> 7.84 while
    the fixed eval windows stayed low. Save/restore the rng state so sampling
    stays deterministic but is invisible to the training data stream.
    """
    was_training = model.training
    rng_state = torch.get_rng_state()
    model.eval()
    torch.manual_seed(42)  # same seed each time -> differences = model learning
    ids = tok.encode(prompt).ids[-model.cfg.block_size:]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, n_tokens, temperature=0.8, top_k=50)
    if was_training:
        model.train()
    torch.set_rng_state(rng_state)
    return tok.decode(out[0].tolist())


def get_batch(data: np.ndarray, batch_size: int, block_size: int):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg) -> dict:
    # Fixed eval windows + rng restore (2026-10-08): eval must never perturb
    # the training batch stream (the old code's side effect trained the model
    # on a repeating 32-window loop), and a constant eval set gives a smooth,
    # directly comparable curve with zero between-window sampling noise.
    rng_state = torch.get_rng_state()
    torch.manual_seed(777)
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, cfg.block_size)
            loss = model(x, y).loss
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    torch.set_rng_state(rng_state)
    return out


def lr_at(it: int, cfg) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def atomic_save(obj, path: pathlib.Path) -> None:
    """Write a checkpoint so it is never observable half-written.

    The watchdog may SIGTERM/SIGKILL at any instant and power loss can tear a
    plain write too; a torn ckpt.pt makes the next --resume raise, which turns
    the supervisor into a permanent crash-restart loop. Write to a temp file in
    the SAME directory (rename is only atomic within one filesystem), fsync it,
    then os.replace() onto the final path. Rename-after-fsync is the standard
    POSIX "atomic file update" idiom.
    """
    tmp = path.parent / f".{path.name}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # drop leftover temp on any failure path
    # fsync the directory so the rename itself survives a power cut.
    try:
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass  # best effort: rename is already atomic, only durability lags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="5m", choices=["5m", "10m", "20m"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--iters", type=int, default=None, help="override max_iters")
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--eval-iters", type=int, default=None)
    ap.add_argument("--no-sample", action="store_true", help="disable live text samples")
    args = ap.parse_args()

    cfg = load_config(args.preset)
    if args.iters:
        cfg.max_iters = args.iters
    if args.eval_interval:
        cfg.eval_interval = args.eval_interval
    if args.eval_iters:
        cfg.eval_iters = args.eval_iters

    meta = json.loads((ROOT / "data" / "meta.json").read_text())
    cfg.vocab_size = meta["vocab_size"]
    print(f"preset={args.preset}  vocab={cfg.vocab_size}  "
          f"block={cfg.block_size}  batch={cfg.batch_size}x{cfg.grad_accum}")

    train_data = np.memmap(ROOT / "data" / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    print(f"data: {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    model = GPT(cfg)
    print(f"model: {model.num_params() / 1e6:.2f}M params")

    tok = None
    if not args.no_sample:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer.json"))
        print("live text samples: ON (--no-sample to disable)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    start_iter = 0
    best_val = float("inf")
    if args.resume:
        if not CKPT.exists():
            raise SystemExit(
                f"--resume requested but {CKPT} does not exist. Restore the "
                "checkpoint from out/old/ (or another backup), or relaunch "
                "WITHOUT --resume to intentionally start a fresh run from iter 0."
            )
        try:
            ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise SystemExit(
                f"checkpoint {CKPT} is corrupt or unreadable "
                f"({type(exc).__name__}: {exc}). Recovery source: {CKPT_BEST} "
                "has valid weights but NO optimizer state, so Adam moments "
                "would reset; copy it over ckpt.pt, or restore a known-good "
                "backup, then relaunch with --resume."
            ) from exc
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        else:
            print("fresh optimizer (Adam moments reset)")
        start_iter = ck["iter"]
        best_val = ck.get("best_val", float("inf"))
        # ckpt.pt's best_val snapshot can be up to ~ckpt_interval iters stale,
        # while ckpt.best.pt is written at every new best. A stale (higher)
        # threshold would let a later, worse val loss OVERWRITE the true-best
        # insurance checkpoint -- so take the min() of both stored thresholds.
        # Missing/corrupt best file is fine: keep the ckpt.pt value.
        try:
            stored_best = torch.load(
                CKPT_BEST, map_location="cpu", weights_only=False
            ).get("best_val")
            if stored_best is not None and stored_best < best_val:
                print(f"best_val {best_val:.3f} was stale; using "
                      f"{stored_best:.3f} from {CKPT_BEST.name}")
                best_val = stored_best
        except Exception:
            pass
        print(f"resumed from iter {start_iter} (best val {best_val:.3f})")

    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    model.train()

    t0 = time.time()
    tokens_seen = 0
    eval_time = 0.0
    for it in range(start_iter, cfg.max_iters):
        lr = lr_at(it, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            x, y = get_batch(train_data, cfg.batch_size, cfg.block_size)
            loss = model(x, y).loss
            (loss / cfg.grad_accum).backward()
            tokens_seen += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            te = time.time()
            losses = estimate_loss(model, train_data, val_data, cfg)
            eval_time += time.time() - te
            dt = time.time() - t0 - eval_time
            tps = tokens_seen / dt if dt > 0 else 0
            print(f"iter {it:6d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| lr {lr:.2e} | {tps:,.0f} tok/s", flush=True)
            if tok is not None:
                prompt = PROMPTS[(it // cfg.eval_interval) % len(PROMPTS)]
                text = sample_snippet(model, tok, prompt)
                print(f"    gen  « {prompt.replace(chr(10), ' / ')}", flush=True)
                for line in text[len(prompt):].strip().splitlines() or [""]:
                    print(f"         {line}", flush=True)
                print(f"    » (seed=42, 60 tokens, temp 0.8)", flush=True)
            if losses["val"] < best_val:
                best_val = losses["val"]
                atomic_save({"model": model.state_dict(), "config": cfg.__dict__,
                             "iter": it, "best_val": best_val}, CKPT_BEST)
            t0 = time.time()
            tokens_seen = 0
            eval_time = 0.0

        if (it + 1) % cfg.ckpt_interval == 0 or it == cfg.max_iters - 1:
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "iter": it + 1, "best_val": best_val,
                         "config": cfg.__dict__}, CKPT)
            print(f"    checkpoint -> {CKPT}")

    print("training complete.")


if __name__ == "__main__":
    main()
