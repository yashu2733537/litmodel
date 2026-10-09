> **STATUS: INCOMPLETE / WIP (2026-10-09)** — pipeline works end-to-end
> (fetch → tokenize → train → watchdog → sample/`--talk`). First full run on
> the 10M preset **converged at val 5.162**; that's the floor for 9.97M params
> on 7.7M tokens (a val under 3 needs the next phase: GPU + ~100-300M params +
> billions of tokens). Checkpoints, venv, raw corpus and generated .bin files
> are excluded (see `.gitignore`). Full history + post-mortems in
> `PROJECT_MEMORY.md`.

# litmodel — a ~5M param literary GPT you can train on a laptop

Trains on Shakespeare, Dostoevsky, Tolstoy, and other public-domain classics.
Built for CPU-only machines (2 cores, 3.2 GB RAM).

## Setup (already done if you're in this repo)

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install tokenizers numpy
```

## 1. Get the corpus

```bash
.venv/bin/python data.py fetch     # ~30 Gutenberg classics -> data/raw/
.venv/bin/python data.py prepare   # train 4k BPE tokenizer + encode -> data/*.bin
```

**Camus (still in copyright):** drop your own copy as
`data/raw/custom/camus.txt` before `prepare` and it is included automatically.

## 2. Train

```bash
.venv/bin/python train.py                    # 5m preset (~5.2M params), default
.venv/bin/python train.py --preset 20m       # bigger (~18.9M), more RAM + slower
.venv/bin/python train.py --resume           # continue from out/ckpt.pt
.venv/bin/python train.py --iters 500        # short run
```

- Saves `out/ckpt.pt` every 500 iters, `out/ckpt.best.pt` on best val loss.
- Prints `tok/s` so you can gauge speed; expect roughly **1–3 hours per epoch**
  on the 5m preset depending on background load.
- Ctrl-C is safe: checkpoints are periodic, then `--resume`.

## 3. Sample

```bash
.venv/bin/python sample.py                                    # random default prompt
.venv/bin/python sample.py --prompt "To be or not to be," --tokens 600 --temp 0.7
.venv/bin/python sample.py --ckpt out/ckpt.best.pt --seed 3
```

## Knobs

- `config.py` — presets: context 256, batch 8×2 grad-accum (5m) / 4×4 (20m),
  lr 3e-4 cosine, dropout 0.1. Edit `PRESETS` to change size yourself
  (`n_embd` must be divisible by `n_head`).
- `data.py` `WORKS` — add/remove books (any Project Gutenberg title).
- RAM budget: 5m ≈ 100 MB training state, 20m ≈ 300 MB + activations.

## Layout

```
config.py   size presets + hyperparams
model.py    GPT (pre-LN, weight-tied, SDPA attention)
data.py     fetch / tokenizer / encode
train.py    training loop, cosine LR, checkpoints
sample.py   generation
data/       raw texts, tokenizer.json, train.bin / val.bin
out/        checkpoints
```
