# PROJECT MEMORY — litmodel (literary GPT)
# Full knowledge base. Linked from ~/MEMORY.md. Updated 2026-10-07.

## What this project is
Train a small GPT (~10M params) on 27 public-domain books + user's Camus PDFs,
on a weak CPU-only laptop, with live text samples during training.

- **Path**: `~/litmodel` — venv: `.venv` (python 3.12, torch 2.14.1+cpu, tokenizers, numpy, rapidocr-onnxruntime)
- **Hardware**: AMD Athlon Silver 3050U, 2 cores, 3.2GB RAM, no GPU, no passwordless sudo.
  Measured throughput: **~700 tok/s** (5m/ctx256), **~115 tok/s** (10m/ctx1024+checkpointing, 2 threads best).
  Matmul ceiling ~30 GFLOPS 1-thread / 35 2-thread (15W chip, poor scaling — normal).

## Architecture (all files in repo root)
- `model.py` — decoder-only GPT: pre-LN blocks, weight-tied lm_head, SDPA with `is_causal=True`
  (causal mask built into kernel, no mask tensor stored). Optional activation checkpointing
  (`cfg.checkpoint`, training-only) — cut 10m peak RSS 1.86GB -> **640MB**.
- `config.py` — presets (block_size must match trained wpe; changing it = full restart):
  - **5m**: 5.20M, d240, 6L, 4h, ctx256, batch8x2. (retired, was first run)
  - **10m** (ACTIVE): 9.97M, d256, 11L, 8h, ctx1024, batch1x8accum (8192 tok/iter),
    checkpoint=True, max_iters=1900 (~2 epochs), warmup30, eval every 4 iters (eval_iters=1),
    ckpt every 50 iters, lr cosine 3e-4 -> 3e-5, dropout 0.1, AdamW(0.9/0.95, wd 0.1), grad clip 1.0.
  - **20m**: 18.8M, d448, 7L, 7h, ctx256.
- `data.py` — `fetch` (Gutenberg scrape) / `prepare` (tokenizer reuse + encode -> bins).
- `train.py` — prints loss+tok/s+**benchmark gen sample every ~5.3 min** (seed 42, 60 tokens).
  Saves `out/ckpt.pt` every 50 iters, `out/ckpt.best.pt` on best val. `--resume` re-reads
  config from config.py (ckpt only restores weights/optimizer/iter/best_val).
- `sample.py` — one-shot generation. **Talk mode**: `--talk "question"
  --voice dostoevsky|camus|shakespeare` frames Q as scene dialogue so completion = in-character answer.
- `ocr_fall.py` — rapidocr fallback for scanned PDFs.

## Data
- Tokenizer: byte-level BPE vocab **4096**, trained on corpus, **FROZEN** (`data/tokenizer.json`).
  3.24 chars/token. `data/meta.json` has vocab/tokens; train.py reads it.
- Corpus: **27 files, 25.5M chars, 7,876,551 tokens** (train 7,719,019 / val 157,532).
  Shakespeare complete(pg100)+3 plays, Dostoevsky x7, Tolstoy x2, Twain, Melville, Austen x2,
  Dickens x2, Wells x2, Poe, Chekhov, **Camus x3** (user's PDFs).
- Camus source: `~/Documents/WORK OF ALBER CAMUS/` — Stranger + Sisyphus via pdftotext,
  The Fall = scanned, OCR'd via rapidocr (166KB, 29.7k words, minor missing-space artifacts).
- Drop-in folder: `data/raw/custom/*.txt` -> included on next `prepare`.
- **NOT on Project Gutenberg** (verified against full author catalogs): The Double,
  The Death of Ivan Ilyich. User may supply own copies.

## Rules / hard-won gotchas
1. **NEVER run `prepare` while training runs** — it rewrites train.bin under the memmap (SIGBUS).
2. Restart semantics: no `--resume` = fresh model; `--resume` = continue from last ckpt
   (loses progress since last save, max ~59 min). Config always read fresh from config.py.
3. Eval noise: eval_iters=1 => single random 1024-tok batch => ±0.15 val wobble is NORMAL.
   Overfitting = sustained rise over 5+ evals while train falls.
4. ~~Val targets: 2.0-2.3 = win for 10M on literary text~~ **RETRACTED
   2026-10-08** — see RUN RESULT at bottom: floor for this scale is ~5.0,
   achieved 5.162. gap train-val < 0.3 healthy.
   Loss is ln (starts 8.32 = ln 4096). perplexity = e^loss.
5. gutendex.com has no reachable IPv4 (hangs) — use www.gutenberg.org search HTML +
   `/ebooks/ID.txt.utf-8` fallback URL. PG phrase-search returns 0 for some queries;
   author pages work: `/ebooks/author/N` + `?start_index=N` for paging.
6. Inline `python -c "..."` with nested quotes breaks in bash — write a script file instead.
7. `pgrep -f 'train.py ...'` self-matches the shell command — use `ps aux | grep -E '[t]rain'`.
8. No tesseract/sudo: scanned-PDF OCR = pip `rapidocr-onnxruntime` + pdftoppm.
9. Killing training: `kill -INT <pid>` = same as Ctrl-C (safe, no handler needed).
10. Fixed `sample.py` race note: reading ckpt mid-save can error — just retry.

## Commands
```bash
cd ~/litmodel
.venv/bin/python -u train.py --preset 10m                 # THE run (defaults are final)
.venv/bin/python sample.py --talk "Are you happy?" --voice camus
.venv/bin/python sample.py --prompt "..." --tokens 300     # free-form
.venv/bin/python -c "import torch; print(torch.load('out/ckpt.pt',map_location='cpu',weights_only=False)['iter'])"
# checkpoint iter check (don't run while training is writing: retry on error)
```

## Session state (2026-10-07 ~19:15 IST)
- **Training RUNNING**: PID 78877, started ~18:49 (fresh restart on final 1900-iter config),
  estimated ~iter 21/1900, first real checkpoint ~19:49, epoch ETA ~35-42h total.
- Benchmark prompt: `OPHELIA: What is the difference between a true word and a fair one? / HAMLET:`
  — samples every ~5.3 min. User watches in terminal 1; optional 10-min loop in terminal 2.
- Early milestones observed: word-salad -> verse line-breaks (Shakespeare layout learning) ->
  "for i might" (first grammatical phrase, user hyped).
- Retired artifacts: 5m smoke ckpt (iter60, overwritten), 10m 1-iter test ckpt (current file).
- Possible next: epoch 3 (`--resume --iters 2850`) if val still falling at 1900;
  corpus-side RAG (BM25 over books) was offered but user chose assistant-memory instead.

## User preferences (recorded 2026-10-07)
- Wants baked-in defaults, minimal babysitting; hype/celebration for milestones, casual register.
- Priorities: live samples ~every 5 min, no underfitting (2+ epochs), in-character "talking"
  over line-completion, benchmark-style fixed prompts for time-lapse comparison.
- Has own Camus PDFs; expects them included (copyright = own files only, no fetching).

## 2026-10-08 — ROOT CAUSE: RNG-reset bug (the "16h divergence")
- **Bug**: `torch.manual_seed(42)` in train.py `sample_snippet` reset the GLOBAL
  rng after every eval (every 4 iters) -> training batch stream restarted
  identically each cycle -> model trained on the SAME ~32 windows (~33k tokens,
  0.4% of the corpus) for 950 iters. Eval windows were fixed too (why curves
  looked unusually smooth).
- **Measured effect** (calib protocol, control-matched to 0.007):
  iter 156: val 6.363 / train 6.140  ->  iter 950: val **7.842** / train **7.573**.
  ALL THREE "LR divergences" (3e-4, 1.5e-4, 7.5e-5) were this bug. LR was never
  the variable. `best_val 6.3559` froze at iter 156 -> that's the only "6.3" in
  the system (resume banner); live evals actually climbed to ~7.8.
- **Fix**: rng save/restore around `sample_snippet` (gen keeps seed-42
  determinism) AND `estimate_loss` (eval now fixed windows seed 777 = smooth
  curve, zero sampling noise, training stream untouched).
- **Truth-check protocol**: `~/litmodel/eval30.py` (30-batch val+train on any
  ckpt: `.venv/bin/python eval30.py ckpt.pt [seed]`). Trust it over any stored
  4-batch number. /tmp copies vanish; this one is durable.
- **Decision**: fresh restart (NO --resume), lr 2e-4, old ckpts archived in
  `out/old/` (best156 val 6.36 = insurance only). Gate: first hour both curves
  must FALL; by iter ~150-200 val should beat 6.36 on full-corpus training.

## Logging convention (2026-10-08): operator never pastes terminal again
- training runs -> `~/litmodel/logs/train.log` (tee -a, append-only, every run
  starts with `=== RUN <date> ===` header). Live watch: `tail -f ~/litmodel/logs/train.log`
- measurements  -> `~/litmodel/logs/eval.log` (eval30.py appends automatically)
- ad-hoc diagnostics -> tee'd into `~/litmodel/logs/` too (same session, same dir)
- durable launch command:
  cd ~/litmodel && echo "=== RUN $(date) ===" | tee -a ~/litmodel/logs/train.log \
    && .venv/bin/python -u train.py --preset 10m 2>&1 | tee -a ~/litmodel/logs/train.log
  (CURRENT: add `--resume` -- ckpt.pt exists; watchdog's LAUNCH uses the resume form)

## Lead-agent audit (2026-10-08 23:35): what it changed, verdict, recovery
User had the `lead` agent edit litmodel (22:08-23:10). Audit verdict: changes
GOOD, one incomplete step (watchdog file edited but its PROCESS never
restarted). Full inventory:

- **config.py** (22:23): refactored to `PRESET_OVERRIDES` + validated
  `build_preset()`; critical values verified unchanged (lr 2e-4, iters 1900,
  block 1024, L11 d256 H8, batch 1x8, warmup 30, eval_iters 4,
  ckpt_interval 50, min_lr 3e-5). TWO judgment calls:
  - `eval_interval 4 -> 12`: eval+sample overhead 11% -> 4%, but live samples
    now every ~14 min instead of ~5 (original operator requirement). One-line
    revert; either cadence is safe under the new watchdog budgets.
  - `checkpoint True -> False` (activation ckpt off): ~15% faster, RSS
    0.3-0.5GB -> ~1.86GB peak. Fits this box (measured 1.7GB avail + 3.5GB
    swap free). REVERT if swap grows >300MB under load.
- **train.py** (22:21): `atomic_save()` (tmp+fsync+os.replace = watchdog-kill
  proof), resume hardening + **best_val staleness fix** (takes min of
  ckpt.pt + ckpt.best.pt -- worked at relaunch: "5.682 was stale; using
  5.664"), new flags `--no-sample/--eval-interval/--eval-iters`. RNG fixes
  INTACT (seed-42 sample / seed-777 eval, both restore global rng state).
- **model.py** (22:22): MLP dropout now actually applied during training (was
  dead config); training-gated so eval/sample stay deterministic; NO
  state_dict change -> all checkpoints stay valid. Good direction: 10M params
  on 7.7M tokens wants regularization.
- **data.py** (22:53): fail-closed `prepare` guard -- hard SystemExit while
  any train.py is alive (/proc scan, checked twice, env escape hatch
  documented). Codifies rule #1. data/ bins + tokenizer NEVER touched
  (mtimes still Oct 7).
- **watchdog.py** (23:10): major hardening pass (FIX 1-9 / ISSUE 1-5 marked in
  comments): ALERT.log append-only, iter-stall gate (90 min), suspend/resume
  grace (25 min), O_EXCL halt flag, atomic .restarts, cadence-aware silence
  budgets for eval_interval=12 (ALERT at 30 min, kill at 60 min). Better than
  v2.

**The one broken state**: watchdog.py was edited but the running watchdog
process (started 15:37) still had OLD 12/20-min thresholds -- under
eval_interval=12 a battery-speed run (20-22 min eval gap) would have been
FALSE-KILLED, and AC runs would false-ALERT every cycle. Fixed at 23:36:
`bash launch_watchdog.sh` (new PID, lead's code) -> `rm .halt` -> watchdog
auto-restarted training 23:37:28, resumed iter 350 / best 5.664.

**Timeline**: 22:08 training died silently (SIGTERM, no traceback = deliberate
kill to edit safely); 22:08:47 watchdog auto-restarted it; 22:10 .halt touched
(empty) + the restart killed again; 22:21-23:10 file edits; 23:26 STATUS
HALTED; 23:36-37 audit + watchdog process restart + re-arm + auto-resume.
Total downtime ~1.5h; only ~6 iters of progress lost (ckpt.pt = iter 350).

## RUN RESULT (2026-10-09 14:52): converged at val 5.162, user stopped it
- Trajectory (fixed-window evals): iter 92: 6.338 -> 360: 5.664 ->
  ~950 (epoch 1): **5.34** -> 1656: **5.162** (train 4.759). Forecast
  "mid-5s by epoch 1" HIT; old 2.0-2.6 roadmap was pre-retracted fantasy.
- **Effectively converged**: last ~130 iters moved -0.009, cosine lr at
  3.7e-5 -> min 3e-5. Finishing to 1900 would add ~0.01-0.02. NOT worth
  resuming; the schedule is spent. No overfit (train-val gap stable ~0.4),
  zero watchdog kills, ran clean overnight at ~207 tok/s (lead's
  checkpoint-off + eval-12 speedups held; peak RSS ~1.4GB, RAM alert never
  fired).
- User killed training ~14:39; watchdog auto-restarted (by design, no .halt)
  37s later -- user fought a zombie until 14:52 when I armed .halt + killed.
  ckpt.pt = iter 1650, ckpt.best.pt = iter 1656 / 5.162. NOTHING LOST.
- Why 2.6 never came: 9.97M params on 7.7M tokens (2 epochs). val 5.16 =
  ppl ~174; val 2.6 = ppl ~13 (needs ~100x params+data). This scale's floor
  is ~5.0-5.1. Samples confirm: voice/meter learned, grammar NOT (word salad
  in iambic drag). More iters: 0. More corpus (Gutenberg/Camus): maybe
  4.7-4.9 for +30h. The real jump = GPU laptop run (50-100M params,
  10-50x data -> val 3s). This run = validated pipeline (tokenizer frozen,
  RNG fix, atomic ckpts, watchdog, eval30) for THAT run.
