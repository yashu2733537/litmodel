"""Size presets for the literary GPT. Actual param counts printed at startup.

Presets are declared as *data* in ``PRESET_OVERRIDES`` (preset name -> dict of
overridden fields) and materialised once by :func:`build_preset`, which merges
each override over the ``Config`` defaults and validates the result. Import
side effects are limited to building + validating the presets; this module is
stdlib-only by design (train.py / sample.py / eval30.py / watchdog.py import
it, and it must never pull in torch or numpy).

COMPATIBILITY CONTRACT: field names and types in ``Config`` are frozen.
Checkpoints store ``Config.__dict__`` and eval30.py rebuilds with an
*unfiltered* ``Config(**ck['config'])``. Changing default values and preset
values is safe; renaming, removing or adding fields is NOT.
"""
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any


@dataclass
class Config:
    vocab_size: int = 4096      # custom BPE trained on the corpus
    block_size: int = 256       # context length
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 240           # must be divisible by n_head
    dropout: float = 0.1
    checkpoint: bool = False      # trade ~15% speed for ~30% less RAM
    # training
    batch_size: int = 8
    grad_accum: int = 2         # effective batch = 16
    max_iters: int = 20000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    eval_interval: int = 250
    eval_iters: int = 50
    ckpt_interval: int = 500


# Fields every preset must leave strictly positive (catches 0/negative typos
# like batch_size=0 or eval_iters=0 that would silently stall training).
_POSITIVE_INT_FIELDS: tuple[str, ...] = (
    "vocab_size", "block_size", "n_layer", "n_head", "n_embd",
    "batch_size", "grad_accum", "max_iters", "warmup_iters",
    "eval_interval", "eval_iters", "ckpt_interval",
)
_POSITIVE_FLOAT_FIELDS: tuple[str, ...] = ("lr", "min_lr")
# Presets use 256 (5m) / 1024 (10m); anything below this trains nothing useful.
_MIN_BLOCK_SIZE: int = 64

_FIELD_NAMES: frozenset[str] = frozenset(f.name for f in fields(Config))
_DEFAULTS: dict[str, Any] = {}
for _f in fields(Config):
    if _f.default is MISSING:
        raise SystemExit(
            f"config.Config field {_f.name!r} has no default; every field must "
            "have one so old checkpoint dicts keep constructing"
        )
    _DEFAULTS[_f.name] = _f.default
del _f


def _fail(preset: str, field: str, value: Any, why: str) -> None:
    """Abort loudly, always naming the preset and the offending field."""
    raise SystemExit(f"invalid preset {preset!r}: {field}={value!r} -- {why}")


def _validate(preset: str, cfg: Config) -> None:
    """Cheap import-time sanity checks. Stdlib-only, no I/O, no heavy imports.

    Raises SystemExit with a message naming the preset and the offending field.
    """
    d = cfg.__dict__
    for name in _POSITIVE_INT_FIELDS:
        value = d[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            _fail(preset, name, value, "expected a positive int")
    for name in _POSITIVE_FLOAT_FIELDS:
        value = d[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            _fail(preset, name, value, "expected a positive float")
    if not isinstance(cfg.dropout, (int, float)) or isinstance(cfg.dropout, bool) \
            or not 0.0 <= float(cfg.dropout) <= 1.0:
        _fail(preset, "dropout", cfg.dropout, "expected 0.0 <= dropout <= 1.0")
    if cfg.block_size < _MIN_BLOCK_SIZE:
        _fail(preset, "block_size", cfg.block_size,
              f"below the sane minimum of {_MIN_BLOCK_SIZE}")
    # grad_accum < 1 and max_iters < 1 are rejected by the positivity loop;
    # vocab_size < 1 likewise (field listed first there).
    if cfg.n_embd % cfg.n_head != 0:
        _fail(preset, "n_embd", cfg.n_embd,
              f"not divisible by n_head={cfg.n_head} (heads must split the embedding)")


def build_preset(name: str, overrides: Mapping[str, Any]) -> Config:
    """Construct and validate one preset: ``Config(**{**defaults, **overrides})``.

    Unknown override keys are a hard error so a typo'd field name can never be
    silently swallowed again.
    """
    unknown = sorted(set(overrides) - _FIELD_NAMES)
    if unknown:
        raise SystemExit(
            f"invalid preset {name!r}: unknown Config field(s) {unknown} -- "
            "check PRESET_OVERRIDES for typos"
        )
    cfg = Config(**{**_DEFAULTS, **overrides})
    _validate(name, cfg)
    return cfg


# Preset definitions as plain data: name -> overridden fields (everything not
# listed falls back to the Config defaults above).
PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    # ~5.2M params  (wte 0.98M + 6 layers x 0.694M) -- pure Config defaults,
    # retired first run, kept for reference / relaunch comparison.
    "5m": {},
    # ~10.0M params, 1024-token context ("talk" model - sees a whole scene)
    "10m": {
        "block_size": 1024,            # a whole scene in one window
        "n_layer": 11, "n_head": 8, "n_embd": 256,   # 256 % 8 == 0 (validated)
        # 8192 tokens/iter; batch 1 keeps RAM sane. Do NOT raise the batch at
        # the same time as checkpoint-off -- they spend the same RAM budget.
        "batch_size": 1, "grad_accum": 8,
        # ~2 epochs (~35-42h). best.pt grabs the val-loss peak automatically -
        # stop anytime.
        "max_iters": 1900,
        # RESET 2026-10-08: every prior "divergence" was ONE bug, not LR --
        # seed-42 rng reset each eval = same 32 training windows
        # forever. Fresh restart, full corpus.
        # 2e-4 = under 3e-4's healthy early
        # descent, over the 1.5e-4 caution line.
        "lr": 2e-4,
        "warmup_iters": 30,
        # eval_iters 1 -> 4 (2026-10-07): single-batch readings had +-0.18 noise
        # (caused 3 false alarms). 4 batches = +-0.09. cost: ~+15s/cycle.
        # KEEP eval_iters at 4 -- lowering it only buys speed by buying noise.
        "eval_iters": 4,
        # THROUGHPUT 2026-10-08: eval_interval 4 -> 12. Measured at eval=4,
        # eval+sample overhead was ~35s per 4-iter cycle = ~11% of wall clock;
        # at eval=12 it drops to ~4% at the same +-0.09 noise level. A benchmark
        # sample now lands ~every 13-14 min instead of ~every 5 min.
        "eval_interval": 12,
        # 50 iters ~ 55 min between saves; bounds crash data loss -- do NOT raise.
        "ckpt_interval": 50,
        # THROUGHPUT 2026-10-08: checkpoint True -> False (old note: "1024-
        # context activations are the RAM hog"). Activation checkpointing
        # traded ~15% speed for ~30% RAM. Measured RSS: ON = 286-452MB,
        # OFF = peak ~1.86GB on a 3.2GB box with ~1.3-1.5GB free + swap, so
        # the headroom exists. state_dict is unaffected (checkpointing is a
        # runtime memory strategy), so existing ckpts stay valid.
        # Expected combined with eval_interval=12: ~116 -> ~145-155 tok/s.
        # REVERT: if swap grows past ~300MB under load (or free RAM crowds
        # below ~300MB), set checkpoint=True again and eat the ~15% slowdown.
        "checkpoint": False,
    },
    # ~18.9M params (wte 1.84M + 7 layers x 2.41M)
    "20m": {"n_layer": 7, "n_head": 7, "n_embd": 448, "batch_size": 4, "grad_accum": 4},
}


PRESETS: dict[str, Config] = {
    name: build_preset(name, overrides)
    for name, overrides in PRESET_OVERRIDES.items()
}


def load_config(preset: str) -> Config:
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}, choose from {list(PRESETS)}")
    return PRESETS[preset]
