"""Corpus pipeline: download public-domain texts, train BPE, encode to .bin.

  python data.py fetch     # download Gutenberg classics -> data/raw/
  python data.py prepare   # merge raw + data/raw/custom/*, train tokenizer, encode

Camus (still in copyright) is NOT fetched: drop your own .txt files into
data/raw/custom/ and `prepare` will include them automatically.
"""
import json
import os
import pathlib
import re
import sys
import urllib.parse
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CUSTOM = RAW / "custom"
OUT = ROOT / "data"

SEARCH = "https://www.gutenberg.org/ebooks/search/?query="
UA = {"User-Agent": "Mozilla/5.0 (litmodel corpus fetch)"}

# (slug, search query, [title keywords that must appear in the matched title])
WORKS = [
    # --- Shakespeare ---
    ("shakespeare_complete", "complete works of shakespeare", ["complete works", "works of shakespeare"]),
    ("hamlet", "hamlet shakespeare", ["hamlet"]),
    ("macbeth", "macbeth shakespeare", ["macbeth"]),
    ("romeo_and_juliet", "romeo and juliet shakespeare", ["romeo"]),
    ("othello", "othello shakespeare", ["othello"]),
    ("king_lear", "king lear shakespeare", ["lear"]),
    ("midsummer", "a midsummer night's dream shakespeare", ["midsummer"]),
    ("merchant", "the merchant of venice shakespeare", ["merchant"]),
    ("twelfth_night", "twelfth night shakespeare", ["twelfth"]),
    ("julius_caesar", "julius caesar shakespeare", ["caesar"]),
    # --- Dostoevsky ---
    ("crime_punishment", "crime and punishment dostoevsky", ["crime and punishment"]),
    ("karamazov", "the brothers karamazov dostoevsky", ["karamazov"]),
    ("underground", "notes from the underground dostoevsky", ["underground"]),
    ("idiot", "the idiot dostoevsky", ["idiot"]),
    ("demons", "the possessed dostoevsky demons", ["possessed", "demons"]),
    ("gambler", "the gambler dostoevsky", ["gambler"]),
    ("white_nights", "white nights dostoevsky", ["white nights"]),
    ("double", "the double dostoevsky", ["double"]),
    # --- Tolstoy ---
    ("war_peace", "war and peace tolstoy", ["war and peace"]),
    ("anna_karenina", "anna karenina tolstoy", ["anna karenina"]),
    ("ivan_ilyich", "the death of ivan ilyich tolstoy", ["ivan ily"]),
    # --- Others (the "etc") ---
    ("huck_finn", "adventures of huckleberry finn twain", ["huckleberry finn"]),
    ("tom_sawyer", "the adventures of tom sawyer", ["tom sawyer"]),
    ("moby_dick", "moby dick melville", ["moby"]),
    ("pride_prejudice", "pride and prejudice austen", ["pride and prejudice"]),
    ("sense_sensibility", "sense and sensibility austen", ["sense and sensibility"]),
    ("tale_two_cities", "a tale of two cities dickens", ["tale of two cities"]),
    ("great_expectations", "great expectations dickens", ["great expectations"]),
    ("time_machine", "the time machine wells", ["time machine"]),
    ("war_worlds", "the war of the worlds wells", ["war of the worlds"]),
    ("poe_tales", "the tell-tale heart and other tales poe", ["poe"]),
    ("chekhov_stories", "the steppe and other stories chekhov", ["chekhov"]),
]

PG_MARK = re.compile(
    r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
    re.DOTALL | re.IGNORECASE,
)
PG_END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG.*", re.DOTALL | re.IGNORECASE)


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def txt_urls(pg_id: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt",
        f"https://www.gutenberg.org/ebooks/{pg_id}.txt.utf-8",
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}-0.txt",
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}.txt",
    ]


def fetch_one(query: str, keywords: list[str]) -> tuple[str, str, str] | None:
    """Search gutenberg.org, try candidate ids, validate header against keywords.

    Returns (slug_text, title, url) of the first candidate that checks out.
    """
    try:
        html = http_get(SEARCH + urllib.parse.quote(query), timeout=20).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    search failed: {e}")
        return None
    ids = list(dict.fromkeys(re.findall(r'href="/ebooks/(\d+)"', html)))[:6]
    for pg_id in ids:
        for url in txt_urls(pg_id):
            try:
                text = http_get(url).decode("utf-8", errors="replace")
            except Exception:
                continue
            header = text[:4000].lower()
            if len(text) > 10_000 and any(k in header for k in keywords):
                return text, f"pg{pg_id}", url
            break  # txt fetched but didn't match -> try next id
    return None


def strip_boilerplate(text: str) -> str:
    m = PG_MARK.search(text)
    if m:
        text = text[m.end():]
    m = PG_END.search(text)
    if m:
        text = text[: m.start()]
    # normalize: collapse >2 blank lines, strip NULs
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def cmd_fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    CUSTOM.mkdir(parents=True, exist_ok=True)
    seen_ids: set[int] = set()
    ok, fail = [], []
    for slug, query, keywords in WORKS:
        dest = RAW / f"{slug}.txt"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[skip] {slug} (already downloaded)")
            ok.append(slug)
            continue
        print(f"[fetch] {slug}: {query!r}")
        got = fetch_one(query, keywords)
        if not got:
            print(f"    !! no match for {query!r}")
            fail.append(slug)
            continue
        text, title, furl = got
        text = strip_boilerplate(text)
        if len(text) < 10_000:
            print(f"    !! too short after cleanup ({len(text)} chars), skipping")
            fail.append(slug)
            continue
        dest.write_text(text, encoding="utf-8")
        print(f"    ok: {title} ({len(text):,} chars)")
        ok.append(slug)
    print(f"\nfetch done: {len(ok)} ok, {len(fail)} failed" + (f" -> {fail}" if fail else ""))


def load_corpus() -> str:
    parts = []
    files = sorted(RAW.glob("*.txt")) + sorted(CUSTOM.glob("*.txt"))
    if not files:
        raise SystemExit("no texts found - run `python data.py fetch` first")
    for f in files:
        t = f.read_text(encoding="utf-8", errors="replace")
        if f.parent == CUSTOM:
            print(f"[custom] {f.name}: {len(t):,} chars (yours, included as-is)")
        else:
            t = strip_boilerplate(t)
        if len(t) > 1000:
            parts.append(t)
    print(f"corpus: {len(parts)} files, {sum(len(p) for p in parts):,} chars total")
    return "\n\n".join(parts)


# Escape hatch for the fail-closed guard below. Set to exactly "1" ONLY when
# you personally know training is stopped AND the /proc scan cannot run here
# (non-Linux / unreadable /proc). OFF by default; never overrides a positive
# train.py detection -- it only lifts the "check unavailable" refusal.
SKIP_GUARD_ENV = "LITMODEL_PREPARE_UNSAFE_SKIP_TRAINER_CHECK"


class TrainerCheckUnavailable(RuntimeError):
    """Raised when the /proc trainer scan could not be completed.

    Deliberately NOT swallowed into a False return: the guard must FAIL CLOSED
    when it cannot verify that no trainer is running, instead of silently
    becoming a no-op on the exact machine where it matters.
    """


def training_is_alive() -> bool:
    """True if a live train.py process is running.

    Scans /proc/*/cmdline directly (stdlib only, no shell/subprocess) so this
    checker's own command line can never match itself -- the `pgrep -f`
    self-match trap. Own PID is skipped; entries that vanish mid-scan are
    tolerated. If /proc itself cannot be enumerated the answer would be a
    guess, so raises TrainerCheckUnavailable instead of returning False --
    callers treat that as "could not verify" and refuse to proceed
    (fail closed; see ensure_no_trainer).
    """
    me = os.getpid()
    try:
        pids = [p.name for p in pathlib.Path("/proc").iterdir() if p.name.isdigit()]
    except OSError as e:
        # cannot enumerate /proc -> incomplete scan; never guess "no trainer"
        raise TrainerCheckUnavailable(f"/proc not enumerable: {e}") from e
    for pid in pids:
        if int(pid) == me:
            continue
        try:
            raw = (pathlib.Path("/proc") / pid / "cmdline").read_bytes()
        except OSError:
            continue  # exited or not readable
        names = [pathlib.Path(a.decode("utf-8", "replace")).name for a in raw.split(b"\0") if a]
        if "train.py" not in names:
            continue
        # must actually be *launched* as python train.py (or ./train.py), so an
        # editor/shell merely mentioning train.py can't trip the guard.
        if names[0].startswith("python") or names[0] == "train.py":
            return True
    return False


def ensure_no_trainer(stage: str) -> None:
    """Refuse to proceed unless no live train.py has been positively verified.

    Single helper shared by BOTH cmd_prepare call sites ("initial" +
    "pre-write") so the two checks cannot drift. Fail-CLOSED semantics:

      * train.py detected       -> always SystemExit(1), no escape hatch;
      * check could not complete
        (/proc unenumerable, any
        unexpected scan error)  -> SystemExit(1), UNLESS the deliberate
        escape hatch env var `LITMODEL_PREPARE_UNSAFE_SKIP_TRAINER_CHECK`
        is set to exactly "1" (OFF by default) -- for a human who personally
        knows training is stopped on a box where the scan cannot run;
      * verified clear          -> return and let prepare continue.

    `stage` names which check failed ("initial" or "pre-write") in the message.

    WHY this refuses at all: `prepare`'s tofile() truncates data/train.bin in
    place; train.py's open np.memmap then dies with SIGBUS and the corpus can
    be left permanently truncated, so every later restart dies too.
    """
    try:
        alive = training_is_alive()
    except Exception as e:  # fail closed on unverifiable/unexpected scan errors
        if os.environ.get(SKIP_GUARD_ENV) == "1":
            print(f"[guard] {stage} check UNVERIFIABLE ({e}); {SKIP_GUARD_ENV}=1 honored")
            return
        print("=" * 78)
        print(f"FATAL [{stage} check]: guard could NOT verify that no train.py is running.")
        print(f"  scan failed: {e}")
        print("  Proceeding blind risks rewriting data/train.bin under a live")
        print("  np.memmap -> SIGBUS the trainer AND a truncated corpus.")
        print("  Stop training first and re-run; only if you are CERTAIN training")
        print(f"  is stopped and /proc cannot be scanned here, set {SKIP_GUARD_ENV}=1.")
        print("=" * 78)
        raise SystemExit(1)
    if alive:
        print("=" * 78)
        print(f"FATAL [{stage} check]: train.py is RUNNING right now -- refusing `prepare`.")
        print("  `prepare` rewrites data/train.bin under the open np.memmap held by")
        print("  training: it would SIGBUS the trainer AND truncate the corpus, so")
        print("  every later restart dies too. Stop training first, then re-run.")
        print("=" * 78)
        raise SystemExit(1)


def cmd_prepare():
    # GUARD (project rule #1): never rewrite train.bin under a live training
    # process -- tofile() truncates it in place, train.py's open np.memmap
    # then dies with SIGBUS and the corpus can be left permanently truncated.
    # Checked twice: here (fast, friendly failure before minutes of work) and
    # again right before tofile() (closes the check-then-write race window).
    ensure_no_trainer("initial")

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    OUT.mkdir(parents=True, exist_ok=True)
    text = load_corpus()

    tok_path = OUT / "tokenizer.json"
    if tok_path.exists():
        tok = Tokenizer.from_file(str(tok_path))
        print(f"[tok] reusing existing tokenizer ({tok.get_vocab_size()} vocab)")
    else:
        print("[tok] training byte-level BPE, vocab=4096 ...")
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=4096, min_frequency=2,
            special_tokens=["<pad>"], show_progress=True,
        )
        # train on a generator over the corpus so memory stays low
        tok.train_from_iterator((chunk for chunk in iter_chunks(text)), trainer)
        tok.save(str(tok_path))
        print(f"[tok] saved {tok_path} ({tok.get_vocab_size()} vocab)")

    print("[encode] tokenizing corpus ...")
    ids = np.array(
        [i for chunk in iter_chunks(text) for i in tok.encode(chunk).ids],
        dtype=np.uint16,
    )
    n = len(ids)
    split = int(n * 0.98)
    # Re-verify immediately before the irreversible write: minutes of encoding
    # have passed since the initial check, so a trainer may have started since
    # (TOCTOU). Fail CLOSED here too, same semantics as the initial check.
    ensure_no_trainer("pre-write")
    ids[:split].tofile(OUT / "train.bin")
    ids[split:].tofile(OUT / "val.bin")
    meta = {"vocab_size": tok.get_vocab_size(), "tokens": int(n), "train_tokens": int(split)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {n:,} tokens -> train {split:,} / val {n - split:,}")
    print(f"[done] vocab_size={meta['vocab_size']} (train.py reads this from meta.json)")


def iter_chunks(text: str, size: int = 1 << 20):
    for i in range(0, len(text), size):
        yield text[i:i + size]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        cmd_fetch()
    elif cmd == "prepare":
        cmd_prepare()
    else:
        raise SystemExit(__doc__)
