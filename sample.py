"""Generate text from a trained checkpoint.

  python sample.py                                    # default prompt, 400 tokens
  python sample.py --prompt "To be or not to be" --tokens 600 --temp 0.7
  python sample.py --ckpt out/ckpt.best.pt            # best-val checkpoint

  # "talk" mode: ask it a question, it answers in character
  python sample.py --talk "Are you happy?"
  python sample.py --talk "What is courage?" --voice shakespeare
  python sample.py --talk "Why do people suffer?" --voice camus --temp 0.7
"""
import argparse
import pathlib

import torch

from model import GPT

ROOT = pathlib.Path(__file__).resolve().parent

DEFAULT_PROMPTS = [
    "To be, or not to be,",
    "The punishment that awaited him",
    "In the beginning of the summer,",
    "It is a truth universally acknowledged",
]

# Each template frames the question as something already happening in a scene,
# ending on a cue ("he demanded.", "I answered:", HAMLET:) so the completion
# that follows IS the answer, in that voice.
VOICES = {
    "dostoevsky": "— {q} he demanded.\n\nI answered at once: ",
    "camus": "— {q} she asked.\n\nI considered it. ",
    "shakespeare": "OPHELIA:\n{q}\n\nHAMLET:\n",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "out" / "ckpt.pt"))
    ap.add_argument("--prompt", default=None, help="omit for a random default prompt")
    ap.add_argument("--talk", default=None, metavar="QUESTION",
                    help="ask a question, get an in-character answer")
    ap.add_argument("--voice", default="dostoevsky", choices=sorted(VOICES),
                    help="who answers in --talk mode")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import json
    from tokenizers import Tokenizer

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_d = ck["config"]
    cfg_d.setdefault("vocab_size", json.loads((ROOT / "data" / "meta.json").read_text())["vocab_size"])

    from config import Config
    cfg = Config(**{k: v for k, v in cfg_d.items() if k in Config.__dataclass_fields__})

    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer.json"))
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.talk is not None:
        prompt = VOICES[args.voice].format(q=args.talk)
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPTS[torch.randint(0, len(DEFAULT_PROMPTS), (1,)).item()]

    prefix_len = len(tok.encode(prompt).ids)
    ids = tok.encode(prompt).ids[-cfg.block_size:]
    prefix_len = min(prefix_len, cfg.block_size)
    x = torch.tensor([ids], dtype=torch.long)

    if args.talk is not None:
        print(f"voice: {args.voice}  |  ckpt iter {ck.get('iter', '?')}  |  temp {args.temp}")
        print(f"you  : {args.talk}")
        print("reply:\n")
    else:
        print(f"--- prompt: {prompt!r} (ckpt iter {ck.get('iter', '?')}) ---\n")

    out = model.generate(x, args.tokens, temperature=args.temp, top_k=args.topk)
    if args.talk is not None:
        reply = tok.decode(out[0].tolist()[prefix_len:])
        print(reply.strip())
        print("\n--- end ---")
    else:
        print(tok.decode(out[0].tolist()))
        print("\n--- end ---")


if __name__ == "__main__":
    main()
