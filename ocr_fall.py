"""OCR the scanned Fall PDF with rapidocr-onnxruntime (no system deps needed)."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PDF = pathlib.Path.home() / "Documents" / "WORK OF ALBER CAMUS" / "2015.125150.The-Fall.pdf"
DEST = ROOT / "data" / "raw" / "custom" / "camus_the_fall.txt"
PAGES = pathlib.Path("/tmp/fallpages")

if DEST.exists() and DEST.stat().st_size > 100_000:
    print("fall txt already exists, nothing to do")
    sys.exit(0)

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    sys.exit("rapidocr-onnxruntime not installed")

PAGES.mkdir(parents=True, exist_ok=True)
if not list(PAGES.glob("page*.png")):
    print("rendering pdf pages ...")
    subprocess.run(
        ["pdftoppm", "-r", "150", "-gray", "-png", str(PDF), str(PAGES / "page")],
        check=True,
    )

engine = RapidOCR()
out = []
files = sorted(PAGES.glob("page*.png"))
for i, p in enumerate(files, 1):
    try:
        res, _ = engine(str(p))
        text = "\n".join(r[1] for r in res) if res else ""
    except Exception as e:
        text = f"[page error: {e}]"
    out.append(text)
    if i % 10 == 0:
        print(f"{i}/{len(files)} pages", flush=True)

DEST.write_text("\n\n".join(out), encoding="utf-8")
print(f"done: {DEST} ({DEST.stat().st_size:,} bytes)")
