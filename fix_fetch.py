"""One-off: grab books PG's phrase search failed to find, via author-wide search."""
import pathlib
import re
import urllib.parse

from data import http_get, strip_boilerplate, txt_urls


def author_books(q, pages=6):
    """id -> title for every book in PG's search results for query q."""
    seen = {}
    for p in range(1, pages + 1):
        url = ("https://www.gutenberg.org/ebooks/search/?query="
               + urllib.parse.quote(q) + (f"&page={p}" if p > 1 else ""))
        try:
            html = http_get(url, 20).decode("utf-8", "replace")
        except Exception as e:
            print("  page fail", p, e)
            break
        pairs = re.findall(
            r'href="/ebooks/(\d+)".*?<span class="title[^"]*">(.*?)</span>', html, re.S
        )
        if not pairs:
            break
        for i, t in pairs:
            seen.setdefault(i, re.sub(r"<[^>]+>", "", t).strip())
    return seen


TARGETS = [
    ("demons", "dostoevsky", ["possessed", "devil"]),
    ("double", "the double fyodor", ["double"]),
    ("ivan_ilyich", "ivan ilyich", ["ivan"]),
]

cache = {}
for slug, q, kw in TARGETS:
    if (pathlib.Path(f"data/raw/{slug}.txt").exists()):
        print(f"[skip] {slug} exists")
        continue
    if q not in cache:
        cache[q] = author_books(q)
        print(f"[{q}] {len(cache[q])} books found")
    hits = [(i, t) for i, t in cache[q].items() if any(k in t.lower() for k in kw)]
    print(f"[{slug}] candidates: {hits[:5]}")
    done = False
    for i, t in hits:
        for u in txt_urls(int(i)):
            try:
                txt = http_get(u).decode("utf-8", "replace")
            except Exception:
                continue
            if len(txt) > 10000:
                text = strip_boilerplate(txt)
                pathlib.Path(f"data/raw/{slug}.txt").write_text(text, encoding="utf-8")
                print(f"  ok: {slug} <- {t} (pg{i}, {len(text):,} chars)")
                done = True
                break
        if done:
            break
    if not done:
        print(f"  FAIL: {slug}")
