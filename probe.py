"""Probe PG search for queries that surface the last two books."""
import re
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def search(q):
    u = "https://www.gutenberg.org/ebooks/search/?query=" + urllib.parse.quote(q)
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=20) as x:
            h = x.read().decode("utf-8", "replace")
    except Exception as e:
        return None, str(e)[:50]
    tot = (re.search(r'totalResults" content="(\d+)', h) or ["", "?"])[1]
    pairs = re.findall(
        r'href="/ebooks/(\d+)".*?<span class="title[^"]*">(.*?)</span>', h, re.S
    )
    return tot, [(i, re.sub(r"<[^>]+>", "", t).strip()) for i, t in pairs]


QUERIES = [
    "the double", "ivan ilich", "death ivan ilich", "tolstoy ilich",
]
KEYS = ["ivan", "ilyi", "double"]

for q in QUERIES:
    tot, res = search(q)
    hits = [r for r in (res or []) if r[1] and any(k in r[1].lower() for k in KEYS)]
    show = hits[:4] if hits else ([r for r in (res or []) if r[1]][:3] or res)
    print(f"{q!r:24} total={tot} -> {show}")
