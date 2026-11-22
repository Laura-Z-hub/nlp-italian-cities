"""Build the Italian corpus: one text per Italian city, taken from it.wikipedia.org.

Text type : the opening paragraphs of Italian Wikipedia articles about Italian cities.
Length    : 220-300 words per text, made of whole paragraphs (the last one cut at a sentence end).
Licence   : CC BY-SA 4.0. Metadata (title, URL, date, licence, words) goes to sources.csv.
Files     : corpus/text01.txt, text02.txt, ... (UTF-8, one paragraph per line).
"""
import csv
import re
import time
from datetime import date
from pathlib import Path

import requests

API = "https://it.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "NLP-Tutorial1-StudentCorpus/1.0 (educational use)"}
TARGET = 20
MIN_W, MAX_W = 220, 300

# Italian cities, roughly by population; extra names are reserves if a text is too short
CITIES = [
    "Roma", "Milano", "Napoli", "Torino", "Palermo", "Genova", "Bologna",
    "Firenze", "Bari", "Catania", "Venezia", "Verona", "Messina", "Padova",
    "Trieste", "Brescia", "Parma", "Taranto", "Modena", "Reggio Calabria",
    "Perugia", "Cagliari", "Pisa", "Siena", "Lecce", "Trento", "Bergamo",
]


def fetch_article(title):
    """Return (title, url, plain text of the article) – one request, retry on HTTP 429."""
    params = {
        "action": "query", "format": "json", "formatversion": 2,
        "prop": "extracts|info", "explaintext": 1, "exsectionformat": "plain",
        "inprop": "url", "redirects": 1, "titles": title,
    }
    for attempt in range(6):
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        page = r.json()["query"]["pages"][0]
        if page.get("missing"):
            return None
        return page["title"], page["fullurl"], page.get("extract", "")
    raise RuntimeError("Wikipedia API is still rate-limiting, try again later")


def paragraphs(text):
    """Split into paragraphs and drop section headings (short lines without a full stop)."""
    out = []
    for p in text.split("\n"):
        p = re.sub(r"[ \t]+", " ", p).strip()
        if len(p.split()) < 12 or not p.endswith((".", "!", "?", ":", ";")):
            continue                        # heading, caption or empty line
        out.append(p)
    return out


def clean_first(p):
    """Remove the pronunciation / dialect bracket right after the city name.
    The bracket can contain nested brackets, e.g. Brescia (afi: /ˈbr(ɛ)ʃːa/ ...)."""
    start = p.find("(")
    if start == -1 or start > 60:
        return p
    depth = 0
    for i in range(start, len(p)):
        depth += {"(": 1, ")": -1}.get(p[i], 0)
        if depth == 0:
            return (p[:start].rstrip() + " " + p[i + 1:].lstrip()).replace(" ,", ",")
    return p


def take_words(paras, max_words):
    """Whole paragraphs until the limit; the last paragraph is cut at a sentence end."""
    out, n = [], 0
    for p in paras:
        w = len(p.split())
        if n + w <= max_words:
            out.append(p)
            n += w
            continue
        part = []
        for s in re.split(r"(?<=[.!?])\s+", p):
            if n + len(s.split()) > max_words:
                break
            part.append(s)
            n += len(s.split())
        if part:
            out.append(" ".join(part))
        break
    return out, n


def main():
    corpus = Path("corpus")
    corpus.mkdir(exist_ok=True)
    for old in corpus.glob("text*.txt"):
        old.unlink()
    rows, i = [], 0
    today = date.today().isoformat()
    for city in CITIES:
        if i >= TARGET:
            break
        res = fetch_article(city)
        time.sleep(2)
        if not res:
            print("missing:", city)
            continue
        title, url, extract = res
        paras = paragraphs(extract)
        if not paras:
            continue
        paras[0] = clean_first(paras[0])
        chosen, words = take_words(paras, MAX_W)
        if words < MIN_W:
            print(f"skip ({words} words): {title}")
            continue
        i += 1
        name = f"text{i:02d}.txt"
        (corpus / name).write_text("\n".join(chosen) + "\n", encoding="utf-8", newline="\n")
        rows.append([name, title, url, today, "CC BY-SA 4.0", words, len(chosen)])
        print(f"{name}: {title} ({words} words, {len(chosen)} paragraphs)")

    with open("sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "title", "url", "date_accessed", "licence", "words", "paragraphs"])
        w.writerows(rows)
    print("Saved", len(rows), "texts")


if __name__ == "__main__":
    main()
