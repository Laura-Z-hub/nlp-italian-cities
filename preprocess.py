"""Phase 1 preprocessing of the Italian city corpus.

Input : corpus/textNN.txt (raw Wikipedia paragraphs, from build_corpus.py)
Output:
  corpus_clean/textNN.txt      cleaned text, one sentence per line
  corpus_processed/textNN.txt  content lemmas (lowercase, no stop words/punctuation), one sentence per line
  output/preprocessing_stats.json   numbers used in the presentation
  output/search_index.json          sentence index used by the search demo (application area)

Run:  python preprocess.py   (environment with spaCy + it_core_news_md)
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import spacy

RAW, CLEAN, PROC, OUT = Path("corpus"), Path("corpus_clean"), Path("corpus_processed"), Path("output")


def clean_text(text):
    """Normalisation before spaCy. Returns (clean text, Counter of fixes)."""
    fixes = Counter()
    text = text.replace("’", "'").replace(" ", " ")          # typographic apostrophe, nbsp

    # thousands written with spaces: 2 743 357 -> 2743357 (one NUM token instead of three)
    def join_number(m):
        fixes["numbers joined"] += 1
        return m.group(0).replace(" ", "")
    text = re.sub(r"\b\d{1,3}(?: \d{3})+\b", join_number, text)

    # template leftovers lost in the plain-text export, e.g. "categoria Gamma -." or "( )"
    text, n = re.subn(r"\s+-\s*\.", ".", text)
    fixes["template leftovers removed"] += n
    text, n = re.subn(r"\(\s*\)", "", text)
    fixes["empty brackets removed"] += n

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([,.;:])", r"\1", text)
    return text.strip(), fixes


def main():
    nlp = spacy.load("it_core_news_md")
    stop = nlp.Defaults.stop_words
    for d in (CLEAN, PROC, OUT):
        d.mkdir(exist_ok=True)

    with open("sources.csv", encoding="utf-8") as f:
        meta = {r["file"]: r for r in csv.DictReader(f)}

    files = sorted(RAW.glob("text*.txt"))
    fixes = Counter()
    raw_tok = alpha_tok = content_tok = 0
    forms, lemmas_all, words_all = Counter(), Counter(), Counter()
    pos, ents = Counter(), Counter()
    ent_examples = {}
    form2lemma = Counter()
    sentences = []           # search index
    per_text = []
    n_sent = 0

    for f in files:
        raw = f.read_text(encoding="utf-8")
        clean, fx = clean_text(raw)
        fixes += fx
        doc = nlp(clean)

        clean_lines, proc_lines = [], []
        for s in doc.sents:
            st = s.text.strip()
            if not st:
                continue
            content = []
            for t in s:
                raw_tok += 1
                if t.is_space:
                    continue
                pos[t.pos_] += 1
                if not t.is_alpha:
                    continue
                alpha_tok += 1
                low, lem = t.text.lower(), t.lemma_.lower()
                words_all[low] += 1
                forms[low] += 1
                if t.is_stop or low in stop or lem in stop:
                    continue
                content_tok += 1
                content.append(lem)
                lemmas_all[lem] += 1
                form2lemma[(low, lem)] += 1
            clean_lines.append(st)
            proc_lines.append(" ".join(content))
            sentences.append({"doc": f.name, "text": st, "lemmas": content})
            n_sent += 1

        for e in doc.ents:
            ents[e.label_] += 1
            ent_examples.setdefault(e.label_, Counter())[e.text] += 1

        (CLEAN / f.name).write_text("\n".join(clean_lines) + "\n", encoding="utf-8", newline="\n")
        (PROC / f.name).write_text("\n".join(proc_lines) + "\n", encoding="utf-8", newline="\n")
        m = meta[f.name]
        per_text.append({"file": f.name, "title": m["title"], "url": m["url"],
                         "words": int(m["words"]), "paragraphs": int(m["paragraphs"]),
                         "sentences": len(clean_lines)})
        print(f"{f.name}: {m['title']:<16} {len(clean_lines)} sentences")

    # most frequent lemma for every surface form (used to lemmatise queries in the browser)
    best = {}
    for (form, lem), c in form2lemma.most_common():
        best.setdefault(form, lem)

    # before / after example for the presentation
    ex_raw = (RAW / "text01.txt").read_text(encoding="utf-8").split(". ")[0] + "."
    ex_clean, _ = clean_text(ex_raw)
    ex_doc = nlp(ex_clean)
    example = {
        "raw": ex_raw,
        "clean": ex_clean,
        "tokens": [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_,
                    "stop": bool(t.is_stop or t.lemma_.lower() in stop), "alpha": t.is_alpha}
                   for t in ex_doc],
        "raw_tokens": [t.text for t in nlp.tokenizer(ex_raw)][:12],
    }

    stats = {
        "texts": len(files),
        "sentences": n_sent,
        "raw_words": sum(p["words"] for p in per_text),
        "tokens_all": raw_tok,
        "tokens_alpha": alpha_tok,
        "tokens_content": content_tok,
        "unique_forms": len(forms),
        "unique_lemmas": len(lemmas_all),
        "fixes": dict(fixes),
        "top_words": words_all.most_common(10),
        "top_lemmas": lemmas_all.most_common(15),
        "pos": pos.most_common(),
        "entities": ents.most_common(),
        "entity_examples": {k: [w for w, _ in v.most_common(5)] for k, v in ent_examples.items()},
        "per_text": per_text,
        "example": example,
    }
    (OUT / "preprocessing_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    index = {"stop": sorted(stop), "form2lemma": best, "sentences": sentences,
             "docs": {p["file"]: {"title": p["title"], "url": p["url"]} for p in per_text}}
    (OUT / "search_index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    print(f"\n{n_sent} sentences | tokens {raw_tok} -> alpha {alpha_tok} -> content {content_tok}")
    print(f"unique forms {len(forms)} -> unique lemmas {len(lemmas_all)} | fixes {dict(fixes)}")


if __name__ == "__main__":
    main()
