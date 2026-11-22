# NLP Project – Italian Cities (Group 5, Italian)

A corpus of Italian Wikipedia texts about 20 Italian cities, preprocessed with spaCy
(`it_core_news_md`). It is the base for a **search engine / question-answering system about Italian cities**.

**Presentation (Phase 1):** open [`index.html`](index.html) in a browser. It works offline and includes a live search demo.

## Phase 1 – status

| Requirement | Where |
|---|---|
| Corpus collection | `build_corpus.py` → `corpus/text01–20.txt`, metadata and licences in `sources.csv` |
| NLP preprocessing | `preprocess.py` → `corpus_clean/`, `corpus_processed/`, `output/` |
| Analysis (POS, dependencies, NER, quality) | `NLP_Phase1_Italian.ipynb` (and its `.html` export), `images/` |
| Application area | Search engine: TF-IDF over lemmas + cosine similarity (notebook §9, demo in `index.html`) |
| Presentation | `index.html` (built by `build_web.py` from `web/template.html`) |

## Corpus

* 20 texts, one per city (Roma … Reggio Calabria): the opening paragraphs of the it.wikipedia.org article.
* 220–300 words per text (5,589 words in total), 236 sentences, UTF-8, one paragraph per line.
* Collected with the MediaWiki API (plain-text extracts, no HTML scraping).
* Licence: CC BY-SA 4.0. Each text's source URL and access date are listed in `sources.csv`.

## Preprocessing pipeline

1. **Cleaning:** typographic apostrophes, non-breaking spaces, template leftovers (`Gamma -.`).
2. **Normalisation:** numbers written with spaces are joined (`2 743 357` → `2743357`), 25 cases in total.
3. **Segmentation:** sentence splitting and tokenization (Italian elisions such as `dell'` and `l'`).
4. **Annotation:** POS, lemma, dependencies, named entities.
5. **Filtering:** lowercase, alphabetic tokens only, stop words removed, lemmas kept.

| Step | Count |
|---|---|
| All tokens | 6,618 |
| Alphabetic words | 5,389 |
| Content words (no stop words) | 2,817 |
| Unique forms → unique lemmas | 1,646 → 1,290 |

## How to run

```bash
conda create -n nlp-tutorial1 -c conda-forge python=3.12
conda activate nlp-tutorial1
pip install -r requirements.txt

python build_corpus.py        # (optional) download the corpus again
python preprocess.py          # clean + preprocess -> corpus_clean/, corpus_processed/, output/
python make_notebook.py       # generate the notebook
jupyter nbconvert --to notebook --execute --inplace NLP_Phase1_Italian.ipynb
python build_web.py           # build index.html
```

## Next steps (Phase 2)

* Grow the corpus (~100 cities and more article sections) and use a custom stop list (spaCy's list contains *città*).
* Better retrieval: word vectors or sentence embeddings, entity-aware ranking.
* A chatbot interface on top of the search, plus an evaluation with test questions (precision@k).
