"""Generate NLP_Phase1_Italian.ipynb (run it afterwards with nbconvert --execute)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# NLP Project – Phase 1: corpus, preprocessing and application area

**Group 5 – Italian** · model: `it_core_news_md`

Corpus: 20 Italian cities – the opening paragraphs of their Italian Wikipedia articles
(248–300 words and 4–8 paragraphs per text), licence CC BY-SA 4.0.
The texts were collected with `build_corpus.py` (Wikipedia API, no scraping);
the metadata is in `sources.csv`.
""")

md("""
## 1. Setup – Anaconda environment, TensorFlow, Keras, spaCy

Environment created with Anaconda (Section 1.3 of the tutorial):

```
conda create -n nlp-tutorial1 -c conda-forge python=3.12
conda activate nlp-tutorial1
conda install -c conda-forge spacy jupyter
pip install tensorflow keras          # conda-forge has no recent TensorFlow for Windows
python -m spacy download it_core_news_md
```
""")
code("""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # hide TensorFlow info messages
import tensorflow as tf
import keras
print('TensorFlow:', tf.__version__)
print('Keras     :', keras.__version__)
""")
code("""
import sys
import spacy
from spacy import displacy

print('Python:', sys.version.split()[0])
print('spaCy :', spacy.__version__)
""")
code("!\"{sys.executable}\" -m spacy info")

md("""
## 2. Loading the Italian model and processing one sentence

Installed with `python -m spacy download it_core_news_md`.
""")
code("""
nlp = spacy.load('it_core_news_md')
print('Pipeline:', nlp.pipe_names)

doc = nlp("Leonardo da Vinci nacque a Vinci nel 1452 e lavorò a Firenze e a Milano.")

print(f"{'TOKEN':<10}{'LEMMA':<10}{'POS':<7}{'DEP':<12}{'HEAD':<10}ENT")
for t in doc:
    print(f"{t.text:<10}{t.lemma_:<10}{t.pos_:<7}{t.dep_:<12}{t.head.text:<10}{t.ent_type_}")

print('\\nEntities:', [(e.text, e.label_) for e in doc.ents])
""")

md("""
## 3. Visualization with displaCy

`displacy.serve(doc, style='dep')` starts a web server on port 5000 (use `port=5001`
if the port is busy). Inside a notebook we use `displacy.render()` instead, which shows
the result directly in the cell. We also save every picture to the `images/` folder
(SVG for dependencies, HTML for entities).
""")
code("""
from pathlib import Path

IMG = Path('images')
IMG.mkdir(exist_ok=True)
DEP_OPTS = {'compact': True, 'distance': 110}

def show_and_save(doc, name):
    displacy.render(doc, style='dep', jupyter=True, options=DEP_OPTS)
    displacy.render(doc, style='ent', jupyter=True)
    (IMG / f'{name}_dep.svg').write_text(
        displacy.render(doc, style='dep', options=DEP_OPTS, jupyter=False), encoding='utf-8')
    (IMG / f'{name}_ent.html').write_text(
        displacy.render(doc, style='ent', page=True, jupyter=False), encoding='utf-8')

show_and_save(doc, 'example')
""")

md("Practice with a few more sentences:")
code("""
practice = [
    "Bill Gates è il fondatore di Microsoft.",
    "Il Papa vive nella Città del Vaticano, a Roma.",
    "Ieri Maria ha comprato un gelato all'Università di Bologna.",
]
for i, s in enumerate(practice, 1):
    d = nlp(s)
    print(s, '->', [(e.text, e.label_) for e in d.ents])
    displacy.render(d, style='ent', jupyter=True)
""")

md("""
## 4. Building the corpus

* **Topic:** Italian cities – Roma, Milano, Napoli, Torino, Palermo, Genova, Bologna, Firenze, Bari,
  Catania, Venezia, Verona, Messina, Padova, Trieste, Brescia, Parma, Taranto, Modena, Reggio Calabria.
* **Text type:** the opening paragraphs of each city's Italian Wikipedia article: whole paragraphs,
  248–300 words per text (the last paragraph is cut at the end of a sentence).
* **Source / licence:** it.wikipedia.org, CC BY-SA 4.0 (reuse allowed with attribution).
* **Files:** `corpus/text01.txt` … `corpus/text20.txt`, UTF-8, one paragraph per line.
* **Metadata:** `sources.csv` (file, title, URL, date accessed, licence, words, paragraphs).

### 4.1 Encoding check (UTF-8, no mojibake)
""")
code("""
import re, csv

files = sorted(Path('corpus').glob('*.txt'))
accents = 'àèéìòùÀÈÉÌÒÙ'
for f in files:
    raw = f.read_bytes()
    raw.decode('utf-8')                       # raises an error if a file is not UTF-8
    text = raw.decode('utf-8')
    mojibake = re.findall(r'Ã.|Â.|â€', text)  # typical signs of a wrong encoding
    found = sorted({c for c in text if c in accents})
    print(f"{f.name}: {len(text.split()):>3} words, accents {''.join(found):<8} mojibake: {len(mojibake)}")
""")

md("### 4.2 Loading and measuring the corpus (Section 6.4)")
code("""
from collections import Counter

texts = [f.read_text(encoding='utf-8') for f in files]

tokens = []
docs = []
for text in texts:
    d = nlp(text)
    docs.append(d)
    tokens += [t.text.lower() for t in d if t.is_alpha]

print('Texts:', len(texts))
print('Tokens:', len(tokens))
print('Unique words:', len(set(tokens)))
print(Counter(tokens).most_common(10))
""")
code("""
# The same ranking without stop words and with lemmas, to see the topic of the corpus
content = [t.lemma_.lower() for d in docs for t in d if t.is_alpha and not t.is_stop]
print('Top 15 content lemmas:', Counter(content).most_common(15))
print('Sentences:', sum(len(list(d.sents)) for d in docs))
print('Paragraphs:', sum(len([p for p in t.splitlines() if p.strip()]) for t in texts))
print('Average tokens per text:', round(len(tokens) / len(texts), 1))
""")

md("""
**What are the ten most frequent words?** *e, di, la, il, del, è, della, città, in, nel*.
Nine of them are stop words (conjunctions, prepositions, articles and the verb *essere*); they
only show that the text is Italian. Note how many are articulated prepositions
(*del*, *della*, *nel* = preposition + article). The one exception is ***città*** (city, 94 times),
which already gives away the topic. After removing stop words the topic is clear:
*comune, centro, italiano, secolo, capitale, capoluogo, abitante, metropolitano, regione* –
the vocabulary of describing a city. (The lemma *il* still appears because the elided article
*l'* is not marked as a stop word.)
""")

md("## 5. Looking for quality problems (Section 6.5)")
code("""
problems = {}
def note(name, msg):
    problems.setdefault(name, []).append(msg)

for f, text in zip(files, texts):
    if re.search(r'<[^>]+>|&[a-z]+;|\\[\\d+\\]', text):
        note(f.name, 'HTML tag / entity / reference mark')
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) != len(set(lines)):
        note(f.name, 'repeated line')
    if re.search(r'\\b(Modifica|Indice|Vai a|Menu|Leggi)\\b', text):
        note(f.name, 'possible menu/boilerplate words')
    if re.search(r'\\d \\d{3}', text):
        note(f.name, 'number written with spaces (e.g. 2 743 357)')
    if re.search(r'\\s-\\.|\\(\\s*\\)|\\s,', text):
        note(f.name, 'leftover of a removed template (e.g. "Gamma -." or "( )")')

# sentences in another language: share of Italian stop words per sentence
for f, d in zip(files, docs):
    for s in d.sents:
        words = [t for t in s if t.is_alpha]
        if len(words) >= 5 and sum(t.is_stop for t in words) / len(words) < 0.15:
            note(f.name, f'sentence with few Italian stop words: "{s.text[:60]}..."')

# foreign words recognised by the model as MISC entities (Latin, English ...)
for f, d in zip(files, docs):
    misc = [e.text for e in d.ents if e.label_ == 'MISC']
    if misc:
        note(f.name, f'MISC entities: {misc[:4]}')

for name in sorted(problems):
    print(name)
    for m in problems[name]:
        print('   -', m)
""")
code("""
# Italian Wikipedia writes thousands with a space, so one number becomes several tokens
d = nlp(texts[0].split('.')[0])
print([t.text for t in d][:12])
""")

md("""
**Findings**

* No HTML tags, menus or repeated lines – the Wikipedia API returns plain text, so the corpus is clean.
* Numbers are written with spaces as thousand separators (*2 743 357 abitanti*) and are split into
  three `NUM` tokens. This changes the dependency tree (see sentence 1 below).
* One small template leftover in `text04.txt` (Torino: *nella categoria Gamma -.*), where a
  ranking symbol was lost when converting to plain text.
* Pronunciation brackets in IPA and dialect names after the city name, e.g.
  *Brescia (afi: /ˈbr(ɛ)ʃːa/, … Brèsa o Bressa in dialetto bresciano …)*. Our collection script
  removes this first bracket (with nested brackets), because it is not running Italian text.
* The sentence flagged as "few Italian stop words" (text07, Bologna) is Italian – a false
  positive of our simple heuristic (it is full of numbers and names).
* Some entities and words are not Italian or are historical names (*Partenope*, *Cumani*,
  *Mediolanum*), and the model labels many of them **MISC**.
* All 20 texts begin with the same pattern (*X è un comune italiano di N abitanti, capoluogo …*),
  which inflates words such as *comune*, *capoluogo*, *abitanti*, *omonimo*.
""")

md("""
### 5.1 Three sentences from the corpus – dependencies and entities
""")
code("""
def find_sentence(doc, start):
    return next(s.text.strip() for s in doc.sents if s.text.strip().startswith(start))

three = [
    find_sentence(docs[0], 'Roma è un comune'),   # Roma (text01)
    find_sentence(docs[2], 'Napoli nasce'),       # Napoli (text03)
    find_sentence(docs[11], 'Di origine'),        # Verona (text12)
]
for i, s in enumerate(three, 1):
    d = nlp(s)
    print(f'--- Sentence {i}:', s)
    print('Entities:', [(e.text, e.label_) for e in d.ents])
    print(' '.join(f'{t.text}/{t.pos_}' for t in d))
    show_and_save(d, f'sentence{i}')
""")
code("""
# The details we discuss below
d1, d2, d3 = (nlp(s) for s in three)
print('S1:', [(t.text, t.pos_, t.dep_, t.head.text) for t in d1 if t.text in ('comune', '2', '743', '357', 'abitanti')])
print('S2:', [(t.text, t.lemma_, t.dep_) for t in d2 if t.text in ('fondata', 'secolo', 'suggeriscano')], [(e.text, e.label_) for e in d2.ents])
print('S3:', [(t.text, t.pos_, t.dep_) for t in d3 if t.text in ('abitato', 'venne', 'rifondato', 'Grande')], [(e.text, e.label_) for e in d3.ents])
""")

md("""
| # | Sentence | Correct ✔ | Wrong ✘ |
|---|---|---|---|
| 1 | *Roma è un comune italiano di 2 743 357 abitanti, capoluogo dell'omonima città metropolitana e della regione Lazio, nonché capitale della Repubblica Italiana.* | *Roma*, *Lazio* and *Repubblica Italiana* are found as entities; the contraction *dell'* is split off and lemmatised as *di il*. | *comune* is tagged **ADJ** instead of NOUN, and because the number is split into three tokens, *abitanti* becomes a second **nsubj** instead of a modifier of *comune*. |
| 2 | *Napoli nasce come Partenope, fondata dai Cumani su Pizzofalcone a fine VIII secolo a.C. su base archeologica, sebbene sporadici reperti suggeriscano una datazione più antica.* | *Napoli*, *Partenope* (the old Greek name) and the hill *Pizzofalcone* are all found as **LOC**; ROOT is *nasce*, and *fondata → fondare* is correctly attached to *Partenope*. | The subjunctive *suggeriscano* gets the non-existing lemma ***suggeriscare*** (should be *suggerire*); *secolo* is analysed as **obl:agent** (the agent of a passive), but it is a time expression. |
| 3 | *Di origine preistorica, l'abitato venne rifondato dai Romani all'interno dell'ansa del fiume …, quando venne occupato dal re germanico Teodorico il Grande.* | The passive with *venire* is analysed correctly: *abitato* = **nsubj:pass**, *venne* = **aux:pass**, ROOT = *rifondato*; *Teodorico il Grande* = **PER**. | *Romani* (a people) and *Impero* are labelled **LOC**; *Grande*, part of the name, becomes a separate subject (**nsubj:pass**) of *occupato*. |

Additional observation: the Italian model only has the labels **PER, LOC, ORG, MISC** – there is
no DATE label, so *I secolo a.C.* is labelled MISC and many dates are not found at all.
""")

md("""
### 5.2 Long words, accents and contractions
""")
code("""
for s in ["l'uomo dell'Italia, del nel all'inizio", "Kraftfahrzeug-Haftpflichtversicherung",
          "L'Università è più antica dell'Europa", "c'è un po' di caffè"]:
    d = nlp(s)
    print(s)
    print('   tokens:', [t.text for t in d])
    print('   lemmas:', [t.lemma_ for t in d])
""")
md("""
* Elided articles and prepositions (*l'*, *dell'*, *all'*, *c'*) are separate tokens and are
  lemmatised to *il*, *di il*, *a il* – good for counting words.
* Articulated prepositions without an apostrophe (*del*, *nel*) stay **one token** but receive the
  two-word lemma *di il* / *in il*.
* Accented letters (*è, più, Università, caffè*) are handled correctly because the corpus is UTF-8.
* A long German compound is not split – the Italian tokenizer only splits on spaces and punctuation.
""")

md("## 6. Our row of the comparison table (Section 6.6)")
code("""
with open('sources.csv', encoding='utf-8') as f:
    sources = list(csv.DictReader(f))

row = {
    'Language': 'Italian',
    'Source': 'it.wikipedia.org (Italian cities, opening paragraphs)',
    'Licence': 'CC BY-SA 4.0',
    'Texts': len(texts),
    'Tokens': len(tokens),
    'Unique words': len(set(tokens)),
    'Problems found': 'numbers split by spaces; 1 template leftover; IPA brackets; same opening pattern in all texts',
}
for k, v in row.items():
    print(f'{k:<15}{v}')
""")
md("""
| Language | Source | Licence | Texts | Tokens | Unique words | Problems found |
|---|---|---|---|---|---|---|
| Italian | it.wikipedia.org (20 Italian cities) | CC BY-SA 4.0 | 20 | 5389 | 1646 | numbers split by spaces, 1 template leftover (*Gamma -.*), IPA pronunciation brackets, same opening pattern, ADJ/NOUN and lemma errors, no DATE entities |

**Discussion**

1. *Most problems:* for Italian they come from formatting (numbers with spaces, lost templates,
   pronunciation brackets) and from ambiguous or rare words (*comune* noun vs. adjective,
   unusual verb forms such as *suggeriscano* that get a wrong lemma).
2. *Long words / accents / contractions:* see Section 5.2 – elisions with apostrophe are split,
   *del/nel* are kept as one token with a two-word lemma, accents are fine in UTF-8.
3. *What we would change:* normalise numbers (remove the space separators), remove template
   leftovers, use more varied parts of the articles (history, economy, culture) so that not every
   text starts with the same sentence, grow the corpus to ~100 cities/texts, and keep
   one sentence per line to make later tokenization and n-gram work easier.
""")

md("## 7. Sources and licences")
code("""
for s in sources:
    print(f"{s['file']}  {s['title']:<25} {s['url']}  ({s['date_accessed']}, {s['licence']})")
""")

md("""
## 8. Preprocessing pipeline (`preprocess.py`)

Steps: **cleaning** (apostrophes, template leftovers) → **normalisation** (numbers written with spaces
are joined, `2 743 357` → `2743357`) → **sentence splitting + tokenization** → **POS / lemma / NER** →
**filtering** (lowercase, only alphabetic tokens, no stop words) → content lemmas.

Output: `corpus_clean/` (one clean sentence per line), `corpus_processed/` (lemmas per sentence),
`output/preprocessing_stats.json`, `output/search_index.json`.
""")
code("""
import json, preprocess
preprocess.main()
""")
code("""
from preprocess import clean_text
before = texts[0].split('.')[0] + '.'
after, fixes = clean_text(before)
print('BEFORE:', [t.text for t in nlp(before)][:10])
print('AFTER :', [t.text for t in nlp(after)][:8])
print()
print('clean    :', Path('corpus_clean/text01.txt').read_text(encoding='utf-8').splitlines()[0])
print('processed:', Path('corpus_processed/text01.txt').read_text(encoding='utf-8').splitlines()[0])
""")
code("""
st = json.load(open('output/preprocessing_stats.json', encoding='utf-8'))
for k in ['texts', 'sentences', 'tokens_all', 'tokens_alpha', 'tokens_content', 'unique_forms', 'unique_lemmas', 'fixes']:
    print(f'{k:<16}{st[k]}')
print('città is a spaCy stop word:', 'città' in nlp.Defaults.stop_words)
""")
md("""
**Effect:** removing stop words drops about half of the words; lemmatisation reduces the vocabulary
from 1646 forms to 1290 lemmas. Note: spaCy's Italian stop list contains *città*, a content word in our
corpus, so in Phase 2 we will use a custom stop list.
""")

md("""
## 9. Application area: a search engine for Italian cities

The user writes a question in Italian; the query goes through the **same preprocessing** (lowercase,
lemma, stop words removed); every corpus sentence is a **TF-IDF** vector of lemmas, and sentences are ranked
by **cosine similarity**. The same algorithm runs in the browser in `index.html` (web presentation).
""")
code("""
import math
from collections import Counter

ix = json.load(open('output/search_index.json', encoding='utf-8'))
sents = ix['sentences']
N = len(sents)
df = Counter(l for s in sents for l in set(s['lemmas']))
idf = lambda l: math.log((N + 1) / (df[l] + 1)) + 1

def vec(lemmas):
    tf = Counter(lemmas)
    v = {l: c * idf(l) for l, c in tf.items()}
    return v, math.sqrt(sum(x * x for x in v.values())) or 1

S = [vec(s['lemmas']) for s in sents]
STOP = set(ix['stop'])
F2L = ix['form2lemma']

def search(query, k=3):
    # same as the web demo: lowercase, form -> lemma from the corpus table, stop words removed
    words = re.findall(r"[a-zà-öø-ÿ]+", query.lower())
    q = [F2L.get(w, w) for w in words if w not in STOP and F2L.get(w, w) not in STOP]
    qv, qn = vec(q)
    scores = sorted(((sum(w * sv.get(l, 0) for l, w in qv.items()) / (qn * sn), i) for i, (sv, sn) in enumerate(S)), reverse=True)
    print('Query:', query, '->', q)
    for sc, i in scores[:k]:
        if sc > 0:
            print(f"  {sc:.3f}  {ix['docs'][sents[i]['doc']]['title']:<10} {sents[i]['text'][:100]}")

for q in ['università più antica', 'porto sul mare', 'patrimonio UNESCO', 'fondata dai Greci']:
    search(q)
""")
md("""
**Next (Phase 2):** a larger corpus (~100 cities, more sections), word vectors / sentence embeddings,
entity-aware ranking, a chatbot interface and an evaluation with test questions (precision@k).
""")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "nlp-tutorial1", "display_name": "Python (nlp-tutorial1)", "language": "python"}
nbf.write(nb, "NLP_Phase1_Italian.ipynb")
print("written", len(C), "cells")
