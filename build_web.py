"""Build the web presentation index.html (one self-contained file, works offline).

Needs: output/preprocessing_stats.json and output/search_index.json (run preprocess.py first)
       images/sentence1_dep.svg ... sentence3_dep.svg (from the notebook)
Run:   python build_web.py
"""
import json
import re
from pathlib import Path

REPO = "https://github.com/Laura-Z-hub/nlp-italian-cities"

tpl = Path("web/template.html").read_text(encoding="utf-8")
stats = Path("output/preprocessing_stats.json").read_text(encoding="utf-8")
index = Path("output/search_index.json").read_text(encoding="utf-8")


def svg(name):
    s = Path("images", name).read_text(encoding="utf-8")
    return re.sub(r"^<\?xml[^>]*>\s*", "", s.strip())


def js(s):
    return s.replace("</", "<\\/")      # never close the <script> tag by accident


html = (tpl.replace("__STATS__", js(stats))
           .replace("__INDEX__", js(index))
           .replace("__SVG1__", svg("sentence1_dep.svg"))
           .replace("__SVG2__", svg("sentence2_dep.svg"))
           .replace("__SVG3__", svg("sentence3_dep.svg"))
           .replace("__REPO_SHORT__", REPO.replace("https://", ""))
           .replace("__REPO__", REPO))
Path("index.html").write_text(html, encoding="utf-8", newline="\n")
print(f"index.html written ({len(html) // 1024} KB)")
