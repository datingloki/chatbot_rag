"""Стабильные ссылки на источники в кейсах набора (замечание ревью КТ2).

    .venv/bin/python -m harness.normalize_evidence                 # dev: отчёт, файл не меняется
    .venv/bin/python -m harness.normalize_evidence --write         # dev: записать
    .venv/bin/python -m harness.normalize_evidence --set eval      # eval: только с HARNESS_ALLOW_EVAL=1
    .venv/bin/python -m harness.normalize_evidence --file "../all cases (q&a)/all_cases.jsonl"
                                                   # любой файл кейсов; не dev — с HARNESS_ALLOW_EVAL=1
                                                   # (all_cases содержит и кейсы eval)

В `evidence` кейсов источник был указан абсолютным путём на личном компьютере
(`/Users/<имя>/Downloads/…doc`), которого у проверяющего нет. Скрипт к каждой ссылке добавляет:

- `source_id` — стабильный ID записи, тот же, что в индексе бота и в ответах:
  `faq:<id>`, `manual:<номер раздела>`, `omnidesk:<id обращения>`; для оснований разметки —
  `guide:annotation_guide_classes`, `design_doc:KT-2`;
- `source_role` — `knowledge_base` (факт из базы знаний) или `labeling_rule` (правило разметки:
  инструкция, design doc — по ним проверяется маршрут, а не содержание ответа);
- `corpus_path` — путь от корня репозитория к файлу корпуса, где лежит запись;
- `source_version` — версия записи: для FAQ — sha256 текста статьи и дата выгрузки, для
  руководства — редакция и sha256 `.docx`, для OmniDesk и правил — коммит файла;
- `record_check` — найдена ли запись в корпусе и совпадает ли с размеченной
  (`ok`, `text_changed`, `line_fixed`, `not_found`, `section_not_found`).

`source_file` заменяется путём от корня репозитория. Номер строки jsonl исправляется, если
запись переехала. Для руководства раздел определяется по выдержке (`excerpt`): ищется раздел
`.docx`, в чанке которого есть ≥ 75% слов выдержки, — так же, как эталон считают харнесс и
retrieval_eval. Полный текст раздела показывает `python -m harness.show_source manual:12.3`:
по выдержке видно отдельный тезис, а применимость и исключения — только в разделе целиком.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from ingest.documents import FAQ_FILES, KB_DIR, MANUAL_FILE, MANUAL_RELEASE, OMNIDESK_FILE
from polybot.settings import REPO_DIR

SETS = {"dev": REPO_DIR / "dev set" / "dev_300_balanced.jsonl",
        "eval": REPO_DIR / "eval set" / "eval_300_balanced.jsonl"}
KB_REL = KB_DIR.relative_to(REPO_DIR)
MANUAL_DOC = "1С УНФ 8  Полиграфия 2 для 3.0.doc"          # исходник в git; индекс строится по .docx
GUIDE = Path("all cases (q&a)") / "annotation_guide_classes.md"
DESIGN_DOC = Path("design_doc_KT-2.md")
COVERAGE = 0.75


def rel(p: Path) -> str:
    return p.as_posix()


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def git_commit(path: Path) -> str:
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(path)], cwd=REPO_DIR,
                         capture_output=True, text=True).stdout.strip()
    return out or "uncommitted"


class Corpus:
    """Записи корпуса по ID: FAQ (с номером строки и текстом), обращения OmniDesk, разделы руководства."""

    def __init__(self) -> None:
        self.faq: dict[str, tuple[str, int, str]] = {}          # id → (файл, строка, текст)
        for name in FAQ_FILES:
            for i, line in enumerate((KB_DIR / name).open(encoding="utf-8"), 1):
                if line.strip():
                    r = json.loads(line)
                    self.faq[r["id"]] = (name, i, r.get("text", ""))
        self.omni: dict[str, int] = {}
        for i, line in enumerate((KB_DIR / OMNIDESK_FILE).open(encoding="utf-8"), 1):
            if line.strip():
                self.omni[str(json.loads(line)["source_case_id"])] = i
        self._manual = None
        self.versions = {
            "omnidesk": f"git:{git_commit(KB_DIR / OMNIDESK_FILE)}, sha256:{sha12(KB_DIR / OMNIDESK_FILE)}",
            "guide": f"git:{git_commit(REPO_DIR / GUIDE)}",
            "design_doc": f"git:{git_commit(REPO_DIR / DESIGN_DOC)}",
        }
        docx = KB_DIR / MANUAL_FILE
        self.versions["manual"] = (f"ред. {MANUAL_RELEASE}, sha256:{sha12(docx)}"
                                   + ("" if git_commit(docx) != "uncommitted" else " (.docx не в git)"))

    def manual_sections(self):
        if self._manual is None:
            from harness.retrieval_eval import words
            from ingest.chunking import chunk_document
            from ingest.documents import load_manual

            docs = load_manual()
            self.titles = {d.doc_id: d.title for d in docs}
            self._manual = [(c.doc_id, words(c.text)) for d in docs for c in chunk_document(d)]
        return self._manual

    def find_section(self, excerpt: str) -> tuple[str | None, float]:
        from harness.retrieval_eval import words

        w = words(excerpt)
        if len(w) < 4:
            return None, 0.0
        best, score = None, 0.0
        for doc_id, cw in self.manual_sections():
            cov = len(w & cw) / len(w)
            if cov > score:
                best, score = doc_id, cov
        return (best, score) if score >= COVERAGE else (None, score)


def kind_of(e: dict[str, Any]) -> str:
    f = (e.get("source_file") or "").lower()
    if e.get("faq_id"):
        return "faq"
    if e.get("source_case_id"):
        return "omnidesk"
    if f.endswith((".doc", ".docx")) or e.get("excerpt"):
        return "manual"
    if "annotation_guide" in f:
        return "guide"
    if "design_doc" in f:
        return "design_doc"
    return "none"


def normalize(e: dict[str, Any], corpus: Corpus) -> str:
    """Дополняет одну ссылку; возвращает record_check (или '' для записей без файла)."""
    kind = kind_of(e)
    if kind == "none":
        return ""
    if kind == "faq":
        fid = e["faq_id"]
        e["source_id"], e["source_role"] = f"faq:{fid}", "knowledge_base"
        hit = corpus.faq.get(fid)
        if hit is None:
            e["record_check"] = "not_found"
        else:
            name, line, text = hit
            e["source_file"] = e["corpus_path"] = rel(KB_REL / name)
            check = "ok"
            if str(e.get("source_jsonl_line")) != str(line):
                e["source_jsonl_line"], check = str(line), "line_fixed"
            if e.get("content_sha256") and hashlib.sha256(text.encode()).hexdigest() != e["content_sha256"]:
                check = "text_changed"
            e["record_check"] = check
        e["source_version"] = f"sha256:{(e.get('content_sha256') or '')[:12]}, saved_at {str(e.get('saved_at', ''))[:10]}"
        # картинки статьи: локальная копия была только на компьютере автора, публичный url остаётся
        for img in e.get("image_refs") or []:
            if isinstance(img, dict):
                img.pop("local_file", None)
    elif kind == "omnidesk":
        cid = str(e["source_case_id"])
        e["source_id"], e["source_role"] = f"omnidesk:{cid}", "knowledge_base"
        e["source_file"] = e["corpus_path"] = rel(KB_REL / OMNIDESK_FILE)
        e["source_version"] = corpus.versions["omnidesk"]
        line = corpus.omni.get(cid)
        if line is None:
            e["record_check"] = "not_found"
        elif str(e.get("source_jsonl_line")) != str(line):
            e["source_jsonl_line"], e["record_check"] = str(line), "line_fixed"
        else:
            e["record_check"] = "ok"
    elif kind == "manual":
        e["source_role"] = "knowledge_base"
        e["source_file"] = rel(KB_REL / MANUAL_DOC)
        e["corpus_path"] = rel(KB_REL / MANUAL_FILE)
        e["source_version"] = corpus.versions["manual"]
        sec, cov = corpus.find_section(e.get("excerpt", ""))
        if sec:
            e["source_id"], e["section_title"] = sec, corpus.titles.get(sec, "")
            e["record_check"] = "ok"
        else:
            e["source_id"], e["record_check"] = "manual:?", "section_not_found"
            e["excerpt_coverage"] = round(cov, 2)
    elif kind == "guide":
        e["source_id"], e["source_role"] = "guide:annotation_guide_classes", "labeling_rule"
        e["source_file"] = e["corpus_path"] = rel(GUIDE)
        e["source_version"], e["record_check"] = corpus.versions["guide"], "ok"
    elif kind == "design_doc":
        e["source_id"], e["source_role"] = "design_doc:KT-2", "labeling_rule"
        e["source_file"] = e["corpus_path"] = rel(DESIGN_DOC)
        e["source_version"], e["record_check"] = corpus.versions["design_doc"], "ok"
    return e.get("record_check", "")


COMPACT = (",", ":")


def line_format(lines: list[str]) -> tuple[str, str] | None:
    """Формат строк файла, чтобы перезапись не меняла ничего, кроме ссылок: dev и eval записаны
    компактно, all_cases — с пробелами (по умолчанию json.dumps)."""
    first = lines[0].rstrip("\n")
    for sep in (COMPACT, None):
        if json.dumps(json.loads(first), ensure_ascii=False, separators=sep) == first:
            return sep
    return None


def run(path: Path, write: bool) -> dict[str, Any]:
    corpus = Corpus()
    lines = [l for l in path.open(encoding="utf-8") if l.strip()]
    sep = line_format(lines)
    rows = [json.loads(l) for l in lines]
    checks: Counter = Counter()
    kinds: Counter = Counter()
    problems = []
    for r in rows:
        # в all_cases (выгрузка из xlsx) evidence — JSON-строка, а не список
        encoded = isinstance(r.get("evidence"), str)
        items = json.loads(r["evidence"]) if encoded else (r.get("evidence") or [])
        for e in items:
            k = kind_of(e)
            kinds[k] += 1
            c = normalize(e, corpus)
            if c:
                checks[c] += 1
                if c not in ("ok", "line_fixed"):
                    problems.append((r["case_id"], e.get("source_id"), c))
        if encoded:
            r["evidence"] = json.dumps(items, ensure_ascii=False, separators=COMPACT)
    absolute = sum("/Users/" in json.dumps(r, ensure_ascii=False) for r in rows)
    if write:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False, separators=sep) + "\n" for r in rows),
                       encoding="utf-8")
        tmp.replace(path)
    return {"file": rel(path.relative_to(REPO_DIR)), "cases": len(rows), "kinds": dict(kinds),
            "record_check": dict(checks), "cases_with_absolute_paths_after": absolute,
            "problems": problems, "versions": corpus.versions, "written": write}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="dev")
    ap.add_argument("--file", type=Path, help="произвольный jsonl с кейсами вместо --set")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    path = (args.file.resolve() if args.file else SETS[args.set])
    if path != SETS["dev"] and os.environ.get("HARNESS_ALLOW_EVAL") != "1":
        raise SystemExit("кроме dev — только с HARNESS_ALLOW_EVAL=1 (eval закрыт до КТ4, all_cases содержит eval)")
    report = run(path, args.write)
    problems = report.pop("problems")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for p in problems[:40]:
        print("  проблема:", *p)
    if len(problems) > 40:
        print(f"  … ещё {len(problems) - 40}")


if __name__ == "__main__":
    main()
