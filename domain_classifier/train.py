"""Обучение классификатора домена ТОЛЬКО на dev-сете.

eval_300_balanced.jsonl не открывается и не читается этим скриптом.

1. Сравнение наборов признаков и C на повторённой StratifiedGroupKFold
   (группы = split_group_id, чтобы варианты одного исходника не попадали
   одновременно в train и test).
2. Выбор порогов in_domain / unclear / out_of_domain по out-of-fold вероятностям.
3. Обучение финальной модели на всём dev и сохранение в model/.

Запуск:  .venv/bin/python train.py
"""
from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from domain_clf import MODEL_PATH, build_pipeline, case_texts

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "dev set" / "dev_300_balanced.jsonl"
OUT_DIR = MODEL_PATH.parent
N_SPLITS, N_REPEATS = 5, 5
CONFIGS = [(f, C) for f in ["lemma", "char", "navec", "lemma+char", "lemma+char+navec"] for C in (1.0, 4.0, 16.0)]

# Ручные пограничные примеры для качественной проверки (не dev, не eval, не метрика).
PROBES = [
    ("in", "Как в 1С:Полиграфии сформировать декларацию по НДС?"),
    ("in", "Сколько клиентских лицензий нужно купить на 15 пользователей?"),
    ("in", "Можно ли работать в программе через RDP?"),
    ("in", "Не открывается база после обновления платформы"),
    ("in", "Как рассчитать стоимость тиража листовок?"),
    ("in", "Почему в заказе не считается бумага?"),
    ("in", "Как связаться с вашей техподдержкой?"),
    ("in", "Где настроить права доступа для менеджера?"),
    ("out", "Какие суммы мне вписать в налоговую декларацию за квартал?"),
    ("out", "Посоветуй хорошую типографию в Москве"),
    ("out", "Напиши стихотворение про осень"),
    ("out", "Как по закону оформить декретный отпуск сотруднице?"),
    ("out", "Какая погода будет в выходные в Питере?"),
    ("out", "Как установить Windows 11 на ноутбук?"),
]


MAX_FALSE_REFUSAL = 0.01   # доля доменных, которые можно отправить в out_of_domain (C05)
MAX_OOD_ACCEPTED = 0.02    # доля посторонних, которые можно принять как in_domain


def load_dev():
    """Обучающая выборка: K1 = 1, прочие классы = 0. Кейсы с тегом domain_unclear
    (K3: неясно, о программе ли вопрос) в обучение и подбор порогов не входят —
    правильный ответ маршрутизатора для них «unclear», а не «in_domain»."""
    assert "eval" not in str(DEV).lower()
    rows = [json.loads(l) for l in DEV.open(encoding="utf-8")]
    assert {r["split"] for r in rows} == {"dev"}, "ожидается только dev"
    train = [r for r in rows if "domain_unclear" not in r["tags"]]
    unclear = [r for r in rows if "domain_unclear" in r["tags"]]
    msgs, ctxs = zip(*(case_texts(r) for r in train))
    y = np.array([int(r["primary_class"] == "K1") for r in train])
    groups = np.array([r["split_group_id"] for r in train])
    return train, list(msgs), list(ctxs), y, groups, unclear


def oof_scores(texts, y, groups, feats, C, ctxs=None, extra=None):
    """Средняя по повторам out-of-fold вероятность p(out_of_domain) + метрики по повторам.
    ctxs — тексты с историей (оцениваются моделью того же фолда); extra — тексты вне
    обучения (оцениваются всеми моделями фолдов, затем усредняются)."""
    X = np.array(texts, dtype=object)
    oof = np.zeros((N_REPEATS, len(y)))
    oof_ctx = np.zeros((N_REPEATS, len(y)))
    ext = np.zeros(len(extra or []))
    n_models = 0
    for rep in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=rep)
        for tr, te in cv.split(X, y, groups):
            pipe = build_pipeline(feats, C).fit(list(X[tr]), y[tr])
            oof[rep, te] = pipe.predict_proba(list(X[te]))[:, 1]
            if ctxs is not None:
                oof_ctx[rep, te] = pipe.predict_proba([ctxs[i] for i in te])[:, 1]
            if extra:
                ext += pipe.predict_proba(extra)[:, 1]
                n_models += 1
    if ctxs is not None or extra:
        return oof.mean(axis=0), oof_ctx.mean(axis=0), (ext / max(n_models, 1))
    per_rep = {
        "roc_auc": [roc_auc_score(y, o) for o in oof],
        "pr_auc": [average_precision_score(y, o) for o in oof],
        "recall_out@0.5": [float(((o >= .5) & (y == 1)).sum() / y.sum()) for o in oof],
        "false_refusal@0.5": [float(((o >= .5) & (y == 0)).sum() / (y == 0).sum()) for o in oof],
    }
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in per_rep.items()}
    return oof.mean(axis=0), summary


def route(p, t_in, t_out):
    return np.where(p >= t_out, "out_of_domain", np.where(p < t_in, "in_domain", "unclear"))


def routed(p, p_ctx, has_hist, t_in, t_out):
    """Логика DomainClassifier.predict: при unclear и наличии истории — оценка с историей."""
    lab = route(p, t_in, t_out)
    lab2 = route(p_ctx, t_in, t_out)
    return np.where((lab == "unclear") & has_hist, lab2, lab)


def three_way(lab, y):
    return {true: dict(Counter(lab[y == v])) for true, v in (("in_domain", 0), ("out_of_domain", 1))}


def pick_thresholds(p, y):
    """t_out — минимальный порог с долей ложных отказов <= MAX_FALSE_REFUSAL;
    t_in — максимальный порог с долей принятых посторонних <= MAX_OOD_ACCEPTED."""
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    p_in, p_ood = p[y == 0], p[y == 1]
    t_out = next(t for t in grid if (p_in >= t).mean() <= MAX_FALSE_REFUSAL)
    t_in = max(t for t in grid if (p_ood < t).mean() <= MAX_OOD_ACCEPTED)
    t_in = min(t_in, t_out)
    return float(t_in), float(t_out)


def main():
    rows, msgs, ctxs, y, groups, unclear_rows = load_dev()
    print(f"dev: {len(y)} кейсов для обучения, out_of_domain (K1) = {y.sum()}, групп = {len(set(groups))}; "
          f"domain_unclear вне обучения: {len(unclear_rows)}")
    print(f"CV: StratifiedGroupKFold {N_SPLITS} фолдов x {N_REPEATS} повторов\n")

    results = []
    for feats, C in CONFIGS:
        _, s = oof_scores(msgs, y, groups, feats, C)
        results.append({"features": feats, "C": C, **s})
        print(f"{feats:18s} C={C:<5} PR-AUC {s['pr_auc']['mean']:.4f}±{s['pr_auc']['std']:.4f}  "
              f"ROC-AUC {s['roc_auc']['mean']:.4f}  recall@.5 {s['recall_out@0.5']['mean']:.3f}  "
              f"false_refusal@.5 {s['false_refusal@0.5']['mean']:.4f}")

    # Лучшая: макс. PR-AUC; при равенстве (±0.002) — меньше ложных отказов, затем более простая.
    top = max(r["pr_auc"]["mean"] for r in results)
    cands = [r for r in results if r["pr_auc"]["mean"] >= top - 0.002]
    best = min(cands, key=lambda r: (r["false_refusal@0.5"]["mean"], -r["recall_out@0.5"]["mean"],
                                     r["features"].count("+")))
    feats, C = best["features"], best["C"]
    print(f"\nВыбрано: {feats}, C={C}")

    unclear_msgs = [case_texts(r)[0] for r in unclear_rows]
    p, p_ctx, p_unc = oof_scores(msgs, y, groups, feats, C, ctxs=ctxs, extra=unclear_msgs)
    p_in, p_ood = p[y == 0], p[y == 1]
    has_hist = np.array([bool(r["input"]["history"]) for r in rows])
    t_in, t_out = pick_thresholds(p, y)
    lab = routed(p, p_ctx, has_hist, t_in, t_out)
    tw = three_way(lab, y)
    print(f"Пороги: in_domain < {t_in:.2f} <= unclear < {t_out:.2f} <= out_of_domain "
          f"(ложные отказы <= {MAX_FALSE_REFUSAL:.0%}, принятые посторонние <= {MAX_OOD_ACCEPTED:.0%})")
    print("Трёхклассовая OOF-матрица (с учётом истории):", json.dumps(tw, ensure_ascii=False))
    # ---- слой правил поверх OOF-решения модели
    from rules import apply_rules
    ruled = [apply_rules(m, l) for m, l in zip(msgs, lab)]
    lab_model = lab
    lab = np.array([l for l, _ in ruled])
    fired = Counter((rule, "K1" if yy else "domain", before, after)
                    for (after, rule), before, yy in zip(ruled, lab_model, y) if rule)
    tw_model = tw
    tw = three_way(lab, y)
    print("Только модель:", json.dumps(tw_model, ensure_ascii=False))
    print("Модель + правила:", json.dumps(tw, ensure_ascii=False))
    print("Срабатывания правил (правило, истинный класс, было, стало):", dict(fired))
    errors = [(rows[i]["case_id"], "K1" if y[i] else "domain", str(lab[i]), msgs[i][:100]) for i in range(len(y))
              if (y[i] == 1 and lab[i] == "in_domain") or (y[i] == 0 and lab[i] == "out_of_domain")]
    print("Грубые ошибки (посторонний -> in_domain, доменный -> out_of_domain):", *errors, sep="\n  ")
    lab_unc_model = route(p_unc, t_in, t_out)
    lab_unc = np.array([apply_rules(m, l)[0] for m, l in zip(unclear_msgs, lab_unc_model)])
    unc_report = [(r["case_id"], round(float(pp), 3), l, m) for r, pp, l, m in zip(unclear_rows, p_unc, lab_unc, unclear_msgs)]
    print("domain_unclear (ожидается unclear; допустимо in_domain, недопустимо out_of_domain):",
          dict(Counter(lab_unc)))
    # по категориям пограничных кейсов
    cats = Counter()
    for r, l, yy in zip(rows, lab, y):
        if r["origin_type"] == "synthetic_domain_boundary":
            cats[(r["origin"].split(":")[1], l)] += 1
    by_cat = {}
    for (c, l), n_ in sorted(cats.items()):
        by_cat.setdefault(c, {})[l] = n_
    print("Пограничные кейсы по категориям:", json.dumps(by_cat, ensure_ascii=False))

    # Ошибки/пограничные по OOF
    order = np.argsort(-p)
    hard_in = [(rows[i]["case_id"], round(float(p[i]), 3), msgs[i][:110]) for i in order if y[i] == 0][:8]
    hard_out = [(rows[i]["case_id"], round(float(p[i]), 3), msgs[i][:110]) for i in order[::-1] if y[i] == 1][:8]
    print("\nДоменные с наибольшим p_out:", *hard_in, sep="\n  ")
    print("Посторонние с наименьшим p_out:", *hard_out, sep="\n  ")

    # Короткие ответы в многоходовых диалогах: сообщение отдельно vs с историей
    multi = [i for i, r in enumerate(rows) if r["input"]["history"]]

    final = build_pipeline(feats, C).fit(msgs, y)
    bundle = {"pipeline": final, "t_in": t_in, "t_out": t_out,
              "meta": {"trained_on": str(DEV.relative_to(ROOT)),
                       "dev_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest(),
                       "n": int(len(y)), "n_out": int(y.sum()),
                       "features": feats, "C": C}}
    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    from domain_clf import DomainClassifier
    clf = DomainClassifier(bundle)
    probes = [{"expected": e, "text": t, **clf.predict(t)} for e, t in PROBES]
    print("\nРучные пограничные примеры (не метрика):")
    for pr in probes:
        print(f"  [{pr['expected']:3s}] {pr['label']:13s} {pr['p_out_of_domain']:.3f}  {pr['text']}")

    report = {"dev_file": str(DEV.relative_to(ROOT)), "n": int(len(y)), "n_out_of_domain": int(y.sum()),
              "n_groups": int(len(set(groups))), "cv": f"StratifiedGroupKFold {N_SPLITS}x{N_REPEATS}, groups=split_group_id",
              "results": results, "selected": {"features": feats, "C": C},
              "thresholds": {"t_in": t_in, "t_out": t_out, "max_false_refusal": MAX_FALSE_REFUSAL,
                             "max_ood_accepted": MAX_OOD_ACCEPTED},
              "oof_three_way_model_only": tw_model, "oof_three_way": tw,
              "rules_fired": [list(k) + [v] for k, v in fired.items()], "gross_errors": errors,
              "domain_unclear_model_only": dict(Counter(lab_unc_model.tolist())),
              "domain_unclear_routing": unc_report, "boundary_by_category": by_cat,
              "oof_p_range": {"in_domain_max": float(p_in.max()), "out_of_domain_min": float(p_ood.min())},
              "hardest_in_domain": hard_in, "hardest_out_of_domain": hard_out,
              "multi_turn_cases": len(multi), "probes": probes}
    (OUT_DIR / "cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nМодель: {MODEL_PATH.relative_to(ROOT)}; отчёт: {(OUT_DIR / 'cv_report.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
