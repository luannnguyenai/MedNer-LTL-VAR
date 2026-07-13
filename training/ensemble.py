"""
ensemble.py — Gộp span từ NHIỀU model NER (khác tokenizer) ở mức SPAN.

Vì XLM-R (subword) và ViHealthBERT (syllable/word) tokenize khác nhau, KHÔNG gộp
được ở mức token. Ta gộp ở mức SPAN (char offset) — bất biến tokenizer:

  - Nhóm các span chồng lấn (theo overlap ký tự).
  - Bỏ phiếu TYPE (trọng số = weight_model * score).
  - Chọn BIÊN từ span có điểm cao nhất trong nhóm.
  - CONFIDENCE nhóm = tổng weight các model "bắn" vào đó.
      + Nếu >=2 model đồng thuận -> boost (giữ chắc).
      + Nếu chỉ 1 model và score thấp -> có thể lọc (giảm FP).
  - assertions/candidates: hợp (union) có kiểm soát.

ensemble_spans() là hàm THUẦN, test được bằng span giả (không cần model).
"""
from collections import defaultdict


def _overlap(a, b):
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def _group_overlaps(spans):
    """Union-Find đơn giản gom span chồng lấn thành nhóm."""
    n = len(spans)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    order = sorted(range(n), key=lambda i: spans[i]["start"])
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = spans[order[i]], spans[order[j]]
            if b["start"] >= a["end"]:
                break
            if _overlap(a, b):
                union(order[i], order[j])
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def ensemble_spans(span_lists, weights=None, min_conf=0.0, require_agree=False):
    """
    span_lists: list các list-span; mỗi span = dict có
        start,end,type,score(0..1),assertions[],candidates[]
    weights: trọng số mỗi model (mặc định 1.0).
    min_conf: ngưỡng confidence nhóm để giữ.
    require_agree: True -> chỉ giữ span >=2 model đồng thuận type.
    """
    k = len(span_lists)
    weights = weights or [1.0] * k
    tagged = []
    for mi, lst in enumerate(span_lists):
        for sp in lst:
            s = dict(sp)
            s["_m"] = mi
            s["_w"] = weights[mi]
            s.setdefault("score", 1.0)
            tagged.append(s)
    if not tagged:
        return []

    out = []
    for grp in _group_overlaps(tagged):
        members = [tagged[i] for i in grp]
        # bỏ phiếu type
        vote = defaultdict(float)
        models_for_type = defaultdict(set)
        for m in members:
            vote[m["type"]] += m["_w"] * m.get("score", 1.0)
            models_for_type[m["type"]].add(m["_m"])
        best_type = max(vote, key=vote.get)
        n_models = len({m["_m"] for m in members})
        n_agree = len(models_for_type[best_type])
        conf = vote[best_type]

        if require_agree and n_agree < 2:
            continue
        if conf < min_conf:
            continue

        # biên: span điểm cao nhất trong nhóm & cùng best_type (nếu có)
        cand = [m for m in members if m["type"] == best_type] or members
        rep = max(cand, key=lambda m: m["_w"] * m.get("score", 1.0))

        # union assertions (chỉ từ member cùng type), union candidates (giữ thứ tự)
        asserts = []
        for m in cand:
            for a in m.get("assertions", []):
                if a not in asserts:
                    asserts.append(a)
        cands = []
        for m in cand:
            for c in m.get("candidates", []):
                if c not in cands:
                    cands.append(c)

        out.append({
            "start": rep["start"], "end": rep["end"], "type": best_type,
            "assertions": asserts[:3], "candidates": cands,
            "score": conf, "n_agree": n_agree, "n_models": n_models,
        })
    out.sort(key=lambda x: x["start"])
    return out


if __name__ == "__main__":
    # TEST bằng span giả: 2 model, có đồng thuận + có bất đồng
    m1 = [
        {"start": 0, "end": 11, "type": "TRIỆU_CHỨNG", "score": 0.9,
         "assertions": [], "candidates": []},
        {"start": 20, "end": 29, "type": "CHẨN_ĐOÁN", "score": 0.6,
         "assertions": [], "candidates": ["J18.9"]},
        {"start": 40, "end": 48, "type": "THUỐC", "score": 0.8,
         "assertions": [], "candidates": ["1191"]},
    ]
    m2 = [
        {"start": 0, "end": 11, "type": "TRIỆU_CHỨNG", "score": 0.85,
         "assertions": ["isNegated"], "candidates": []},           # đồng thuận
        {"start": 20, "end": 29, "type": "TRIỆU_CHỨNG", "score": 0.55,
         "assertions": [], "candidates": []},                       # bất đồng type
        {"start": 60, "end": 68, "type": "TÊN_XÉT_NGHIỆM", "score": 0.7,
         "assertions": [], "candidates": []},                       # chỉ m2 có
    ]
    print("=== ensemble (weight 1:1, giữ tất cả) ===")
    for s in ensemble_spans([m1, m2], weights=[1.0, 1.0]):
        print(f"  [{s['start']:>2},{s['end']:>2}] {s['type']:16s} "
              f"conf={s['score']:.2f} agree={s['n_agree']}/{s['n_models']} "
              f"assert={s['assertions']} cand={s['candidates']}")
    print("\n=== chỉ giữ span >=2 model đồng thuận (require_agree) ===")
    for s in ensemble_spans([m1, m2], require_agree=True):
        print(f"  [{s['start']:>2},{s['end']:>2}] {s['type']:16s} conf={s['score']:.2f}")
    print("\n(Span [20,29] type do XLM-R nặng hơn quyết định; "
          "span đồng thuận [0,11] gộp assertion isNegated)")
