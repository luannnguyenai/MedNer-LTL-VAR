"""
retriever_data.py — PHASE 3: khai mỏ TRIPLET tương phản cho fine-tune BGE-m3.

Mỗi triplet: (anchor=mention, positive=tên KB đúng, negative=tên KB SAI-nhưng-gần).
Hard negative = ứng viên fuzzy của linker KHÁC code gold (khó phân biệt nhất).

Hai nguồn positive:
  (1) GOLD link: span có candidates (mã đúng) -> tên KB của mã đó.
  (2) SYNTH từ KB: lấy tên chuẩn, tạo "mention" nhiễu (bỏ dấu/rút gọn/đảo)
      -> positive = tên gốc; cung cấp lượng lớn cặp không cần người gán.

Xuất JSONL: {"anchor","positive","negative"} (negative optional; MNRL dùng
in-batch negatives + hard negative nếu có).
"""
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vntext import strip_accents  # noqa

random.seed(11)


def _noisy(name):
    """Tạo mention nhiễu từ tên KB chuẩn (mô phỏng biểu đạt thực tế)."""
    s = name
    r = random.random()
    if r < 0.3:
        s = strip_accents(s)                      # gõ không dấu
    elif r < 0.5:
        s = re.sub(r"\s*[-,].*$", "", s)          # cắt phần sau dấu -,
    elif r < 0.65:
        toks = s.split()
        if len(toks) > 3:
            s = " ".join(toks[:max(2, len(toks) // 2)])   # rút gọn
    return s.strip()


def from_gold(unified_jsonl, icd=None, rx=None, topk=8):
    """Nguồn (1): span có gold code -> triplet với hard negative từ linker."""
    rows = [json.loads(l) for l in open(unified_jsonl, encoding="utf-8") if l.strip()]
    out = []
    for ex in rows:
        for s in ex["spans"]:
            gold = s.get("candidates") or []
            if not gold:
                continue
            if s.get("fine") == "CHẨN_ĐOÁN" and icd is not None:
                pos_name = icd._name(gold[0])
                cand = icd.link(s["text"], topk=topk) or []
                neg_names = [icd._name(c) for c in cand if c not in set(gold)]
            elif s.get("fine") == "THUỐC" and rx is not None:
                pos_name = rx.by_rxcui.get(gold[0], "")
                cand = rx.link(s["text"], topk=topk) or []
                neg_names = [rx.by_rxcui.get(c, "") for c in cand if c not in set(gold)]
            else:
                continue
            if not pos_name:
                continue
            neg = neg_names[0] if neg_names else None
            row = {"anchor": s["text"], "positive": pos_name}
            if neg:
                row["negative"] = neg
            out.append(row)
    return out


def from_kb_synth(names, n=20000, hard_neg_names=None):
    """Nguồn (2): tên KB -> (mention nhiễu, tên gốc) + hard negative ngẫu nhiên."""
    out = []
    pool = names if len(names) <= n else random.sample(names, n)
    for nm in pool:
        anc = _noisy(nm)
        if not anc or anc == nm and random.random() < 0.5:
            anc = nm
        row = {"anchor": anc, "positive": nm}
        if hard_neg_names and len(hard_neg_names) > 1:
            neg = random.choice(hard_neg_names)
            tries = 0
            while neg == nm and tries < 5:
                neg = random.choice(hard_neg_names)
                tries += 1
            if neg != nm:
                row["negative"] = neg
        out.append(row)
    return out


def write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[retriever_data] {len(rows)} triplet -> {path}")


if __name__ == "__main__":
    # TEST bằng mock linker (không tải KB thật)
    class MockICD:
        _names = {"K21.0": "Bệnh trào ngược dạ dày-thực quản có viêm thực quản",
                  "K21.9": "Bệnh trào ngược dạ dày-thực quản không viêm thực quản",
                  "K20": "Viêm thực quản", "J18.9": "Viêm phổi không xác định"}

        def _name(self, c): return self._names.get(c, "")
        def link(self, t, topk=8): return ["K21.0", "K21.9", "K20"]

    demo = [{"text": "trào ngược dạ dày thực quản", "source": "d",
             "spans": [{"text": "trào ngược dạ dày thực quản", "fine": "CHẨN_ĐOÁN",
                        "candidates": ["K21.0"]}]}]
    open("/tmp/u.jsonl", "w").write(json.dumps(demo[0], ensure_ascii=False) + "\n")

    print("=== from_gold (hard negative từ linker) ===")
    for r in from_gold("/tmp/u.jsonl", icd=MockICD()):
        print(" ", r)

    print("\n=== from_kb_synth (mention nhiễu từ tên KB) ===")
    names = list(MockICD._names.values())
    for r in from_kb_synth(names, n=4, hard_neg_names=names)[:4]:
        print(" ", r)
    print("\nTriplet OK — dùng cho MultipleNegativesRankingLoss (train_retriever.py)")
