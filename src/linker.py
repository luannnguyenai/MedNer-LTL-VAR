"""
linker.py — Bộ ánh xạ (entity linking) DETERMINISTIC sang ICD-10 / RxNorm.

TRIẾT LÝ CHỐNG HALLUCINATION:
  Mọi mã trả ra ĐỀU tồn tại trong KB đã cung cấp. LLM/NER chỉ đưa ra *chuỗi text*;
  linker này (không dùng model sinh) mới quyết định mã. Nếu không đủ tự tin -> trả []
  thay vì bịa mã.

Cơ chế:
  1) Exact match trên norm_key.
  2) Fuzzy: inverted index char-trigram -> ứng viên chia sẻ n-gram; chấm điểm bằng
     kết hợp Jaccard(trigram) + Dice(token) + bonus chứa trọn cụm.
  3) Ngưỡng chấp nhận cấu hình được; trả về top-k mã (khử trùng lặp).

Riêng thuốc: chuẩn hóa "biến thể muối" (maleate/hydrochloride/sodium...) và tách
  hàm lượng để tăng recall khi text = "Chlorpheniramine 0.4 MG/ML".
"""
import json
import os
import re
from collections import defaultdict

from vntext import norm_key, norm_key_loose, char_ngrams, tokens

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "kb")

# Hậu tố muối/ester thường gặp -> bỏ khi so khớp lỏng
_SALTS = ["maleate", "hydrochloride", "hcl", "sodium", "sulfate", "sulphate",
          "phosphate", "succinate", "tartrate", "besylate", "mesylate",
          "citrate", "acetate", "fumarate", "bitartrate", "potassium",
          "calcium", "hydrobromide", "nitrate", "dihydrate", "monohydrate"]
_STRENGTH = re.compile(r"\d+([.,]\d+)?\s*(mg|ml|mcg|g|unit|iu|%|mg/ml|mcg/ml|mg/actuat)?", re.I)


def _strip_salts(key: str) -> str:
    for s in _SALTS:
        key = re.sub(r"\b" + s + r"\b", " ", key)
    return re.sub(r"\s+", " ", key).strip()


class FuzzyIndex:
    """Inverted char-trigram index cho fuzzy retrieval nhanh."""

    def __init__(self, entries, text_field, code_field):
        self.entries = entries
        self.text_field = text_field
        self.code_field = code_field
        self.inv = defaultdict(list)     # trigram -> [entry_idx]
        self.grams = []                  # entry_idx -> set(trigram)
        self.toks = []                   # entry_idx -> set(token)
        for i, e in enumerate(entries):
            g = char_ngrams(e[text_field], 3)
            self.grams.append(g)
            self.toks.append(set(tokens(e[text_field])))
            for tg in g:
                self.inv[tg].append(i)

    def candidates(self, query, cap=400):
        qg = char_ngrams(query, 3)
        if not qg:
            return []
        counts = defaultdict(int)
        for tg in qg:
            lst = self.inv.get(tg)
            if not lst:
                continue
            # bỏ trigram quá phổ biến để tránh nổ ứng viên
            if len(lst) > 8000:
                continue
            for i in lst:
                counts[i] += 1
        # lấy các entry chia sẻ nhiều trigram nhất
        ranked = sorted(counts.items(), key=lambda x: -x[1])[:cap]
        return [i for i, _ in ranked]

    def score(self, query_key, i, entry_text=None):
        qg = char_ngrams(query_key, 3)
        qt = set(tokens(query_key))
        g = self.grams[i]
        t = self.toks[i]
        jac = len(qg & g) / len(qg | g) if (qg or g) else 0.0
        dice = 2 * len(qt & t) / (len(qt) + len(t)) if (qt or t) else 0.0
        # chứa trọn token: mọi token query nằm trong entry (rất mạnh cho cụm ngắn)
        contain = 1.0 if qt and qt <= t else 0.0
        # substring ở mức ký tự: tên bệnh bắt đầu / chứa cụm query nguyên vẹn
        sub = 0.0
        if entry_text is not None and query_key:
            if entry_text.startswith(query_key):
                sub = 1.0
            elif query_key in entry_text:
                sub = 0.7
        # coverage: bao nhiêu phần token query được entry phủ (0..1)
        cov = len(qt & t) / len(qt) if qt else 0.0
        return 0.40 * jac + 0.22 * dice + 0.18 * contain + 0.12 * sub + 0.08 * cov


class ICDLinker:
    def __init__(self, path=None):
        path = path or os.path.join(KB_DIR, "icd_index.json")
        with open(path, encoding="utf-8") as f:
            kb = json.load(f)
        self.entries = kb["entries"]
        self.exact = kb["exact"]
        # map mã 3 ký tự -> danh sách mã lá con (K21 -> [K21.0, K21.9])
        self.children = {}
        for e in self.entries:
            c = e["code"]
            if "." in c:
                self.children.setdefault(c.split(".")[0], []).append(c)
        # index trên tên VN (chính) và EN
        self.fx_vn = FuzzyIndex([e for e in self.entries if e["key_vn"]],
                                "key_vn", "code")
        self.fx_en = FuzzyIndex([e for e in self.entries if e["key_en"]],
                                "key_en", "code")

    def link(self, text, topk=3, thr=0.62, expand_leaves=True):
        k = norm_key(text)
        if not k:
            return []
        # 1) exact
        if k in self.exact:
            codes = list(self.exact[k])
            # nếu là mã 3 ký tự & có con -> ưu tiên trả mã lá (cụ thể hơn)
            if expand_leaves:
                out = []
                for c in codes:
                    if "." not in c and c in self.children:
                        out.extend(self.children[c][:topk])
                    else:
                        out.append(c)
                # khử trùng, giữ thứ tự
                seen = set(); ded = []
                for c in out:
                    if c not in seen:
                        seen.add(c); ded.append(c)
                return ded[:max(topk, 2)]
            return codes[:topk]
        # 2) fuzzy trên VN rồi EN
        best = {}  # code -> score
        for fx, fld in ((self.fx_vn, "key_vn"), (self.fx_en, "key_en")):
            for i in fx.candidates(k):
                s = fx.score(k, i, fx.entries[i][fld])
                if s >= thr:
                    c = fx.entries[i]["code"]
                    if s > best.get(c, 0):
                        best[c] = s
        ranked = sorted(best.items(), key=lambda x: -x[1])
        return [c for c, _ in ranked[:topk]]

    def explain(self, text, topk=5, thr=0.0):
        """Debug: trả (code, name_vn, score)."""
        k = norm_key(text)
        rows = []
        seen = set()
        if k in self.exact:
            for c in self.exact[k]:
                rows.append((c, self._name(c), 1.0))
                seen.add(c)
        for fx, fld in ((self.fx_vn, "key_vn"), (self.fx_en, "key_en")):
            for i in fx.candidates(k):
                s = fx.score(k, i, fx.entries[i][fld])
                c = fx.entries[i]["code"]
                if c in seen:
                    continue
                rows.append((c, fx.entries[i].get("vn") or fx.entries[i].get("en"), s))
                seen.add(c)
        rows.sort(key=lambda x: -x[2])
        return rows[:topk]

    def _name(self, code):
        for e in self.entries:
            if e["code"] == code:
                return e["vn"] or e["en"]
        return ""


class RxNormLinker:
    def __init__(self, path=None):
        path = path or os.path.join(KB_DIR, "rxnorm_index.json")
        with open(path, encoding="utf-8") as f:
            kb = json.load(f)
        self.entries = kb["entries"]
        self.exact = kb["exact"]
        self.by_rxcui = kb["by_rxcui"]
        self.fx = FuzzyIndex(self.entries, "key", "rxcui")
        # index phụ theo khóa đã bỏ muối để bắt biến thể
        self.loose = defaultdict(list)   # key_no_salt -> [rxcui]
        for e in self.entries:
            lk = _strip_salts(e["key"])
            if lk and lk != e["key"]:
                if e["rxcui"] not in self.loose[lk]:
                    self.loose[lk].append(e["rxcui"])

    def link(self, text, topk=3, thr=0.60):
        k = norm_key(text)
        if not k:
            return []
        # 1) exact
        if k in self.exact:
            return self.exact[k][:topk]
        # 2) exact sau khi bỏ muối (text "chlorpheniramine 0.4 mg/ml"
        #    khớp entry "chlorpheniramine maleate 0.4 mg/ml")
        ks = _strip_salts(k)
        # thử map: chèn từng muối vào giữa ingredient và strength? -> thay vì vậy,
        # so khớp lỏng: tìm entry mà bỏ muối == ks
        if ks in self.loose:
            return self.loose[ks][:topk]
        # 3) fuzzy
        best = {}
        for i in self.fx.candidates(k):
            s = self.fx.score(k, i, self.entries[i]["key"])
            # cộng thêm nếu khớp sau bỏ muối
            if _strip_salts(self.entries[i]["key"]) == ks:
                s += 0.15
            if s >= thr:
                r = self.entries[i]["rxcui"]
                if s > best.get(r, 0):
                    best[r] = s
        ranked = sorted(best.items(), key=lambda x: -x[1])
        return [r for r, _ in ranked[:topk]]

    def explain(self, text, topk=6):
        k = norm_key(text)
        rows = []
        seen = set()
        if k in self.exact:
            for r in self.exact[k]:
                rows.append((r, self.by_rxcui.get(r, ""), 1.0))
                seen.add(r)
        for i in self.fx.candidates(k):
            s = self.fx.score(k, i, self.entries[i]["key"])
            r = self.entries[i]["rxcui"]
            if r in seen:
                continue
            rows.append((r, self.entries[i]["str"], s))
            seen.add(r)
        rows.sort(key=lambda x: -x[2])
        return rows[:topk]


if __name__ == "__main__":
    print(">> Loading KBs ...")
    icd = ICDLinker()
    rx = RxNormLinker()

    print("\n=== ICD tests ===")
    for t in ["bệnh trào ngược dạ dày - thực quản", "Tăng huyết áp",
              "đái tháo đường", "rung nhĩ", "viêm phổi", "Nhồi máu cơ tim"]:
        print(f"\n[{t}] -> {icd.link(t)}")
        for c, n, s in icd.explain(t, 3):
            print(f"    {c:8s} {s:.2f}  {n}")

    print("\n=== RxNorm tests ===")
    for t in ["Chlorpheniramine 0.4 MG/ML", "metoprolol", "aspirin 325mg",
              "nitroglycerin", "omeprazole"]:
        print(f"\n[{t}] -> {rx.link(t)}")
        for r, n, s in rx.explain(t, 4):
            print(f"    {r:9s} {s:.2f}  {n}")
