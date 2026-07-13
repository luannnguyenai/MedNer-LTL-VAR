"""
retriever_infer.py — Dùng BGE-m3 (đã fine-tune) để BỔ SUNG ứng viên ngữ nghĩa
cho linking, TRƯỚC bước rerank. Bắt các mention biểu đạt lệch KB mà fuzzy trigram
bỏ sót.

Quy trình:
  1) Precompute embedding cho toàn bộ tên KB (ICD tiếng Việt/EN, RxNorm STR) — 1 lần.
  2) Khi có mention: embed -> tìm top-k tên gần nhất (cosine) -> trả CODE tương ứng.
  3) Hợp (union) với ứng viên từ linker deterministic -> tập ứng viên cho rerank.

Mã trả ra vẫn ⊆ KB (chỉ là tra cứu ngữ nghĩa), không sinh -> không hallucination.

Yêu cầu: sentence-transformers, numpy. (FAISS tùy chọn để nhanh hơn.)
"""
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "kb")


class SemanticIndex:
    def __init__(self, model_path="BAAI/bge-m3", cache_dir="cache_emb"):
        self.model = SentenceTransformer(model_path)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.names = []       # list str
        self.codes = []       # code tương ứng name
        self.emb = None       # (N, d) normalized

    def build_icd(self):
        kb = json.load(open(os.path.join(KB_DIR, "icd_index.json"), encoding="utf-8"))
        for e in kb["entries"]:
            if e["vn"]:
                self.names.append(e["vn"]); self.codes.append(e["code"])
            if e["en"]:
                self.names.append(e["en"]); self.codes.append(e["code"])
        self._encode("icd")

    def build_rxnorm(self):
        kb = json.load(open(os.path.join(KB_DIR, "rxnorm_index.json"), encoding="utf-8"))
        for e in kb["entries"]:
            self.names.append(e["str"]); self.codes.append(e["rxcui"])
        self._encode("rxnorm")

    def _encode(self, tag):
        cache = os.path.join(self.cache_dir, f"{tag}.npy")
        if os.path.exists(cache):
            self.emb = np.load(cache)
            return
        self.emb = self.model.encode(
            self.names, batch_size=256, normalize_embeddings=True,
            show_progress_bar=True).astype("float32")
        np.save(cache, self.emb)

    def search(self, mention, topk=5):
        q = self.model.encode([mention], normalize_embeddings=True).astype("float32")
        sims = self.emb @ q[0]                       # cosine (đã normalize)
        idx = np.argpartition(-sims, min(topk * 3, len(sims) - 1))[:topk * 3]
        idx = idx[np.argsort(-sims[idx])]
        out, seen = [], set()
        for i in idx:
            c = self.codes[i]
            if c in seen:
                continue
            seen.add(c)
            out.append((c, float(sims[i]), self.names[i]))
            if len(out) >= topk:
                break
        return out


def augment_candidates(linker_codes, semantic_hits, thr=0.55, cap=5):
    """Hợp ứng viên linker + ngữ nghĩa (lọc ngưỡng), giữ thứ tự ưu tiên linker."""
    out = list(linker_codes)
    for c, score, _ in semantic_hits:
        if score >= thr and c not in out:
            out.append(c)
    return out[:cap]


if __name__ == "__main__":
    print(__doc__)
    print(">> Dùng thật:")
    print("""
    from retriever_infer import SemanticIndex, augment_candidates
    from linker import ICDLinker
    idx = SemanticIndex("ckpt/bge-m3-medvn"); idx.build_icd()   # cache 1 lần
    icd = ICDLinker()
    mention = "đau dạ dày trào ngược"
    lk = icd.link(mention, topk=3)                    # ứng viên fuzzy
    sem = idx.search(mention, topk=5)                 # ứng viên ngữ nghĩa
    cands = augment_candidates(lk, sem)               # hợp -> đưa vào rerank.py
    """)
