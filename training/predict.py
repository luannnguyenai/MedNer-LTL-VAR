"""
predict.py — Suy luận: raw text -> spans (char) + type + assertions, rồi NỐI
vào linker (ICD/RxNorm) để sinh candidates. Đây là bộ THAY THẾ extractor rule
trong pipeline production (giữ nguyên tầng linker + validate).

Chống hallucination: model chỉ ra span+type+assertion; candidates VẪN do linker
lấy từ KB. Không đổi cam kết EXACT.
"""
import os
import sys
import torch
from transformers import AutoTokenizer

from model import MultiTaskNER
from dataset import ASSERT2ID
from schema import BIO_FINE, BIO_COARSE, ASSERTIONS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from linker import ICDLinker, RxNormLinker  # noqa

ASSERTABLE = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}
CAND = {"CHẨN_ĐOÁN", "THUỐC"}


class NERPredictor:
    def __init__(self, ckpt_dir, encoder=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(ckpt_dir)
        enc = encoder or "xlm-roberta-large"
        self.model = MultiTaskNER(enc, len(BIO_FINE), len(BIO_COARSE),
                                  len(ASSERTIONS)).to(self.device)
        self.model.load_state_dict(torch.load(
            os.path.join(ckpt_dir, "model.pt"), map_location=self.device))
        self.model.eval()
        self.icd = ICDLinker()
        self.rx = RxNormLinker()

    @torch.no_grad()
    def _spans(self, text, max_len=256):
        enc = self.tok(text, truncation=True, max_length=max_len,
                       return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        out = self.model(enc["input_ids"].to(self.device),
                         enc["attention_mask"].to(self.device))
        fine = out["pred_fine"][0].tolist()
        a_prob = torch.sigmoid(out["logit_assert"][0]).tolist()

        spans, cur = [], None
        for idx, (tid, (cs, ce)) in enumerate(zip(fine, offs)):
            lab = BIO_FINE[tid] if 0 <= tid < len(BIO_FINE) else "O"
            if lab.startswith("B-"):
                if cur:
                    spans.append(cur)
                cur = {"s": cs, "e": ce, "t": lab[2:], "idx": [idx]}
            elif lab.startswith("I-") and cur and lab[2:] == cur["t"]:
                cur["e"] = ce
                cur["idx"].append(idx)
            else:
                if cur:
                    spans.append(cur)
                cur = None
        if cur:
            spans.append(cur)

        # assertion = trung bình prob trên token thực thể, ngưỡng 0.5
        for sp in spans:
            if sp["t"] in ASSERTABLE:
                probs = [a_prob[i] for i in sp["idx"]]
                avg = [sum(p[j] for p in probs) / len(probs) for j in range(3)]
                sp["assert"] = [ASSERTIONS[j] for j in range(3) if avg[j] >= 0.5]
            else:
                sp["assert"] = []
        return text, spans

    def predict(self, text):
        text, spans = self._spans(text)
        out = []
        for sp in spans:
            t = text[sp["s"]:sp["e"]]
            item = {"text": t, "position": [sp["s"], sp["e"]], "type": sp["t"],
                    "assertions": sp["assert"] if sp["t"] in ASSERTABLE else [],
                    "candidates": []}
            if sp["t"] == "CHẨN_ĐOÁN":
                item["candidates"] = self.icd.link(t, topk=3)
            elif sp["t"] == "THUỐC":
                item["candidates"] = self.rx.link(t, topk=3)
            out.append(item)
        return out


if __name__ == "__main__":
    import json
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("txt")
    ap.add_argument("--encoder", default="xlm-roberta-large")
    a = ap.parse_args()
    pred = NERPredictor(a.ckpt, a.encoder)
    txt = open(a.txt, encoding="utf-8").read()
    print(json.dumps(pred.predict(txt), ensure_ascii=False, indent=2))
