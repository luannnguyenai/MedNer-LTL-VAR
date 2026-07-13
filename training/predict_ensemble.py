"""
predict_ensemble.py — Bộ dự đoán ENSEMBLE (pipeline tối ưu cho cuộc thi).

Chạy N model NER (vd XLM-R-large + ViHealthBERT), gộp span ở mức char bằng
ensemble.ensemble_spans, RỒI mới nối linker để cấp candidates (ICD/RxNorm).

Cam kết EXACT/không hallucination giữ nguyên: candidates do linker lấy từ KB.
Ensemble chỉ tăng độ chính xác NER + assertion, không "sinh" mã.

Dùng:
  from predict_ensemble import EnsembleNER
  ner = EnsembleNER([
      ("ckpt/nerA-xlmr",     "xlm-roberta-large",                 1.0),
      ("ckpt/nerA-vihealth", "demdecuong/vihealthbert-base-syllable", 0.8),
  ])
  items = ner.predict(open("input/1.txt").read())
"""
import os
import sys

from predict import NERPredictor, ASSERTABLE, CAND
from ensemble import ensemble_spans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from linker import ICDLinker, RxNormLinker  # noqa


class EnsembleNER:
    def __init__(self, members, min_conf=0.0, require_agree=False):
        """members: list (ckpt_dir, encoder_name, weight)."""
        self.models = []
        self.weights = []
        for ckpt, enc, w in members:
            self.models.append(NERPredictor(ckpt, enc))
            self.weights.append(w)
        self.min_conf = min_conf
        self.require_agree = require_agree
        # linker chỉ cần load 1 lần (dùng chung)
        self.icd = self.models[0].icd
        self.rx = self.models[0].rx

    def predict(self, text):
        span_lists = []
        for m in self.models:
            _, spans = m._spans(text)
            span_lists.append([{
                "start": sp["s"], "end": sp["e"], "type": sp["t"],
                "score": sp["score"],
                "assertions": sp["assert"] if sp["t"] in ASSERTABLE else [],
                "candidates": [],
            } for sp in spans])

        merged = ensemble_spans(span_lists, self.weights,
                                min_conf=self.min_conf,
                                require_agree=self.require_agree)

        out = []
        for sp in merged:
            t = text[sp["start"]:sp["end"]]
            item = {"text": t, "position": [sp["start"], sp["end"]],
                    "type": sp["type"],
                    "assertions": sp["assertions"] if sp["type"] in ASSERTABLE else [],
                    "candidates": []}
            if sp["type"] == "CHẨN_ĐOÁN":
                item["candidates"] = self.icd.link(t, topk=3)
            elif sp["type"] == "THUỐC":
                item["candidates"] = self.rx.link(t, topk=3)
            out.append(item)
        return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("txt")
    ap.add_argument("--member", action="append", nargs=3,
                    metavar=("CKPT", "ENCODER", "WEIGHT"), required=True,
                    help="lặp lại cho mỗi model, vd --member ckpt/xlmr xlm-roberta-large 1.0")
    a = ap.parse_args()
    members = [(c, e, float(w)) for c, e, w in a.member]
    ner = EnsembleNER(members)
    print(json.dumps(ner.predict(open(a.txt, encoding="utf-8").read()),
                     ensure_ascii=False, indent=2))
