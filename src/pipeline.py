"""
pipeline.py — Ghép NER + assertions + linking -> JSON đúng schema đề bài.

Schema mỗi khái niệm:
  {text, position:[start,end], type, assertions:[...], candidates:[...]}

Ràng buộc đề bài được ÁP DẶT ở tầng này (đảm bảo hợp lệ tuyệt đối):
  - candidates chỉ cho CHẨN_ĐOÁN (ICD) & THUỐC (RxNorm)
  - assertions chỉ cho CHẨN_ĐOÁN / THUỐC / TRIỆU_CHỨNG, tối đa 3
  - position theo ký tự, 0-indexed
"""
import json
from extract import Extractor
from assertions import assertions_for

CAND_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}
ASSERT_TYPES = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}


class Pipeline:
    def __init__(self):
        self.ex = Extractor()

    def process(self, text):
        spans, secs = self.ex.extract(text)
        out = []
        for s, e, t, ty, cands in spans:
            sec = self.ex.section_of(s, secs)
            item = {"text": t, "position": [s, e], "type": ty}
            if ty in ASSERT_TYPES:
                item["assertions"] = assertions_for(text, s, e, sec, ty)
            else:
                item["assertions"] = []
            if ty in CAND_TYPES:
                item["candidates"] = cands or []
            else:
                item["candidates"] = []
            # kiểm tra position khớp text gốc (an toàn)
            if text[s:e] != t:
                item["text"] = text[s:e]
            out.append(item)
        return out


if __name__ == "__main__":
    import sys
    pipe = Pipeline()
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/work/input/1.txt"
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    res = pipe.process(txt)
    print(json.dumps(res, ensure_ascii=False, indent=2))
