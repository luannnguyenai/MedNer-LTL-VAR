"""
dataset.py — Nạp JSONL hợp nhất -> tensor, CĂN NHÃN theo offset_mapping.

Điểm mấu chốt (giữ đúng CHAR position của đề): dùng fast-tokenizer với
return_offsets_mapping=True. Mỗi subword có (char_start,char_end) trên RAW text;
ta gán BIO fine/coarse + assertion theo span char. Subword không phải token đầu
của thực thể -> I-; special/padding -> -100 (bỏ khỏi loss & CRF).

align_labels() là hàm THUẦN, test được bằng offset_mapping giả (không cần tải
tokenizer).
"""
import json
import torch
from torch.utils.data import Dataset

from schema import FINE2ID, COARSE2ID, ASSERTIONS

ASSERT2ID = {a: i for i, a in enumerate(ASSERTIONS)}
ASSERTABLE_FINE = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}


def _span_bio(offsets, spans, key, label2id):
    """Trả list nhãn-id BIO cho từng subword theo offset & span[key]."""
    n = len(offsets)
    tags = [label2id["O"]] * n
    for sp in spans:
        lab = sp.get(key)
        if not lab:
            continue
        s, e = sp["start"], sp["end"]
        first = True
        for i, (cs, ce) in enumerate(offsets):
            if cs == ce:                     # special token (0,0)
                continue
            # subword giao với span (dùng tâm để tránh biên lệch)
            if cs >= s and ce <= e or (cs < e and ce > s and (ce - cs) > 0
                                       and max(cs, s) < min(ce, e)):
                if first:
                    tags[i] = label2id[f"B-{lab}"]
                    first = False
                else:
                    tags[i] = label2id[f"I-{lab}"]
    return tags


def align_labels(offsets, spans, special_mask=None):
    """Trả dict: fine_tags, coarse_tags, assert_labels(n,3), assert_mask(n)."""
    n = len(offsets)
    fine = _span_bio(offsets, spans, "fine", FINE2ID)
    coarse = _span_bio(offsets, spans, "coarse", COARSE2ID)
    a_lab = [[0, 0, 0] for _ in range(n)]
    a_mask = [0] * n
    for sp in spans:
        if sp.get("fine") not in ASSERTABLE_FINE:
            continue
        s, e = sp["start"], sp["end"]
        vec = [1 if a in (sp.get("assertions") or []) else 0 for a in ASSERTIONS]
        for i, (cs, ce) in enumerate(offsets):
            if cs == ce:
                continue
            if max(cs, s) < min(ce, e):
                a_mask[i] = 1
                for j in range(3):
                    a_lab[i][j] = max(a_lab[i][j], vec[j])
    # special tokens -> -100 cho fine/coarse
    for i, (cs, ce) in enumerate(offsets):
        if cs == ce or (special_mask and special_mask[i]):
            fine[i] = -100
            coarse[i] = -100
            a_mask[i] = 0
    return {"fine": fine, "coarse": coarse, "assert": a_lab, "assert_mask": a_mask}


class NERDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")
                     if l.strip()]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        enc = self.tok(ex["text"], truncation=True, max_length=self.max_len,
                       return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        lab = align_labels(offsets, ex["spans"])
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "fine": lab["fine"], "coarse": lab["coarse"],
            "assert": lab["assert"], "assert_mask": lab["assert_mask"],
        }


def collate(batch, pad_id=1):
    """Pad động. pad_id mặc định 1 (XLM-R <pad>)."""
    maxlen = max(len(b["input_ids"]) for b in batch)

    def pad(seq, val):
        return seq + [val] * (maxlen - len(seq))

    ids = torch.tensor([pad(b["input_ids"], pad_id) for b in batch])
    am = torch.tensor([pad(b["attention_mask"], 0) for b in batch])
    fine = torch.tensor([pad(b["fine"], -100) for b in batch])
    coarse = torch.tensor([pad(b["coarse"], -100) for b in batch])
    amask = torch.tensor([pad(b["assert_mask"], 0) for b in batch])
    a3 = torch.tensor([b["assert"] + [[0, 0, 0]] * (maxlen - len(b["assert"]))
                       for b in batch])
    return {"input_ids": ids, "attention_mask": am, "fine_tags": fine,
            "coarse_tags": coarse, "assert_labels": a3, "assert_mask": amask}


if __name__ == "__main__":
    # TEST align_labels bằng offset_mapping GIẢ (không cần tokenizer thật)
    text = "ho đờm xanh, dùng aspirin"
    #        0123456789...
    spans = [
        {"start": 0, "end": 11, "text": "ho đờm xanh",
         "fine": "TRIỆU_CHỨNG", "coarse": "PROBLEM", "assertions": ["isNegated"]},
        {"start": 18, "end": 25, "text": "aspirin",
         "fine": "THUỐC", "coarse": "TREATMENT", "assertions": []},
    ]
    # giả lập subword: [CLS] ho | đờm | xanh | , | dùng | asp | irin [SEP]
    offsets = [(0, 0), (0, 2), (3, 6), (7, 11), (11, 12), (13, 17),
               (18, 21), (21, 25), (0, 0)]
    lab = align_labels(offsets, spans)
    from schema import BIO_FINE
    print("subword     offset      fine")
    toks = ["[CLS]", "ho", "đờm", "xanh", ",", "dùng", "asp", "irin", "[SEP]"]
    for tk, off, f in zip(toks, offsets, lab["fine"]):
        fl = BIO_FINE[f] if f != -100 else "-100"
        print(f"  {tk:7s} {str(off):10s} {fl}")
    print("assert_mask:", lab["assert_mask"])
    print("assert(isNeg,isFam,isHist) trên 'ho':", lab["assert"][1])
    assert lab["fine"][1] == FINE2ID["B-TRIỆU_CHỨNG"]
    assert lab["fine"][2] == FINE2ID["I-TRIỆU_CHỨNG"]
    assert lab["fine"][6] == FINE2ID["B-THUỐC"]
    assert lab["fine"][7] == FINE2ID["I-THUỐC"]
    assert lab["assert"][1] == [1, 0, 0]     # isNegated trên token của 'ho'
    print("ALIGN OK")
