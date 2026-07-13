"""
selftest_train.py — Kiểm thử TÍCH HỢP vòng train mà KHÔNG tải model từ mạng.

Dùng tokenizer giả (whitespace + offset_mapping) + model nhỏ (BertConfig) để chạy:
  encode -> collate -> loss đa nhiệm -> backward -> eval span-F1.
Mục tiêu: chứng minh toàn bộ đường train/inference nối đúng.
"""
import json
import re
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertConfig

from model import MultiTaskNER
from dataset import NERDataset, collate, align_labels
from schema import BIO_FINE, BIO_COARSE, ASSERTIONS


class FakeTok:
    """Tokenizer whitespace tối giản, có offset_mapping thật trên raw text."""
    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=256,
                 return_offsets_mapping=False, return_tensors=None):
        ids = [2]                     # [CLS]
        offs = [(0, 0)]
        for m in re.finditer(r"\S+", text):
            ids.append((hash(m.group()) % 190) + 5)
            offs.append((m.start(), m.end()))
            if len(ids) >= max_length - 1:
                break
        ids.append(3)                 # [SEP]
        offs.append((0, 0))
        am = [1] * len(ids)
        out = {"input_ids": ids, "attention_mask": am}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        if return_tensors == "pt":
            out = {k: torch.tensor([v]) if k != "offset_mapping"
                   else torch.tensor([v]) for k, v in out.items()}
        return out

    def save_pretrained(self, p):
        pass


def make_data(path, n=8):
    base = [
        {"text": "ho đờm xanh và tức ngực",
         "spans": [{"start": 0, "end": 11, "text": "ho đờm xanh",
                    "fine": "TRIỆU_CHỨNG", "coarse": "PROBLEM", "assertions": []},
                   {"start": 15, "end": 23, "text": "tức ngực",
                    "fine": "TRIỆU_CHỨNG", "coarse": "PROBLEM", "assertions": []}]},
        {"text": "chẩn đoán viêm phổi dùng aspirin",
         "spans": [{"start": 10, "end": 19, "text": "viêm phổi",
                    "fine": "CHẨN_ĐOÁN", "coarse": "PROBLEM", "assertions": []},
                   {"start": 25, "end": 32, "text": "aspirin",
                    "fine": "THUỐC", "coarse": "TREATMENT", "assertions": []}]},
        {"text": "không sốt không ho",
         "spans": [{"start": 6, "end": 10, "text": "sốt",
                    "fine": "TRIỆU_CHỨNG", "coarse": "PROBLEM",
                    "assertions": ["isNegated"]},
                   {"start": 15, "end": 18, "text": "ho",
                    "fine": "TRIỆU_CHỨNG", "coarse": "PROBLEM",
                    "assertions": ["isNegated"]}]},
    ]
    rows = (base * ((n // len(base)) + 1))[:n]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    torch.manual_seed(0)
    make_data("/tmp/tr.jsonl", 12)
    make_data("/tmp/dev.jsonl", 3)
    tok = FakeTok()

    ds = NERDataset("/tmp/tr.jsonl", tok, 64)
    dl = DataLoader(ds, batch_size=4, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    cfg = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=64)
    model = MultiTaskNER("(t)", len(BIO_FINE), len(BIO_COARSE),
                         len(ASSERTIONS), config=cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("training tiny model 15 epochs (overfit sanity)...")
    for ep in range(15):
        model.train()
        tot = 0.0
        for batch in dl:
            out = model(**batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            tot += float(out["loss"])
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1}: loss={tot/len(dl):.3f}")

    # eval span-F1 trên train (overfit -> nên cao)
    from train_ner import evaluate
    p, r, f1 = evaluate(model, tok, "/tmp/tr.jsonl", "cpu", 64)
    print(f"train-set span P={p:.2f} R={r:.2f} F1={f1:.2f} (overfit check)")
    print("INTEGRATION OK — vòng train/encode/collate/loss/decode/eval chạy thông")


if __name__ == "__main__":
    main()
