"""
train_ner.py — Vòng huấn luyện encoder NER đa nhiệm.

Hỗ trợ TRỘN NHIỀU NGUỒN (mỗi nguồn 1 file JSONL hợp nhất) + đánh giá span-F1
(HEAD A, 5-type) tự hiện thực (không cần seqeval).

Ví dụ (khuyến nghị — xem README_TRAIN.md để biết quy trình 3 giai đoạn):
  python train_ner.py \
     --encoder xlm-roberta-large \
     --train data/silver.jsonl data/vimedner.jsonl data/i2b2_vi.jsonl \
             data/phoner.jsonl data/aug.jsonl \
     --dev   data/gold_dev.jsonl \
     --epochs 8 --bs 8 --lr 2e-5 --out ckpt/nerA

Lưu ý encoder: XLM-R (subword, KHÔNG cần word-seg) -> offset ánh xạ char sạch,
hợp với yêu cầu vị trí ký tự. Nếu dùng PhoBERT/ViHealthBERT (cần VnCoreNLP
RDRSegmenter), phải giữ mapping token->char (xem README_TRAIN §2).
"""
import argparse
import json
import os
import random
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model import MultiTaskNER
from dataset import NERDataset, collate
from schema import BIO_FINE, BIO_COARSE, ASSERTIONS


def spans_from_bio(tag_ids, offsets):
    """Giải BIO(id)+offset -> set (start,end,type) theo char."""
    out = []
    cur = None
    for tid, (cs, ce) in zip(tag_ids, offsets):
        lab = BIO_FINE[tid] if 0 <= tid < len(BIO_FINE) else "O"
        if lab.startswith("B-"):
            if cur:
                out.append(cur)
            cur = [cs, ce, lab[2:]]
        elif lab.startswith("I-") and cur and lab[2:] == cur[2]:
            cur[1] = ce
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return {(s, e, t) for s, e, t in out}


@torch.no_grad()
def evaluate(model, tok, dev_path, device, max_len=256):
    rows = [json.loads(l) for l in open(dev_path, encoding="utf-8") if l.strip()]
    tp = fp = fn = 0
    model.eval()
    for ex in rows:
        enc = tok(ex["text"], truncation=True, max_length=max_len,
                  return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        pred = model(enc["input_ids"].to(device),
                     enc["attention_mask"].to(device))["pred_fine"][0].tolist()
        pset = spans_from_bio(pred, offs)
        gset = {(sp["start"], sp["end"], sp["fine"]) for sp in ex["spans"]
                if sp.get("fine")}
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="xlm-roberta-large")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lambda_b", type=float, default=0.3)
    ap.add_argument("--lambda_c", type=float, default=0.5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--out", default="ckpt/nerA")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.encoder)
    pad_id = tok.pad_token_id or 1

    dsets = [NERDataset(p, tok, args.max_len) for p in args.train]
    train_ds = ConcatDataset(dsets)
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, pad_id))

    model = MultiTaskNER(args.encoder, n_fine=len(BIO_FINE),
                         n_coarse=len(BIO_COARSE), n_assert=len(ASSERTIONS),
                         lambda_b=args.lambda_b, lambda_c=args.lambda_c).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)

    best = -1
    os.makedirs(args.out, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tot += float(loss)
        msg = f"epoch {ep+1}/{args.epochs}  loss={tot/len(dl):.4f}"
        if args.dev:
            p, r, f1 = evaluate(model, tok, args.dev, device, args.max_len)
            msg += f"  dev P={p:.3f} R={r:.3f} F1={f1:.3f}"
            if f1 > best:
                best = f1
                torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
                tok.save_pretrained(args.out)
                json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"))
                msg += "  [saved best]"
        print(msg)
    if not args.dev:
        torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
        tok.save_pretrained(args.out)
    print("done. best dev F1:", best)


if __name__ == "__main__":
    main()
