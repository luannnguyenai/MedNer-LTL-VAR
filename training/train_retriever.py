"""
train_retriever.py — PHASE 3: fine-tune BGE-m3 (retriever) bằng contrastive.

Loss: MultipleNegativesRankingLoss (InfoNCE) — dùng in-batch negatives; nếu
triplet có cột 'negative' (hard negative từ linker) thì càng mạnh.

Dữ liệu: JSONL {"anchor","positive"[,"negative"]} từ retriever_data.py.

Ví dụ:
  python -c "import retriever_data as R, json; \
     rows=R.from_gold('data/unified.jsonl', icd=__import__('linker').ICDLinker(), \
                      rx=__import__('linker').RxNormLinker()); \
     names=[e['vn'] for e in json.load(open('../kb/icd_index.json'))['entries'] if e['vn']]; \
     rows+=R.from_kb_synth(names, n=30000, hard_neg_names=names); \
     R.write(rows,'data/retr.jsonl')"
  python train_retriever.py --data data/retr.jsonl --model BAAI/bge-m3 \
     --out ckpt/bge-m3-medvn --epochs 1 --bs 64

Yêu cầu: sentence-transformers>=3.0, datasets, accelerate.
GPU khuyến nghị; batch lớn giúp MNRL (nhiều in-batch negatives).
"""
import argparse
from datasets import load_dataset
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval_data", default=None)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--out", default="ckpt/bge-m3-medvn")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_len

    # cột phải đúng thứ tự (anchor, positive[, negative]) cho MNRL
    ds = load_dataset("json", data_files=args.data, split="train")
    keep = [c for c in ["anchor", "positive", "negative"] if c in ds.column_names]
    ds = ds.select_columns(keep)
    eval_ds = None
    if args.eval_data:
        eval_ds = load_dataset("json", data_files=args.eval_data, split="train")
        eval_ds = eval_ds.select_columns(
            [c for c in keep if c in eval_ds.column_names])

    loss = MultipleNegativesRankingLoss(model)

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        fp16=True,
        batch_sampler="no_duplicates",   # quan trọng cho MNRL
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=ds,
        eval_dataset=eval_ds, loss=loss)
    trainer.train()
    model.save_pretrained(args.out)
    print(f"[train_retriever] done -> {args.out}")


if __name__ == "__main__":
    main()
