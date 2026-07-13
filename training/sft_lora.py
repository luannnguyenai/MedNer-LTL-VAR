"""
sft_lora.py — PHASE 2: QLoRA SFT cho Qwen2.5-7B (reasoner đa vai trò R1–R4).

Chạy trên GPU (khuyến nghị >=24GB cho 7B QLoRA 4-bit; 16GB vẫn được với
batch nhỏ + grad-accum + gradient_checkpointing).

Dữ liệu: chat-format JSONL {"messages":[...]} do training/llm_data.py sinh
(gộp mọi role R1–R4). `assistant_only_loss=True` -> chỉ tính loss trên phần
assistant (không học lại prompt).

Ví dụ:
  # 1) dựng data
  python -c "from llm_data import build; build('data/unified.jsonl','data/sft.jsonl', \
             roles=('ner','assertion','rerank'), icd=__import__('linker').ICDLinker(), \
             rx=__import__('linker').RxNormLinker())"
  # 2) train
  python sft_lora.py --data data/sft.jsonl --model Qwen/Qwen2.5-7B-Instruct \
         --out ckpt/qwen-medner-lora --epochs 3 --bs 2 --grad_accum 8

Yêu cầu: transformers, trl>=0.12, peft, bitsandbytes, accelerate, datasets.
"""
import argparse
import os
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# target modules cho kiến trúc Qwen2/2.5 (attention + MLP)
QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# ---- xử lý đa GPU (DDP) ----
# Khi chạy dưới accelerate/torchrun (WORLD_SIZE>1), MỖI tiến trình phải nạp TRỌN
# model lên GPU riêng của nó -> device_map={"":local_rank}. device_map="auto"
# (model-parallel) sẽ XUNG ĐỘT với DDP. Qwen2.5-7B 4-bit ~5GB nên mỗi A100 thừa sức.
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE_MAP = {"": LOCAL_RANK} if WORLD_SIZE > 1 else "auto"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL chat-format từ llm_data.py")
    ap.add_argument("--eval_data", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="ckpt/qwen-medner-lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)          # LoRA ~10x base
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--no_4bit", action="store_true", help="tắt QLoRA (LoRA fp16)")
    ap.add_argument("--packing", action="store_true", help="bật packing (tắt khi cần mask assistant chuẩn)")
    args = ap.parse_args()

    # ---- model (QLoRA 4-bit) ----
    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map=DEVICE_MAP,
        torch_dtype=torch.bfloat16, attn_implementation="eager")
    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=QWEN_TARGETS)

    # ---- data ----
    ds = load_dataset("json", data_files=args.data, split="train")
    eval_ds = (load_dataset("json", data_files=args.eval_data, split="train")
               if args.eval_data else None)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_len,
        packing=args.packing,
        assistant_only_loss=not args.packing,   # chỉ học phần assistant
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,     # DDP: LoRA không có param thừa
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds, eval_dataset=eval_ds,
        peft_config=lora, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)          # lưu LoRA adapter
    tok.save_pretrained(args.out)
    print(f"[sft_lora] done -> {args.out}")


if __name__ == "__main__":
    main()
