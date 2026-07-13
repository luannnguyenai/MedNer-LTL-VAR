## GPU Memory Requirements — Pipeline MAX

> Tính toán dựa trên model size + batch config + QLoRA 4-bit + gradient checkpointing.

---

## 1. PHASE 1 — Encoder NER (XLM-R-large fine-tune)

**Config mặc định (`train_ner.py`):**
- Model: XLM-R-large 0.56B (fp16)
- Batch size: 8 per-device (gradient_accum 1)
- Tokenizer: XLM-R (subword)
- Max length: 256 tokens
- Loss: CRF + multi-task

| Setup | GPU vRAM | Lỏng lẻo | Ghi chú |
|---|---:|---|---|
| **RTX 4090 / A100** (48GB) | 16–18GB | ✅ rộng | đạo tạo 8 epoch thoải mái |
| **A40 / RTX 3090** (24GB) | 14–16GB | ✅ vừa | batch_size=8 OK; nếu giảm gradient_accum=2 thì ~10GB |
| **RTX 4080 / 3080 Ti** (12GB) | ~12GB | ⚠️ sít | batch_size=4 + gradient_accum=2 → ~8–10GB |
| **RTX 3070 Ti** (8GB) | ~7–8GB | ⚠️ tight | batch_size=1, gradient_accum=8, gradient_checkpointing=True → ~7–8GB (chậm) |

**Cách giảm từ 16GB xuống 8GB (nếu cần):**
```bash
python train_ner.py --bs 1 --grad_accum 16 --model xlm-roberta-large ...
# → ~7GB nhưng train chậm 16x (32 steps thay 2 per epoch)
```

---

## 2. PHASE 2 — LLM QLoRA (Qwen2.5-7B-Instruct)

**Config mặc định (`sft_lora.py`):**
- Model: Qwen2.5-7B-Instruct (fp16 + **4-bit QLoRA**)
- LoRA: r=32, alpha=32
- Batch size: 2 per-device (gradient_accum 8)
- Max length: 2048 tokens (chat format)
- Liger kernel OFF / gradient checkpointing ON

| Setup | GPU vRAM | Lỏng lẻo | Ghi chú |
|---|---:|---|---|
| **RTX A100 (80GB) / H100** | 24–32GB | ✅ thoải mái | chạy batch_size=8 được; epoch 3 trong ~1–2h |
| **A40 / RTX 4090 (48GB)** | 28–32GB | ✅ vừa | batch_size=2 + grad_accum=8; epoch 3 trong ~3–4h |
| **RTX 3090 (24GB)** | ~22–24GB | ⚠️ sít | batch_size=2 + grad_accum=8; cuối epoch nước đạo → ổn nếu max_len tắt padding |
| **RTX 4080 (12GB)** | ~10–12GB | ❌ khó | **cần**: batch_size=1, grad_accum=8 → ~10GB nhưng RẤT chậm (~8–12h/epoch) |
| **RTX 3070 Ti (8GB)** | không | ❌ không thể | QLoRA 4-bit 7B vẫn cần >8GB vì activation gradient |

**Tối ưu cho RTX 3090 24GB → ~20GB dùng thực tế:**
```bash
python sft_lora.py --bs 2 --grad_accum 8 --max_len 1024 \
  --model Qwen/Qwen2.5-7B-Instruct --data data/sft.jsonl ...
# Giảm max_len từ 2048 → 1024 giảm ~30% memory
```

---

## 3. PHASE 3 — Retriever (BGE-m3 contrastive)

**Config mặc định (`train_retriever.py`):**
- Model: BAAI/bge-m3 (0.57B fp16)
- Batch size: 64 (large batch tốt cho MNRL)
- Loss: MultipleNegativesRankingLoss (in-batch + hard negatives)
- Max length: 128 tokens (embedding KB names)

| Setup | GPU vRAM | Lỏng lẻo | Ghi chú |
|---|---:|---|---|
| **RTX A100 / 4090** | 12–16GB | ✅ thoải mái | batch_size=64 tối ưu MNRL; epoch 1 trong ~30–60 phút |
| **RTX 3090 (24GB)** | ~10–12GB | ✅ vừa | batch_size=64 được; ~1–2h/epoch |
| **RTX 4080 (12GB)** | ~8–10GB | ⚠️ sít | batch_size=32; ~2–3h/epoch |
| **RTX 3070 Ti (8GB)** | ~7–8GB | ⚠️ tight | batch_size=16 → ~7GB; ~4–6h/epoch |

**Tối ưu batch lớn (MNRL cần in-batch negatives):**
```bash
python train_retriever.py --bs 64 --model BAAI/bge-m3 --data data/retr.jsonl ...
# batch_sampler="no_duplicates" → quan trọng cho MNRL, không ảnh hưởng memory nhiều
```

---

## 4. INFERENCE (chạy 100 file)

**Stack inference:**
- Encoder (XLM-R-large): 1GB
- LoRA adapter (ckpt/qwen-medner-lora): ~500MB shared
- LLM Qwen2.5-7B (tải 1 lần): ~15GB (fp16) hoặc ~8GB (8-bit)
- BGE-m3 retriever: shared context (bỏ sau inference)
- KB index (ICD/RxNorm): ~2–3GB RAM (không GPU VRAM)

| Setup | GPU vRAM | Batch 1 file | Thời gian 100 file | Ghi chú |
|---|---:|---|---|---|
| **RTX A100 (80GB)** | 20–24GB | 0.5–1s (LLM overhead) | ~50–100s nếu tuần tự | nhanh nhất, thoải mái |
| **RTX 4090 (24GB)** | 18–20GB | 1–1.5s | ~2–2.5 phút tuần tự | ổn; dùng vLLM để nhanh hơn |
| **RTX 3090 (24GB)** | 16–18GB | 1–2s | ~2–3 phút | OK nhưng slow |
| **RTX 3080 Ti (12GB)** | ~12GB | 2–4s (swap) | ~4–6 phút | vLLM + LoRA tức thì → CPU swap chậm |
| **RTX 3070 Ti (8GB)** | >8GB needed | không | ❌ OOM | không thể chạy Qwen full (8-bit vẫn 9–10GB) |

**Nhanh hóa inference bằng vLLM:**
```bash
# thay vì transformers.generate chậm
pip install vllm
python -c "
from vllm import LLM
llm = LLM('Qwen/Qwen2.5-7B-Instruct', enable_lora=True)
# 100 file chạy trong ~30–60s nếu batch
"
```

---

## Khuyến nghị GPU cho cuộc thi

| Mục tiêu | GPU | Lý do |
|---|---|---|
| **Chạy quick test (inference trên gold dev)** | RTX 4080 (12GB) hoặc A40 | Chỉ inference; không cần train |
| **Train Phase 1 + Phase 3 (không LLM)** | RTX 3090 (24GB) hoặc A40 | Đủ; ~2–3 ngày train song song |
| **Train toàn bộ 3 phase (MAX)** | **RTX A100 (40GB+) hoặc 2x RTX 4090** | Phase 2 (LLM) là nút cổ chai; 24GB sít khi 3 phases cùng lúc |
| **Deploy inference 100 file** | RTX 4080+ (12GB) hoặc RTX 3090 (24GB) + vLLM | vLLM nhanh gấp 3–5x; inference đủ nhanh |

**Lộ trình thực tế (được chấp nhận):**
1. **Máy 24GB (RTX 3090 / 4090)**: train Phase 1 (2–3h) + Phase 3 (2–3h) = 4–6h
2. Chuyển GPU hoặc xin cloud (8–12GB/24h): Phase 2 LLM (4–8h tùy instance)
3. Inference (8–12GB): chạy trên máy lúc kiểm tra (hoặc cloud cheap instance)

---

## Nếu chỉ có 8GB (RTX 3070 Ti)

**Giải pháp:**
- ❌ **KHÔNG thể train Phase 2** (LLM 7B) — nếu cứ train, thêm 16GB SWAP → chậm lắm (~24h+)
- ✅ **Train Phase 1** (XLM-R) với batch nhỏ → sinh silver dữ liệu
- ✅ **Dùng Qwen2.5-7B pre-trained** không fine-tune (hoặc dùng lora người khác từ Hub) → inference chậm ~2–4s/file
- ⚠️ Ngân sách thua so MAX

Cách chiến: train Phase 1 + Phase 3 tốt, Phase 2 bỏ qua hoặc dùng 7B base không adapt.

---

## Tóm tắt (1 cái nhìn)

```
| Phase | Model | Batch | Max Length | Memory | Duration |
|-------|-------|-------|------------|--------|----------|
| 1     | XLM-R-large | 8 | 256 | 14–16GB | 2–3h (8 epochs) |
| 2     | Qwen 7B QLoRA | 2 | 2048 | 20–24GB | 4–8h (3 epochs) |
| 3     | BGE-m3 | 64 | 128 | 8–12GB | 2–3h (1 epoch) |
| Infer | All 3 | 1 file | — | 18–20GB | 50–100s (100 files tuần tự) |
```

**Nếu RTX 3090 (24GB) duy nhất:** chạy Phase 1 + Phase 3 song song (không lúc cùng), rồi Phase 2 trên cloud. Hoặc Phase 2 dùng lora từ community/skip.

---

## Script memory profiling (nếu muốn debug)

```bash
# Trước train, chạy profile trên 1–2 batch
python -c "
import torch; from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', device_map='auto')
print(f'Model VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB')
# + batch sẽ thêm 5–10GB
"
```
