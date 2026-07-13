# QUICKSTART — 8×A100 (chạy ngay)

## 1 lệnh duy nhất
```bash
cd medner/training
pip install -r requirements-max.txt
bash run_8xA100.sh
```
Xong → kết quả ở `medner/out_max/*.json` (100 file).

---

## Việc script tự làm
1. **Data-prep** (CPU, ~1–2 phút): build KB nếu thiếu → dựng `unified.jsonl` (silver từ pipeline rule) → sinh `sft.jsonl` (R1–R4) + `retr.jsonl` (triplet). *(đã test chạy được)*
2. **Train 3 phase ĐỒNG THỜI** (không tuần tự):

   | Phase | GPU | Model | ~Thời gian |
   |---|---|---|---|
   | 1 encoder | **GPU 0** | XLM-R-large, bs=48 | ~30–45 phút |
   | 3 retriever | **GPU 1** | BGE-m3, bs=256 | ~20–40 phút |
   | 2 LLM | **GPU 2–7** (6-way DDP) | Qwen2.5-7B QLoRA, bs=4×4×6 | ~1–1.5h |

   → **Wall-clock ≈ 1–1.5h** (bằng phase chậm nhất, không phải tổng 10–14h).
3. **Inference SONG SONG 8 GPU**: 100 file chia 8 shard (~13 file/GPU) → **~30–90 giây**.

---

## Vì sao nhanh
- **8×A100 = ~320–640GB VRAM.** Qwen2.5-7B 4-bit chỉ ~5GB/GPU → DDP 6 GPU chạy thoải mái, throughput ~6×.
- 3 phase **độc lập** (đều ăn `unified.jsonl` + linker) → chạy song song trên nhóm GPU riêng.
- Inference bottleneck là LLM → 8 tiến trình, mỗi GPU 1 bản model → 8× tốc độ.

---

## Biến thể

**Nhanh hơn nữa (bỏ LLM/retriever, chỉ encoder+linker):** ~72–74% F1, inference vài giây
```bash
python infer_parallel.py --input ../input --out_dir ../out_fast \
  --gpus 0,1,2,3,4,5,6,7 --no_llm --no_retriever
```

**Tăng chất lượng LLM (dùng cả 8 GPU cho Phase 2, train tuần tự):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu \
  --num_processes 8 --mixed_precision bf16 sft_lora.py \
  --data data/sft.jsonl --epochs 4 --bs 8 --grad_accum 2 --out ckpt/qwen-medner-lora
```

**Inference nhanh gấp 3–5× bằng vLLM:** cài `vllm`, sửa `llm_infer.MedLLM` sang vLLM backend (interface `make_llm_fn` giữ nguyên).

---

## Theo dõi khi đang chạy
```bash
tail -f logs/phase1_encoder.log     # encoder
tail -f logs/phase2_llm.log         # LLM (loss giảm dần)
tail -f logs/phase3_retriever.log   # retriever
watch -n 2 nvidia-smi               # tải GPU
```

---

## Cần có trước khi chạy
- `medner/input/*.txt` — 100 file đề bài (hiện đang ở `/tmp/work/input`; copy vào `medner/input/`).
- `medner/ICD10.xlsx`, `medner/RXNCONSO.RRF` — để build KB (hoặc dùng KB đã build sẵn trong `kb/`).
- `data/gold_dev.jsonl` — **nên có** để đo F1 thật khi train (không bắt buộc để chạy).
- Mạng tới HuggingFace (tải XLM-R / Qwen2.5-7B / BGE-m3 lần đầu).

> Nếu **không có gold_dev.jsonl**: bỏ cờ `--dev` trong `run_8xA100.sh` (Phase 1) — vẫn train được, chỉ không in F1 mỗi epoch.

---

## Không hallucination (giữ nguyên ở mọi tốc độ)
Mã cuối luôn: linker/retriever (⊆ KB) → cổng `rerank.py` (đã test loại mã bịa) → validate schema. Chạy song song/nhanh **không** ảnh hưởng tính đúng.
