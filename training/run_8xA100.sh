#!/usr/bin/env bash
# run_8xA100.sh — Chạy TRỌN bản MAX trên 8×A100, tối ưu wall-clock.
#
# Chiến lược: 3 phase train ĐỘC LẬP -> chạy ĐỒNG THỜI trên các nhóm GPU khác nhau,
# nên wall-clock ≈ phase chậm nhất (Phase 2 LLM ~1–1.5h) thay vì cộng dồn 10–14h.
#
#   GPU 0      -> Phase 1: encoder XLM-R-large (single-GPU, batch lớn)
#   GPU 1      -> Phase 3: retriever BGE-m3     (single-GPU, batch lớn)
#   GPU 2..7   -> Phase 2: Qwen2.5-7B QLoRA     (6-way DDP qua accelerate)
#
# Sau khi cả 3 xong -> inference SONG SONG trên cả 8 GPU (100 file ~ vài chục giây).
#
# Chạy:  bash run_8xA100.sh
set -e
cd "$(dirname "$0")"                       # vào thư mục training/
mkdir -p logs ckpt data
export TOKENIZERS_PARALLELISM=false

echo "==================== 0. CHUẨN BỊ DỮ LIỆU ===================="
# KB (bỏ qua nếu đã build)
[ -f ../kb/icd_index.json ]     || python ../kb/build_icd.py            ../ICD10.xlsx
[ -f ../kb/rxnorm_index.json ]  || python ../kb/build_rxnorm.py         ../RXNCONSO.RRF
[ -f ../kb/drug_gazetteer.json ]|| python ../kb/build_drug_gazetteer.py ../RXNCONSO.RRF

# data/unified.jsonl phải có sẵn (silver+ViMedNER+pseudo_fine+gold+aug).
# Nếu chưa có, dựng silver tối thiểu từ pipeline rule:
if [ ! -f data/unified.jsonl ]; then
  echo "[data] chưa có unified.jsonl — dựng silver từ pipeline rule..."
  python - <<'PY'
import sys, json, glob, os
sys.path.insert(0, "../src")
from pipeline import Pipeline           # pipeline rule đã có
p = Pipeline()
rows = []
for fp in sorted(glob.glob("../input/*.txt")) or sorted(glob.glob("../input/*")):
    text = open(fp, encoding="utf-8").read()
    items = p.process(text)
    spans = [{"text": it["text"], "fine": it["type"], "coarse": None,
              "start": it["position"][0], "end": it["position"][1],
              "assertions": it.get("assertions", []),
              "candidates": it.get("candidates", [])} for it in items]
    rows.append({"text": text, "source": "silver", "spans": spans})
os.makedirs("data", exist_ok=True)
open("data/unified.jsonl","w",encoding="utf-8").write(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
print(f"[data] silver -> data/unified.jsonl ({len(rows)} câu)")
PY
fi

# data SFT cho LLM (R1–R4) + triplet retriever
python - <<'PY'
import sys, json
sys.path.insert(0, "../src")
from linker import ICDLinker, RxNormLinker
icd, rx = ICDLinker(), RxNormLinker()

# SFT data (llm_data)
from llm_data import build
build("data/unified.jsonl", "data/sft.jsonl",
      roles=("ner","assertion","rerank"), icd=icd, rx=rx)

# retriever triplets (retriever_data)
import retriever_data as R
rows = R.from_gold("data/unified.jsonl", icd=icd, rx=rx)
names = [e["vn"] for e in json.load(open("../kb/icd_index.json"))["entries"] if e["vn"]]
rows += R.from_kb_synth(names, n=30000, hard_neg_names=names)
R.write(rows, "data/retr.jsonl")
PY

echo "==================== TRAIN 3 PHASE ĐỒNG THỜI ===================="

# ---- Phase 1: encoder trên GPU 0 (batch lớn tận dụng A100 80GB) ----
CUDA_VISIBLE_DEVICES=0 python train_ner.py \
  --encoder xlm-roberta-large \
  --train data/unified.jsonl --dev data/gold_dev.jsonl \
  --epochs 8 --bs 48 --lr 2e-5 --out ckpt/nerA-xlmr \
  > logs/phase1_encoder.log 2>&1 &
P1=$!; echo "  [Phase1] encoder -> GPU0 (pid $P1)"

# ---- Phase 3: retriever trên GPU 1 (batch lớn cho MNRL) ----
CUDA_VISIBLE_DEVICES=1 python train_retriever.py \
  --data data/retr.jsonl --model BAAI/bge-m3 \
  --out ckpt/bge-m3-medvn --epochs 1 --bs 256 \
  > logs/phase3_retriever.log 2>&1 &
P3=$!; echo "  [Phase3] retriever -> GPU1 (pid $P3)"

# ---- Phase 2: LLM QLoRA trên GPU 2..7 (6-way DDP) ----
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 6 --mixed_precision bf16 \
  sft_lora.py \
  --data data/sft.jsonl --model Qwen/Qwen2.5-7B-Instruct \
  --out ckpt/qwen-medner-lora --epochs 3 --bs 4 --grad_accum 4 --max_len 2048 \
  > logs/phase2_llm.log 2>&1 &
P2=$!; echo "  [Phase2] LLM QLoRA -> GPU2-7 DDP (pid $P2)"

echo "  ...đang train (theo dõi: tail -f logs/phase*.log)"
wait $P1 && echo "  [Phase1] ✓ done"
wait $P3 && echo "  [Phase3] ✓ done"
wait $P2 && echo "  [Phase2] ✓ done"
echo "==================== TRAIN XONG — INFERENCE SONG SONG ===================="

# ---- Inference 100 file song song 8 GPU ----
python infer_parallel.py \
  --input ../input --out_dir ../out_max \
  --gpus 0,1,2,3,4,5,6,7 \
  --ner_ckpt ckpt/nerA-xlmr --encoder xlm-roberta-large \
  --llm_adapter ckpt/qwen-medner-lora --retriever ckpt/bge-m3-medvn

echo "==================== HOÀN TẤT -> ../out_max/*.json ===================="
