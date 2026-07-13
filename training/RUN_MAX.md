# RUN_MAX.md — Chạy bản MAX end-to-end (khi có GPU)

> Nối trọn 3 phase + inference. Xem `ARCHITECTURE_MAX.md` cho thiết kế & lý do.
> Ngân sách runtime: XLM-R-large 0.56B + Qwen2.5-7B(LoRA) 7.61B + BGE-m3 0.57B = **8.74B < 9B**.

```bash
cd training
pip install -r requirements-max.txt
```

## 0. Chuẩn bị KB + dữ liệu hợp nhất
```bash
# KB (nếu chưa build)
python ../kb/build_icd.py            ../ICD10.xlsx
python ../kb/build_rxnorm.py         ../RXNCONSO.RRF
python ../kb/build_drug_gazetteer.py ../RXNCONSO.RRF

# Dữ liệu 5-type hợp nhất (silver + ViMedNER + pseudo_fine + gold + aug) -> data/unified.jsonl
#   (dùng adapters.py + augment.py + silver từ src/run_all.py; xem README_TRAIN.md §0,§2)
```

## 1. PHASE 1 — Encoder NER (nền tảng)
```bash
python train_ner.py --encoder xlm-roberta-large \
  --train data/unified.jsonl --dev data/gold_dev.jsonl \
  --epochs 8 --bs 8 --lr 2e-5 --out ckpt/nerA-xlmr
```

## 2. PHASE 3 — Retriever (chạy song song P1 được)
```bash
# 3a. khai mỏ triplet (gold + synth từ KB), hard-negative từ linker
python -c "import retriever_data as R, json; \
import sys; sys.path.insert(0,'../src'); from linker import ICDLinker, RxNormLinker; \
rows=R.from_gold('data/unified.jsonl', icd=ICDLinker(), rx=RxNormLinker()); \
names=[e['vn'] for e in json.load(open('../kb/icd_index.json'))['entries'] if e['vn']]; \
rows+=R.from_kb_synth(names, n=30000, hard_neg_names=names); \
R.write(rows,'data/retr.jsonl')"

# 3b. fine-tune BGE-m3
python train_retriever.py --data data/retr.jsonl --model BAAI/bge-m3 \
  --out ckpt/bge-m3-medvn --epochs 1 --bs 64
```

## 3. PHASE 2 — LLM đa vai trò (Qwen2.5-7B QLoRA)
```bash
# 2a. dựng data SFT R1–R4 (dùng linker cho role rerank)
python -c "import sys; sys.path.insert(0,'../src'); \
from linker import ICDLinker, RxNormLinker; from llm_data import build; \
build('data/unified.jsonl','data/sft.jsonl', roles=('ner','assertion','rerank'), \
      icd=ICDLinker(), rx=RxNormLinker())"

# 2b. QLoRA SFT (TRL) — hoặc axolotl: accelerate launch -m axolotl.cli.train qwen_qlora_axolotl.yaml
python sft_lora.py --data data/sft.jsonl --model Qwen/Qwen2.5-7B-Instruct \
  --out ckpt/qwen-medner-lora --epochs 3 --bs 2 --grad_accum 8
```

## 4. INFERENCE end-to-end (ghép mọi thành phần)
```python
import sys; sys.path.insert(0, "../src")
from predict import NERPredictor              # Phase 1 encoder -> spans+type+assertion(nháp)
from llm_infer import MedLLM                  # Phase 2 LLM (R1 NER khó, R2 assertion, rerank)
from retriever_infer import SemanticIndex, augment_candidates   # Phase 3
from rerank import rerank
from linker import ICDLinker, RxNormLinker

ner = NERPredictor("ckpt/nerA-xlmr", "xlm-roberta-large")
llm = MedLLM("Qwen/Qwen2.5-7B-Instruct", adapter="ckpt/qwen-medner-lora")
llm_fn = llm.make_llm_fn()
sidx = SemanticIndex("ckpt/bge-m3-medvn"); sidx.build_icd(); sidx.build_rxnorm()  # cache 1 lần
icd, rx = ICDLinker(), RxNormLinker()

def run(text):
    items = ner.predict(text)                              # ② spans (+assertion nháp)
    # ③ second-opinion NER cho câu khó -> gộp (ensemble.ensemble_spans) [tùy chọn]
    for it in items:
        t, ty = it["text"], it["type"]
        # ④ assertion bằng LLM (override nếu cần)
        if ty in ("CHẨN_ĐOÁN","THUỐC","TRIỆU_CHỨNG"):
            it["assertions"] = llm.assertion(text, t, ty) or it["assertions"]
        # ⑥ ứng viên: linker ∪ retriever ngữ nghĩa
        if ty == "CHẨN_ĐOÁN":
            lk = icd.link(t, topk=3)
            cands = augment_candidates(lk, sidx.search(t, topk=5))
            pairs = [(c, icd._name(c)) for c in cands]
            it["candidates"] = rerank(t, ty, text, pairs, llm_fn)   # ⑦ rerank ràng buộc
        elif ty == "THUỐC":
            lk = rx.link(t, topk=3)
            cands = augment_candidates(lk, sidx.search(t, topk=5))
            pairs = [(c, rx.by_rxcui.get(c,"")) for c in cands]
            it["candidates"] = rerank(t, ty, text, pairs, llm_fn)
    return items   # ⑨ validate schema (src/run_all.py::validate)
```

## Ghi chú
- **Không hallucination:** mọi mã đi qua linker/retriever (⊆ KB) + cổng `rerank.py` (đã test loại mã bịa) + validate.
- **Tốc độ:** thay `transformers.generate` bằng **vLLM + LoRA** cho R1–R4 khi chấm 100 file.
- **A/B từng phase** trên gold dev để giữ phần đáng giá (Phase 1 là nền; Phase 2/3 cộng thêm ở đuôi khó/assertion/chọn mã).
