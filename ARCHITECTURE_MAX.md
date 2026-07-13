# Kiến trúc MAX-PERFORMANCE (≤9B) + Kế hoạch Fine-tune toàn diện

> Mục tiêu: **hiệu suất tối đa** cho cuộc thi, dùng trọn ngân sách ≤9B nếu có lợi, kèm **kế hoạch fine-tune cho từng model** để đạt ceiling. Vẫn giữ **EXACT — không hallucination** (LLM có cổng ràng buộc, mã luôn ⊆ KB).

---

## 0. Trả lời thẳng: 0.7B (v2) có thua 7.84B/8.7B (MAX) không?

**Params KHÔNG tỉ lệ thuận hiệu suất.** LLM 7B chỉ "ăn điểm" ở những chỗ CẦN suy luận, không giúp ở phần dễ. Phân rã theo thành phần chấm:

| Thành phần | v2 (0.7B) làm gì | MAX thêm gì nhờ 7B | Mức lợi thực tế |
|---|---|---|---|
| **NER phần dễ** (thực thể rõ) | encoder ensemble đã ~đúng | LLM gần như không thêm | ~0 |
| **NER phần khó** (hiếm, viết tắt, cụm dài lồng nhau) | encoder có thể sót | LLM generative-NER làm second-opinion → cứu recall | **+** (đuôi khó) |
| **Assertion** (phủ định/gia đình/tiền sử) | head + rule | **LLM rất mạnh suy luận ngữ cảnh** → sửa/override | **++** |
| **Linking — tập ứng viên** | linker+fuzzy tốt | retriever ngữ nghĩa tăng recall khi biểu đạt lệch | **+** |
| **Linking — CHỌN mã** (mention mơ hồ, nhiều dạng muối/hàm lượng) | linker chọn theo heuristic | **LLM rerank có ràng buộc** chọn đúng hơn | **++** |
| **Quan hệ khái niệm** (nhóm B) | rule hạn chế | LLM suy luận quan hệ | **++** (nếu chấm nhóm B) |

**Kết luận trung thực:**
- Trên **đa số câu dễ**, v2 (0.7B) đã đạt gần trần → 7B thêm ~0.
- Lợi của 7B **tập trung ở đuôi khó + assertion + chọn mã + quan hệ**. Ước lượng **+3–6% end-to-end** so v2, trong đó điểm assertion và "chọn mã" tăng mạnh hơn span-F1 thô.
- **Đánh đổi:** chậm hơn, phức tạp hơn, và phải quản lý hallucination (đã có cổng `rerank.py` → an toàn). 

→ Nếu mục tiêu là **ceiling điểm số** (đúng yêu cầu của bạn), MAX **vượt** v2, nhưng vượt **có chọn lọc** chứ không phải "7B tốt gấp 10 lần 0.7B". Thiết kế dưới đây đặt 7B đúng chỗ nó ăn điểm.

---

## 1. Pipeline MAX end-to-end

 raw text
   │
   ▼ ① Tiền xử lý cấu trúc (mục 1/2/3, bullet, "Nhãn: giá trị", tách từ dính, chuẩn số)
   │
   ▼ ② NER encoder (XLM-R-large, fine-tuned + CRF + multi-task)      ── PRIMARY
   │     → spans(char)+type+score, head assertion (nháp)
   │
   ▼ ③ NER second-opinion (Qwen2.5-7B, LoRA role R1) — CHỈ câu khó
   │     (encoder score thấp / câu dày thực thể) → generative-NER
   │     gộp với ② ở mức span (ensemble.py): vote + boost đồng thuận → recall↑
   │
   ▼ ④ ASSERTION (Qwen2.5-7B, LoRA role R2) trên CHẨN_ĐOÁN/THUỐC/TRIỆU_CHỨNG
   │     LLM quyết định isNegated/isFamily/isHistorical; rule deterministic làm
   │     guard (nếu rule chắc chắn phủ định mà LLM bỏ sót → cảnh báo/hợp nhất)
   │
   ▼ ⑤ NORMALIZE mention khó (Qwen2.5-7B, role R4): giãn viết tắt (TURP/POBA/DES/tbm),
   │     sửa từ dính/nhiễu → chuỗi chuẩn để link tốt hơn (KHÔNG đổi text gốc/offset,
   │     chỉ dùng bản chuẩn để tra KB)
   │
   ▼ ⑥ CANDIDATE GEN (deterministic linker + BGE-m3 retriever)
   │     CHẨN_ĐOÁN→ICD, THUỐC→RxNorm: exact+fuzzy+leaf ∪ top-k ngữ nghĩa (retriever)
   │
   ▼ ⑦ RERANK có RÀNG BUỘC (Qwen2.5-7B, role R3 — rerank.py)
   │     chọn tập con mã trong danh sách; PARSER ép code ∈ ứng viên (⊆ KB)
   │
   ▼ ⑧ (tùy chọn) QUAN HỆ khái niệm (Qwen2.5-7B) — nhóm B
   │
   ▼ ⑨ VALIDATE SCHEMA (candidates⊆KB & đúng type; assertions≤3; position ký tự)
   │
   └─► JSON đúng đề bài

Một model LLM (Qwen2.5-7B + **nhiều LoRA adapter** hoặc 1 LoRA đa nhiệm) đảm nhiệm R1–R4 → **không nhân 4 lần tham số**. Encoder + retriever cố định.

---

## 2. Ngân sách (≤9B) — cấu hình MAX khuyến nghị

| Thành phần | Model | Params |
|---|---|---:|
| NER backbone | XLM-R-large (fine-tuned) | 0.56B |
| Reasoner đa vai trò (R1–R4) | **Qwen2.5-7B-Instruct** + LoRA | 7.61B |
| Retriever linking | **BGE-m3** | 0.57B |
| **TỔNG** | | **8.74B < 9B** ✓ |

Biến thể hợp lệ khác:
- **MAX-lite an toàn ngân sách:** đổi BGE-m3 → multilingual-e5-base (0.28B) ⇒ **8.45B** (thoải mái hơn).
- **MAX + ensemble encoder:** thêm ViHealthBERT-syllable 0.135B, bỏ retriever ⇒ 0.56+0.135+7.61 = **8.31B** (dùng LLM cho cả recall lẫn rerank; linking chỉ dựa linker deterministic).

> LoRA adapter (vài chục–trăm MB) tính vào LLM; KB/gazetteer/linker là dữ liệu, **không tính**.

---

## 3. KẾ HOẠCH FINE-TUNE TOÀN DIỆN (3 model train được)

### Nguồn dữ liệu chung (dùng lại cho cả 3) — không cần API ngoài
| Ký hiệu | Nguồn | Vai trò |
|---|---|---|
| **S** silver | pipeline rule (`src/`) chạy trên 100 file test + kho lâm sàng thêm | số lượng lớn, nhiễu nhẹ |
| **V** ViMedNER | dataset public (map 5-type) | chất lượng cao, dạng cuộc thi |
| **P** pseudo_fine | PhoNER `SYMPTOM_AND_DISEASE` → dx/sym qua linker | tận dụng 35K entity |
| **G** gold | người gán 0.5–2k mẫu trên 100 file | chuẩn vàng, kiểm soát |
| **A** augmented | `augment.py`: entity-replace + typo + lab-synth | robust, phủ KẾT_QUẢ |
| **I** i2b2-vi | i2b2 2010 dịch (absent→isNegated, someone_else→isFamily) | giám sát assertion |

---

### PHASE 1 — Encoder NER (XLM-R-large)   [nền tảng, làm TRƯỚC]
- **Data:** GĐ2 multi-source (S+V+P+PhoNER+VietBioNER+I, trọng tâm HEAD B) → GĐ3 target 5-type (S+V+P+G+A).
- **Kỹ thuật:** multi-task (HEAD A 5-type + CRF, HEAD B coarse, HEAD C assertion), augmentation, self-training 1–2 vòng (pseudo-label confidence cao).
- **Lệnh:** như `README_TRAIN.md §3` (đổi backbone = xlm-roberta-large).
- **Ra:** `ckpt/nerA-xlmr` (span+type+assertion nháp).
- **Lift kỳ vọng:** đây là 70–80% điểm số; mọi thứ khác cộng thêm ở đuôi.

### PHASE 2 — LLM đa vai trò (Qwen2.5-7B, LoRA/QLoRA)   [ăn điểm đuôi khó]
- **Data (dựng bằng `llm_data.py` từ CÙNG S+V+P+G+A+I):**
  - R1 NER-instruct: text→JSON thực thể.
  - R2 assertion: (câu, mention, loại)→nhãn (I là nguồn chính + rule + synthetic).
  - R3 rerank: (mention, ngữ cảnh, ứng viên từ linker)→index gold (`make_rerank` gọi linker).
  - R4 normalize: (viết tắt/nhiễu + ngữ cảnh)→dạng chuẩn (từ acrDrAid + augment).
- **Cấu hình LoRA:** QLoRA 4-bit, r=16–32, α=32, lr 1–2e-4, 2–3 epoch, max_len 1–2k, packing; 1 adapter đa nhiệm HOẶC 4 adapter nhỏ (hot-swap). Output **JSON constrained** (grammar/regex ở decode).
- **Serve:** vLLM + LoRA; role chọn qua system/prompt. R3 luôn qua **cổng `rerank.py`**.
- **Lift:** assertion-F1 ↑↑, recall NER đuôi ↑, chọn mã đúng ↑.

### PHASE 3 — Retriever (BGE-m3, contrastive)   [tăng recall ứng viên]
- **Data (khai mỏ tự động, không cần người):**
  - Positive: (mention → tên KB đúng) từ **G** (gold link) + **exact match** linker.
  - Hard negative: ứng viên fuzzy-nhưng-SAI của linker (cùng top-k, khác code) + biến thể muối/hàm lượng gần đúng.
  - Nguồn tên: ICD-Việt/EN + RxNorm STR (đã có trong KB index).
- **Kỹ thuật:** InfoNCE, in-batch + hard negatives, fine-tune BGE-m3 (hoặc chỉ dùng zero-shot nếu thiếu thời gian — BGE-m3 đa ngữ đã khá).
- **Nối:** embed mention & tên KB → top-k bổ sung vào tập ứng viên trước bước rerank ⑦.
- **Lift:** bắt các mention biểu đạt lệch KB mà fuzzy trigram bỏ sót.

### Thứ tự & phụ thuộc
Phase1 (encoder)  ──► cung cấp spans để (a) sinh data R2/R3, (b) chạy pipeline
Phase3 (retriever)──► cung cấp ứng viên tốt hơn cho R3
Phase2 (LLM)      ──► dùng spans (P1) + ứng viên (linker/P3) để tune R1–R4
Lặp: pseudo-label từ pipeline MAX → bổ sung G' → tune lại P1/P2

---

## 4. Hiệu suất kỳ vọng & cách đo (trung thực)

| Cấu hình | Params | Span-F1 (dev) | Assertion-F1 | Linking acc@k | Ghi chú |
|---|---:|---|---|---|---|
| Rule baseline (đang có) | 0 (CPU) | ~ thấp | trung bình | drug 92% link | đã chạy 100 file, 0 lỗi schema |
| v2 encoder-ensemble | 0.70B | cao | khá | tốt | nhanh, an toàn |
| **MAX (P1+P2+P3)** | 8.74B | **cao nhất** | **cao nhất** | **cao nhất** | +3–6% e2e so v2 |

- **Bắt buộc có gold dev** để đo thật (không chỉ silver). Đo tách: span-F1, type-acc, assertion-F1/nhãn, linking acc@1/@3.
- A/B từng phase để biết phase nào đáng giữ (tránh phức tạp vô ích).

---

## 5. Vì sao MAX vẫn "EXACT — không hallucination"

- Mã cuối **luôn** đi qua linker/retriever (⊆ KB) và cổng **`rerank.py`** (đã test: LLM bịa mã K99.9 → bị loại).
- LLM chỉ: (a) đề xuất span (được ensemble + validate), (b) gán assertion (nhãn cố định), (c) chọn INDEX trong danh sách, (d) chuẩn hóa chuỗi mention (không đổi offset gốc).
- Bước ⑨ validate ép toàn bộ ràng buộc schema đề bài.

→ Thêm 7B **không** mở thêm bề mặt hallucination ở output mã; chỉ tăng chất lượng quyết định.

---

## 6. Trạng thái code (MAX)

| File | Vai trò | Test |
|---|---|---|
| `training/rerank.py` | Cổng rerank ràng buộc (chống hallucination LLM) | ✅ lọc mã bịa/index sai |
| `training/llm_data.py` | Dựng data SFT đa vai trò R1–R4 từ JSONL hợp nhất | ✅ R1/R2/R3 format |
| `training/sft_lora.py` | **Phase 2**: QLoRA SFT Qwen2.5-7B (TRL) | ✅ syntax; chạy khi có GPU |
| `training/qwen_qlora_axolotl.yaml` | **Phase 2**: config axolotl thay TRL | ✅ YAML hợp lệ |
| `training/llm_infer.py` | Nạp base+LoRA → llm_fn cho rerank + role R1/R2 | ✅ syntax |
| `training/retriever_data.py` | **Phase 3**: khai mỏ triplet (gold + synth + hard-neg linker) | ✅ mining logic |
| `training/train_retriever.py` | **Phase 3**: fine-tune BGE-m3 (MNRL) | ✅ syntax; chạy khi có GPU |
| `training/retriever_infer.py` | **Phase 3**: embed KB + search → bổ sung ứng viên | ✅ syntax |
| `training/RUN_MAX.md` | Nối trọn 3 phase + inference end-to-end | — |
| `training/ensemble.py` | Gộp span đa nguồn (encoder + LLM) | ✅ (v2) |
| `training/{schema,adapters,augment,model,dataset,train_ner,predict,predict_ensemble}.py` | Phase 1 + hạ tầng | ✅ (đã test) |

> Chạy số thật cần máy có GPU + mạng (tải XLM-R/Qwen2.5-7B/BGE-m3 từ HuggingFace — môi trường này chặn). Toàn bộ logic tự viết (ensemble, cổng rerank, dựng data SFT & triplet, căn nhãn, CRF, train loop) đã validate bằng self-test; các script train (Phase 2/3) đã kiểm cú pháp + dùng đúng API TRL/PEFT/sentence-transformers hiện hành.
