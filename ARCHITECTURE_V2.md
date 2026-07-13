# Kiến trúc TỐI ƯU (v2) — dựa trên bằng chứng benchmark

> Bản v2 thay cho khuyến nghị encoder ở v1. Thay đổi cốt lõi: **chọn model theo BẰNG CHỨNG trên benchmark gần cuộc thi nhất (ViMedNER), không giáo điều "chọn model y khoa"**, + **ensemble + empirical selection**, + tận dụng domain-pretraining ĐÚNG CHỖ (assertion, viết tắt).

---

## 0. Đính chính quan trọng so với v1

Ở v1 tôi mô tả việc chọn XLM-R là "sai" và đề xuất thay bằng ViHealthBERT. **Kiểm tra lại benchmark cho thấy điều ngược lại trên loại dữ liệu của cuộc thi:**

| Benchmark | Loại văn bản | Model thắng |
|---|---|---|
| **ViMedNER** (2024) — *gần cuộc thi nhất*: bệnh/triệu chứng/xét nghiệm/điều trị từ web y tế | lâm sàng/mô tả bệnh | **XLM-R > ViHealthBERT, ViPubmedDeBERTa, ViDeBERTa, PhoBERT** |
| PhoNER_COVID19 | tin tức COVID | ViHealthBERT / PhoBERT > XLM-R |
| ViMQ | hỏi đáp y tế | ViHealthBERT |
| acrDrAid / ViMedNLI | viết tắt / suy luận | ViPubmedT5, ViHealthBERT |

→ **Kết luận đúng:** với dữ liệu cuộc thi (bệnh án/kết quả/EHR — dạng ViMedNER), **XLM-R-large là lựa chọn mạnh nhất theo số liệu**. Domain-pretraining KHÔNG tự động thắng; quy mô XLM-R-large thắng ở NER dạng này. Cách làm của kỹ sư: **A/B trên dev + ensemble**, không cãi nhau bằng cảm tính.

---

## 1. Nguyên tắc tối ưu

1. **Chọn theo dev-F1, không theo "nhãn hiệu".** Train nhiều encoder (đều là base/large rẻ), đo trên **gold dev**, chọn hoặc **ensemble** cái tốt nhất.
2. **Domain-pretraining dùng đúng chỗ:** nơi nó thực sự thắng — **assertion** (phủ định/gia đình/tiền sử) và **giãn viết tắt** (TURP/POBA/DES/tbm) — chứ không ép làm backbone NER.
3. **Grounding y khoa mạnh nhất = KB linker**, không phải BERT. Mã luôn từ KB → không hallucination.
4. **Tối giản để chắc thắng:** cấu hình mặc định KHÔNG cần LLM 7B — NER (ensemble) + linker deterministic đã đủ và an toàn nhất. LLM chỉ bật nếu dev chứng minh có lợi.

---

## 2. Pipeline tối ưu

```
 raw text
   │
   ▼  ① Tiền xử lý cấu trúc (mục 1/2/3, bullet, "Nhãn: giá trị", tách từ dính)
   │
   ▼  ② NER ENSEMBLE  (predict_ensemble.EnsembleNER)
   │     ├─ XLM-R-large   (0.56B)  — backbone chính (SOTA ViMedNER)
   │     └─ ViHealthBERT-syllable (0.135B) — thành viên ensemble (mạnh COVID/ViMQ)
   │     gộp SPAN ở mức char (ensemble.ensemble_spans): vote type + boost đồng thuận
   │
   ├─────► ③ ASSERTION
   │         ├─ head assertion (đa nhiệm, ưu tiên ViHealthBERT)
   │         └─ + rule deterministic (pseudo-negation, section) — precision cao
   │
   ▼  ④ GIÃN VIẾT TẮT (chỉ span khó): acrDrAid + ViPubmedT5 → text chuẩn
   │
   ▼  ⑤ LINKER (src/linker.py, DETERMINISTIC)
   │     CHẨN_ĐOÁN→ICD, THUỐC→RxNorm; exact+fuzzy+leaf; (tùy chọn) + retriever BGE-m3
   │
   ▼  ⑥ (TÙY CHỌN) LLM rerank có ràng buộc — chỉ khi dev cho thấy có lợi
   │
   ▼  ⑦ VALIDATE SCHEMA (candidates⊆KB, đúng type, assertions≤3, position ký tự)
   │
   └─► JSON đúng đề bài
```

**Điểm mấu chốt:** ② và ③ do model (đã fine-tune) đảm nhiệm; ⑤ (mã) luôn deterministic từ KB. Ensemble chỉ nâng NER, không đụng tới việc cấp mã.

---

## 3. Ngân sách ≤ 9B (tính lại cho v2)

| Cấu hình | Thành phần | Tổng | Khi nào dùng |
|---|---|---:|---|
| **A. Tối giản (KHUYẾN NGHỊ)** | XLM-R-large 0.56B + ViHealthBERT 0.135B | **0.70B** | Mặc định cuộc thi: nhanh, mã hoàn toàn deterministic, **không hallucination**, headroom cực lớn |
| **B. + Retriever ngữ nghĩa** | A + BGE-m3 0.57B | **1.27B** | Khi tên bệnh/thuốc biểu đạt lệch nhiều so KB |
| **C. + LLM rerank** | B + Qwen2.5-7B | **8.27B** | Chỉ khi dev chứng minh LLM rerank giúp |
| D. Đơn model | XLM-R-large 0.56B | 0.56B | Baseline nhanh nhất |

So với v1 (XLM-R 0.56 + retriever 0.28 + LLM 7B = 7.84B), **v2 mặc định nhẹ hơn nhiều (0.70B)** vì bỏ LLM khỏi đường mặc định — an toàn hơn cho tiêu chí "EXACT, không hallucination" và nhanh hơn khi chấm 100 file.

> Vì sao dám bỏ LLM ở mặc định: việc cấp mã đã do linker deterministic lo trọn; LLM chỉ thêm giá trị ở rerank/giãn viết tắt — thứ có thể thay bằng acrDrAid + retriever. Ít thành phần sinh = ít bề mặt hallucination = đúng tinh thần cuộc thi.

---

## 4. Vai trò từng model y khoa (đúng chỗ)

| Model | Đã pretrain trên | Dùng cho | Vì sao |
|---|---|---|---|
| **XLM-R-large** | 100 ngôn ngữ (2.5TB) | **Backbone NER** | SOTA trên ViMedNER — dạng dữ liệu cuộc thi |
| **ViHealthBERT-syllable** | Y khoa Việt (news/FAQ/clinical) | **Ensemble NER + head assertion** | Thắng PhoNER/ViMQ; syllable → offset char sạch, không cần VnCoreNLP |
| **ViPubmedT5** (220M) | 20M abstract PubMed dịch | **Giãn viết tắt / chuẩn hóa cụm** | SOTA acrDrAid/ViMedNLI |
| **BGE-m3** (0.57B) | đa ngữ, gồm y sinh | **Retriever ứng viên** (tùy chọn) | Recall ngữ nghĩa khi biểu đạt lệch KB |
| Qwen2.5-7B | đa ngữ + code + math | **Rerank có ràng buộc** (tùy chọn) | Suy luận chọn mã trong danh sách; constrained decoding |

---

## 5. Quy trình huấn luyện tối ưu (cập nhật)

Bỏ DAPT tự làm (tốn kém, lợi ích không chắc trên ViMedNER-style) — thay bằng **empirical selection + ensemble**:

```bash
cd training
# 1) Train 2 encoder song song (cùng dữ liệu 5-type: silver+vimedner+pseudo_fine+gold+aug)
python train_ner.py --encoder xlm-roberta-large \
   --train data/silver.jsonl data/vimedner.jsonl data/pseudo_fine.jsonl \
           data/gold_train.jsonl data/aug.jsonl \
   --dev data/gold_dev.jsonl --epochs 8 --out ckpt/nerA-xlmr

python train_ner.py --encoder demdecuong/vihealthbert-base-syllable \
   --train data/silver.jsonl data/vimedner.jsonl data/pseudo_fine.jsonl \
           data/gold_train.jsonl data/aug.jsonl \
   --dev data/gold_dev.jsonl --epochs 8 --out ckpt/nerA-vihealth

# 2) So dev-F1 từng model; rồi ENSEMBLE (gộp span, chọn theo trọng số dev)
python predict_ensemble.py input/1.txt \
   --member ckpt/nerA-xlmr     xlm-roberta-large                    1.0 \
   --member ckpt/nerA-vihealth demdecuong/vihealthbert-base-syllable 0.8
```

> Nếu 1 model vượt trội hẳn → dùng đơn model (cấu hình D). Nếu 2 model bù nhau (thường xảy ra vì train trên phân phối khác) → ensemble thường +2–4% F1.

Vẫn giữ nguyên: multi-task (HEAD B coarse phủ biên), CRF, augmentation, self-training, pseudo_fine (PROBLEM→dx/sym qua linker) — xem `README_TRAIN.md`.

---

## 6. Vì sao đây là "tối ưu cho cuộc thi"

1. **Đúng bằng chứng:** backbone = model thắng trên benchmark giống cuộc thi nhất (ViMedNER), không đoán mò.
2. **Bù trừ điểm yếu:** ensemble với ViHealthBERT vá phần văn phong COVID/hỏi đáp mà XLM-R yếu hơn.
3. **An toàn tuyệt đối phần mã:** linker deterministic + validate → 0 hallucination, đúng yêu cầu "y tế EXACT".
4. **Nhẹ & nhanh:** 0.70B mặc định, chạy CPU/GPU nhỏ cho 100 file; không phụ thuộc LLM nặng.
5. **Có đường nâng cấp rõ:** bật retriever/LLM chỉ khi dev chứng minh có lợi — không phức tạp hóa vô ích.

---

## 7. Trạng thái code (v2)

| File | Vai trò | Test |
|---|---|---|
| `training/ensemble.py` | Gộp span đa-model (bất biến tokenizer) | ✅ đồng thuận/bất đồng/agree-only |
| `training/predict.py` | Infer 1 model → span (+score) → linker | ✅ syntax; ✅ score wiring |
| `training/predict_ensemble.py` | Ensemble N model → gộp → linker | ✅ syntax/import |
| (còn lại) | schema/adapters/augment/model/dataset/train_ner | ✅ như v1 (đã test) |

> Số F1 thật cần chạy trên máy có mạng (tải XLM-R/ViHealthBERT từ HuggingFace — môi trường này chặn). Toàn bộ machinery ensemble/căn nhãn/train đã validate bằng self-test.
