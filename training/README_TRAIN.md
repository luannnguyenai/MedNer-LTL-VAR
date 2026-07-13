# Recipe fine-tune Encoder NER — đẩy khả năng lên tối đa

> Mục tiêu: từ các **dataset y tế tiếng Việt public** + label-projection từ tiếng Anh + silver-label (pipeline rule) → huấn luyện encoder NER đa nhiệm mạnh nhất trong ngân sách **≤9B**, **giữ đúng vị trí ký tự** và **không hallucination** (mã vẫn do linker cấp).
>
> Toàn bộ code trong `training/` đã chạy-thông (self-test tích hợp `selftest_train.py`). Phần dưới là quy trình vận hành cụ thể.

---

## 0. Dataset public — tải & vai trò

| Dataset | Lấy ở đâu | Nhãn gốc | Đóng góp cho 5-type |
|---|---|---|---|
| **PhoNER_COVID19** | `github.com/VinAIResearch/PhoNER_COVID19` hoặc HF `SEACrowd/pho_ner_covid` | 10 loại, có `SYMPTOM_AND_DISEASE` | HEAD B (PROBLEM) + biên/ngữ cảnh |
| **ViMedNER** | EAI 2024 (`publications.eai.eu/.../5221`, liên hệ tác giả) | disease/symptom/**diagnostic**/treatment/cause | **HEAD A**: disease→CHẨN_ĐOÁN, symptom→TRIỆU_CHỨNG, diagnostic→TÊN_XÉT_NGHIỆM |
| **VietBioNER** | ACL LREC 2022 (`aclanthology.org/2022.lrec-1.385`) | Symptom&Disease, **Diagnostic_Procedure** | HEAD A (TÊN_XÉT_NGHIỆM) + HEAD B |
| **ViMQ** | Huy et al. 2021 (repo tác giả) | symptom/medical entity + intent | HEAD B (PROBLEM) |
| **acrDrAid** (ViHealthBERT) | `github.com/demdecuong/vihealthbert` | 135 bộ viết tắt | Bảng giãn viết tắt (TURP/POBA/DES…) |
| **i2b2 2010** (dịch) | n2c2/DBMI (cần đăng ký, dùng offline) | problem/test/treatment + **assertion** | HEAD A (TÊN_XÉT_NGHIỆM) + **HEAD C assertion** |

> Bản quyền: các bộ này "research/education only" — hợp lệ cho cuộc thi. i2b2/n2c2 cần thỏa thuận dữ liệu; dịch & train **offline**, không đẩy dữ liệu ra API ngoài.

**Chuyển về JSONL hợp nhất** (đã có adapter):
```bash
cd training
python adapters.py            # xem demo chuyển đổi + kiểm tra offset
# thực tế:
python -c "from adapters import convert_phoner;   convert_phoner('PhoNER/word/train.conll','data/phoner.jsonl')"
python -c "from adapters import convert_vimedner;  convert_vimedner('ViMedNER/train.conll','data/vimedner.jsonl')"
# dataset dạng span-JSON: dùng load_spans_json(...) rồi write_jsonl(...)
```

---

## 1. Encoder — chọn & lý do (liên quan trực tiếp vị trí ký tự)

| Encoder | Params | Ưu | Nhược |
|---|---|---:|---|
| **XLM-R-large** ✅ khuyến nghị chính | 0.56B | subword **KHÔNG cần word-seg** → offset_mapping ra **char sạch**; tốt nhất trên ViMedNER | không chuyên tiếng Việt |
| **ViHealthBERT-base** | 0.135B | domain y tế Việt, SOTA COVID/ViMQ | cần **VnCoreNLP RDRSegmenter**; bản word-level làm lệch offset |
| PhoBERT-large | 0.37B | monolingual mạnh, > XLM-R trên PhoNER | cần word-seg |

**Vị trí ký tự (điểm phải cẩn thận):** đề chấm theo **offset ký tự trên RAW text**. XLM-R dùng SentencePiece subword → `return_offsets_mapping=True` cho `(char_start,char_end)` trực tiếp trên raw → không lệch. Nếu chọn PhoBERT/ViHealthBERT-**word**, phải:
1. word-segment bằng RDRSegmenter → lưu **mapping token↔offset gốc**;
2. train trên chuỗi đã seg nhưng **ánh xạ ngược span về offset raw** khi xuất.
→ Dùng bản **ViHealthBERT-syllable** (âm tiết) để tránh phần lớn rắc rối offset. **Khuyến nghị: XLM-R-large làm model chính**; ViHealthBERT dùng cho ensemble (§6).

---

## 2. Xử lý bất đối xứng schema (đã cài trong `schema.py`)

- **HEAD A (5-type, CRF)** — output cuộc thi. Chỉ học ở dữ liệu có phân biệt 5 chiều: **silver** (pipeline rule), **ViMedNER**, **gold người gán**, **i2b2 test→TÊN_XÉT_NGHIỆM**, **augmented**.
- **HEAD B (thô PROBLEM/TEST/TREATMENT/OTHER)** — học ở **mọi** nguồn (PhoNER, VietBioNER, ViMQ, i2b2 problem…). Dạy encoder nhận **biên** thực thể + biểu diễn miền. Auxiliary loss `λ_B=0.3`.
- **HEAD C (assertion, BCE)** — học từ **i2b2 (absent→isNegated, someone_else→isFamily)** + rule-label (isHistorical theo section) + synthetic. `λ_C=0.5`.

**Chuyển PROBLEM thô → CHẨN_ĐOÁN/TRIỆU_CHỨNG (tinh)** để tạo thêm nhãn cho HEAD A:
> span PROBLEM (từ PhoNER/i2b2) → chạy `ICDLinker.link`; **link ICD mạnh** ⇒ CHẨN_ĐOÁN, ngược lại ⇒ TRIỆU_CHỨNG. Tận dụng chính linker đã có, biến dữ liệu coarse thành pseudo-label fine.

`KẾT_QUẢ_XÉT_NGHIỆM` **không có** trong dataset public nào → nguồn: **silver (regex số+đơn vị, độ chính xác cao)** + **synthetic** (`augment.add_lab_synth`).

---

## 3. Quy trình 3 giai đoạn (khuyến nghị để "maximize")

### GĐ1 — Domain-Adaptive Pretraining (DAPT/TAPT), tùy chọn nhưng hiệu quả
Tiếp tục MLM encoder trên **văn bản lâm sàng Việt không nhãn** (100 file test + ViMedNER raw + bệnh án dịch từ MIMIC + crawl trang y tế). Rẻ, tăng mạnh miền.
```bash
# dùng script MLM chuẩn của HF (run_mlm.py) — offline, không cần nhãn
python run_mlm.py --model_name_or_path xlm-roberta-large \
  --train_file data/clinical_unlabeled.txt --line_by_line \
  --max_seq_length 256 --do_train --output_dir ckpt/xlmr-dapt \
  --per_device_train_batch_size 8 --num_train_epochs 1 --fp16
```

### GĐ2 — Multi-source NER pretraining (auxiliary)
Train encoder + HEAD A/B/C trên **hợp nhất mọi nguồn** (trọng tâm HEAD B để phủ biên). Cho encoder "thấy" tối đa thực thể y khoa.
```bash
python train_ner.py --encoder ckpt/xlmr-dapt \
  --train data/phoner.jsonl data/vietbioner.jsonl data/vimq.jsonl \
          data/vimedner.jsonl data/i2b2_vi.jsonl \
  --dev data/gold_dev.jsonl \
  --epochs 5 --bs 8 --lr 2e-5 --lambda_b 0.5 --lambda_c 0.5 --out ckpt/ner-multi
```

### GĐ3 — Target fine-tuning (schema cuộc thi)
Fine-tune tiếp trên dữ liệu **đúng 5-type**: silver + ViMedNER (map fine) + pseudo-fine (PROBLEM→dx/sym qua linker) + **gold người gán** + augmented. HEAD A là bản dùng để nộp.
```bash
python augment.py    # sinh data/aug.jsonl (entity-replace + typo + lab-synth)
python train_ner.py --encoder ckpt/ner-multi \
  --train data/silver.jsonl data/vimedner.jsonl data/pseudo_fine.jsonl \
          data/gold_train.jsonl data/aug.jsonl \
  --dev data/gold_dev.jsonl \
  --epochs 8 --bs 8 --lr 1.5e-5 --lambda_b 0.2 --lambda_c 0.5 --out ckpt/nerA
```

---

## 4. Kỹ thuật đẩy chất lượng tối đa (đã hoặc dễ bật)

1. **CRF linear-chain** (đã cài, `model.py`): chặn chuyển BIO bất hợp lệ (O→I, B-X→I-Y) → span mạch lạc hơn softmax.
2. **Multi-task** (đã cài): HEAD B/C chia sẻ encoder → regularize, tăng biểu diễn miền.
3. **Augmentation** (đã cài, `augment.py`): thay thực thể theo gazetteer ICD/RxNorm; tiêm lỗi/gỡ dấu/dính từ đúng đặc thù test; synth KẾT_QUẢ.
4. **Self-training / pseudo-label**: dùng `nerA` gán nhãn kho lâm sàng chưa nhãn → giữ span **confidence cao** (CRF score) → thêm vào GĐ3, train lại. Lặp 1–2 vòng.
5. **Class-imbalance**: focal/weighted loss cho HEAD A (KẾT_QUẢ & TÊN_XÉT_NGHIỆM hiếm) — chỉnh trong `MultiTaskNER` (thay `CrossEntropyLoss`→focal cho nhánh softmax, hoặc weight cho CRF emissions).
6. **Ensemble** (nếu ngân sách cho phép, §6): XLM-R-large + ViHealthBERT → trung bình emission/vote span.
7. **Biaffine/Span-head thay BIO** (nâng cao): head span-classification/biaffine bắt biên & lồng nhau tốt hơn — hữu ích cho cụm dài "bệnh trào ngược dạ dày - thực quản". Thay HEAD A, giữ nguyên phần còn lại.
8. **Word-segmentation feature** (nếu dùng PhoBERT): thêm VnCoreNLP RDRSegmenter → PhoNER cho thấy seg giúp NER.

---

## 5. Đánh giá

- **Span-F1 HEAD A** (đã có trong `train_ner.evaluate`): exact-match (start,end,type).
- Bổ sung khi cần: **type-accuracy**, **assertion-F1** (per nhãn), **boundary-relaxed F1** (overlap ≥ 0.5).
- **Linking acc@k** đo riêng ở tầng linker (lưu ý phiên bản RxNorm — xem `ARCHITECTURE.md §4`).
- **Bắt buộc có gold người gán** cho dev/test thật (không chỉ silver) để số đo trung thực.

---

## 6. Ngân sách khi ensemble (kiểm tra ≤9B)

| Cấu hình | Encoder(s) | + Retriever | + LLM reranker | Tổng |
|---|---|---|---|---:|
| Gọn (an toàn) | XLM-R-large 0.56B | — | 7B | **7.56B** |
| Chuẩn | XLM-R-large 0.56B | e5-base 0.28B | 7B | **7.84B** |
| Ensemble | XLM-R-large 0.56B + ViHealthBERT 0.135B | e5-base 0.28B | 7B | **~7.98B** ✓ |

Ba encoder + 7B sẽ **sát/vượt trần** → tối đa **2 encoder**. Nếu chỉ cần NER top, có thể **bỏ LLM** và dồn cho ensemble encoder + retriever (vẫn « 9B).

---

## 7. Nối lại pipeline (thay extractor rule)

`training/predict.py::NERPredictor.predict(text)` xuất đúng schema
`{text, position, type, assertions, candidates}` và **gọi linker** cho candidates.
Trong `src/pipeline.py`, thay `Extractor` bằng `NERPredictor` là xong — tầng
`linker` + `validate` (bảo chứng EXACT/không hallucination) **giữ nguyên**.

```python
# pipeline_ml.py (phiên bản dùng model)
from training.predict import NERPredictor
pred = NERPredictor("ckpt/nerA", encoder="xlm-roberta-large")
items = pred.predict(open("input/1.txt").read())   # -> list dict đúng schema
```

---

## 8. Trạng thái code training (đã kiểm thử)

| File | Chức năng | Test |
|---|---|---|
| `schema.py` | Không gian nhãn + ánh xạ đa nguồn | ✅ in bảng ánh xạ |
| `adapters.py` | CoNLL/spans-JSON → JSONL hợp nhất (char offset) | ✅ offset khớp text |
| `augment.py` | entity-replace + typo + lab-synth | ✅ offset đúng sau biến đổi |
| `model.py` | encoder + 2 BIO head + CRF + assertion | ✅ forward/backward/decode |
| `dataset.py` | align nhãn theo offset_mapping | ✅ subword B/I + assertion |
| `train_ner.py` | vòng train + span-F1 | ✅ (qua selftest) |
| `selftest_train.py` | integration (tokenizer giả, model nhỏ) | ✅ loss giảm, eval chạy |
| `predict.py` | infer → span → linker | ✅ wiring (cần ckpt thật để chạy số) |

> Chưa chạy được số thật ở đây vì tải XLM-R/ViHealthBERT cần mạng HuggingFace (môi trường này chặn). Trên máy có mạng, chạy đúng các lệnh §0–§3 là ra checkpoint.
