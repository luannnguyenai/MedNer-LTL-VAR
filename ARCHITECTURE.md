# Kiến trúc hệ thống — Chuẩn hóa Khái niệm Y khoa Tiếng Việt & Suy luận Ontology

> Bản thiết kế cho hệ thống thi đấu: **self-host hoàn toàn, tổng tham số mọi model ≤ 9B, EXACT — không hallucination ở output cuối.**
> Repo kèm theo đã hiện thực đầy đủ tầng KB + linker + assertion + pipeline baseline (chạy được ngay trên 100 file test).

---

## 0. Tóm tắt quyết định thiết kế

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Không hallucination mã | **Mã CHỈ đến từ retrieval trên KB**; model sinh không bao giờ tự "đọc ra" mã | Mã ICD/RxNorm là định danh — sinh tự do = bịa. Tách "hiểu ngôn ngữ" khỏi "cấp mã". |
| ≤ 9B tham số | Encoder NER (~0.55B) + Retriever bi-encoder (~0.3B) + LLM reasoner 7B = **~7.85B** | Còn headroom, tránh Gemma-2-9B sát trần. |
| Self-host | vLLM (LLM) + ONNX/Torch (encoder) — **0 lời gọi API ngoài lúc chạy** | Ràng buộc cứng của đề. |
| Dữ liệu train | Silver-label bằng chính pipeline rule + chiếu nhãn từ bộ i2b2/n2c2 dịch sang tiếng Việt + distant supervision từ gazetteer | Đề cho phép/khuyến khích tạo thêm data ngoài lời giải chính. |

**Nhận định dữ liệu quan trọng:** văn bản test có cấu trúc mục 1/2/3, tên thuốc tiếng Anh (`metoprolol`, `nitroglycerin`, `DES`, `POBA`, `TURP`), lối diễn đạt rất giống **bệnh án MIMIC/i2b2 dịch máy sang tiếng Việt**. Điều này mở ra hướng **chiếu nhãn (label projection)** từ các bộ NER/assertion tiếng Anh có sẵn — đặc biệt **i2b2 2010 Assertion** (present/absent/possible/hypothetical/conditional/**family**) ánh xạ gần như 1-1 với `isNegated`/`isHistorical`/`isFamily`.

---

## 1. Sơ đồ pipeline (inference)

```
                 ┌─────────────────────────────────────────────────────────┐
 free-form text  │ 1. TIỀN XỬ LÝ CẤU TRÚC                                   │
 ───────────────▶│  - tách mục 1/2/3, bullet, "Nhãn: giá trị"               │
                 │  - chuẩn hóa khoảng trắng, gộp từ dính ("atenololtrong") │
                 └───────────────┬─────────────────────────────────────────┘
                                 ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │ 2. NER (encoder token-classification, ~0.55B)            │
                 │  BIO span + 5 nhãn type. Multi-task head #2: assertion.  │
                 └───────────────┬─────────────────────────────────────────┘
                                 ▼
          ┌──────────────────────┴───────────────────────┐
          ▼ (CHẨN_ĐOÁN / THUỐC)                          ▼ (mọi type: TRIỆU_CHỨNG/…)
 ┌───────────────────────────────┐            ┌────────────────────────────────┐
 │ 3. ỨNG VIÊN (candidate gen)   │            │ 3'. ASSERTION                  │
 │  a) Linker LEXICAL (exact+    │            │  head multi-task HOẶC rule     │
 │     fuzzy trigram) — repo này │            │  (negation scope, family, hist)│
 │  b) Retriever BI-ENCODER      │            └────────────────────────────────┘
 │     (embed span vs KB names)  │
 │  → hợp nhất top-N ỨNG VIÊN KB │
 └───────────────┬───────────────┘
                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ 4. RERANK có RÀNG BUỘC (LLM 7B, JSON-constrained)             │
 │  Input: span + ngữ cảnh + DANH SÁCH tên ứng viên (kèm mã)     │
 │  Output: chọn tập con mã ĐÚNG NHẤT — BẮT BUỘC thuộc danh sách  │
 │  (không có ứng viên phù hợp → trả []).                        │
 └───────────────┬───────────────────────────────────────────────┘
                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ 5. LẮP RÁP + VALIDATE SCHEMA (ràng buộc đề bài)               │
 │  candidates ⊆ KB; ⊆ đúng type; assertions ≤3; position ký tự  │
 └───────────────────────────────────────────────────────────────┘
```

**Điểm chốt chống hallucination (bước 4):** LLM chỉ *chọn trong danh sách* mã đã retrieve từ KB, dùng **constrained decoding** (grammar/regex ép output là chỉ số trong danh sách ứng viên). Mọi mã cuối cùng đều `assert code in KB`. Nếu LLM "muốn" mã ngoài danh sách → bị chặn ở decoder + bị lọc ở bước 5.

---

## 2. Ngân sách tham số (≤ 9B)

| Thành phần | Model đề xuất | Tham số | Vai trò |
|---|---|---:|---|
| NER + Assertion encoder | **XLM-RoBERTa-large** (hoặc PhoBERT-large / ViHealthBERT) | ~0.56B | Gán span + 5 type + 3 assertion (multi-task) |
| Retriever bi-encoder | **multilingual-e5-base** (hoặc BGE-m3 distil) | ~0.28B | Embed span & tên KB → truy hồi ngữ nghĩa ứng viên |
| LLM reasoner/reranker | **Qwen2.5-7B-Instruct** (hoặc SeaLLM-7B / Vistral-7B) | 7.0B | Rerank ứng viên, giãn viết tắt, suy luận quan hệ |
| **Tổng** | | **≈ 7.85B** | **< 9B ✓** |

Phương án tiết kiệm hơn (nếu muốn biên an toàn): bỏ retriever bi-encoder, chỉ dùng linker lexical (đã có trong repo) + LLM 7B → ~7.6B, vẫn mạnh vì linker lexical đã cho recall tốt trên tên chuẩn (xem §5).

> Lưu ý cách "đếm tham số": chỉ tính **trọng số model tải lúc chạy**. Linker lexical, gazetteer, KB index (§4) là cấu trúc dữ liệu, **không tính** vào ngân sách.

---

## 3. Đầu ra từng tầng & ràng buộc đề bài

Schema mỗi khái niệm: `{text, position:[start,end], type, assertions[], candidates[]}`.

Ràng buộc được **ép ở tầng validate** (`src/run_all.py::validate`), không tin tưởng model:
- `candidates` chỉ cho `CHẨN_ĐOÁN` (ICD) & `THUỐC` (RxNorm); type khác → `[]`.
- `assertions` chỉ cho `CHẨN_ĐOÁN`/`THUỐC`/`TRIỆU_CHỨNG`; ≤ 3; ∈ {isNegated, isFamily, isHistorical}.
- `position` theo **ký tự**, 0-indexed, và `text == input[start:end]` (kiểm tra cứng).
- Mọi mã `candidates` phải tồn tại trong KB (`code in kb`).

---

## 4. Tầng tri thức (KB) — đã hiện thực

- **ICD-10** (`kb/build_icd.py`): 15.845 mã song ngữ Anh–Việt. Index: `exact{norm_key→[codes]}` + fuzzy trigram + map `parent→children` để mở rộng mã 3 ký tự ra mã lá (K21 → K21.0/K21.9, khớp đáp án mẫu).
- **RxNorm** (`kb/build_rxnorm.py`): parse `RXNCONSO.RRF`, giữ `SAB=RXNORM`, 140k tên, 81k RXCUI; ưu tiên TTY (SCD>SBD>SCDC>…); xử lý biến thể **muối** (maleate/HCl/…) và **hàm lượng**.
- **Gazetteer thuốc** (`kb/build_drug_gazetteer.py`): 12.390 tên hoạt chất/brand để NER bắt thuốc.

### ⚠ Rủi ro phiên bản RxNorm (phải báo BTC)
File `RXNCONSO.RRF` được cấp **KHÔNG** chứa các RXCUI trong ví dụ đề bài:
- `Chlorpheniramine 0.4 MG/ML` → đề nói `360047` (dạng base), **file cấp chỉ có** `chlorpheniramine *maleate* 0.4 MG/ML` = `996986`.
- `Capsaicin 0.38 MG/ML` → đề nói `1660761`, **file cấp chỉ có** bản `0.35`/`0.33 MG/ML`.

→ Bộ RxNorm được cấp là **phiên bản khác/nhỏ hơn** bộ tạo đáp án. **Quyết định thiết kế:** map theo **KB được cấp** (nguồn chân lý lúc chạy, đúng luật "không API ngoài"); linker để **cấu hình đổi KB** nếu BTC xác nhận dùng bản RxNorm đầy đủ. Cần hỏi rõ BTC bộ nào dùng để chấm.

---

## 5. Bằng chứng linker hoạt động (đã test trên dữ liệu thật)

```
bệnh trào ngược dạ dày - thực quản  -> ['K21.0', 'K21.9']   ✓ khớp đáp án mẫu
Nhồi máu cơ tim                     -> ['I25.2', 'I21', 'I22']
rung nhĩ                            -> ['I48.1', 'I48.2', 'I48.0']
đái tháo đường típ 2                -> ['E11', 'E11.0', 'E11.2†']
Chlorpheniramine 0.4 MG/ML          -> ['996986']  (bản maleate trong KB cấp)
aspirin 325mg                       -> ['317300']
metoprolol / doxycycline / atenolol -> 6918 / 3640 / 1202
```
Chạy toàn bộ 100 file: **0 lỗi schema**, ~160 ms/file (baseline, không cần GPU), thuốc link RxNorm **190/206 (92%)**.

---

## 6. Chiến lược DỮ LIỆU HUẤN LUYỆN (không dùng API ngoài)

Bốn nguồn, hợp nhất → tập train cho encoder NER/assertion + fine-tune LLM reranker:

1. **Silver-label bằng pipeline rule (repo này).** Chạy trên 100 file test + corpus lâm sàng tiếng Việt bổ sung → nhãn thô. Dùng làm *distant supervision*.
2. **Chiếu nhãn từ dữ liệu tiếng Anh** (mấu chốt). Bộ **i2b2 2010** (concept + **assertion**), **n2c2 2018 ADE/medication**, **NCBI-Disease**, **BC5CDR** có nhãn chuẩn. Dịch câu sang tiếng Việt bằng **MT offline** (NLLB-200/ виНMT — cũng self-host), rồi **chiếu span qua word-alignment** (awesome-align). Assertion i2b2 (absent→isNegated, family→isFamily, hypothetical/historical→isHistorical) cho **giám sát assertion chất lượng cao** mà tiếng Việt đang thiếu.
3. **Distant supervision từ gazetteer.** Quét tên ICD-Việt & RxNorm trong văn bản → auto span + **gold code miễn phí** (vì khớp trực tiếp KB). Sinh nhiều cặp (span, code) để train retriever bi-encoder (contrastive) + reranker.
4. **Sinh tổng hợp (synthetic).** (a) Template bệnh án Việt với slot bệnh/thuốc/triệu chứng đã biết nhãn; (b) dùng **LLM 7B self-host** để *paraphrase*/thêm viết tắt-lỗi chính tả (đúng luật: LLM local, không API), tăng độ bền với `tbm`, `POBA`, từ dính.

**Vòng lặp:** silver → train v0 → dự đoán → người sửa mẫu nhỏ (active learning) → train v1 … Chỉ cần vài trăm–vài nghìn câu gold để encoder vượt xa rule.

---

## 7. Xử lý các hiện tượng khó (đã thấy trong data)

| Hiện tượng | Ví dụ trong test | Cách xử lý |
|---|---|---|
| Viết tắt/thuật ngữ | `TURP`, `POBA`, `DES`, `tbm`, `MIBI`, `EF30` | LLM giãn nghĩa → text chuẩn → linker; gazetteer viết tắt. |
| Từ dính | `atenololtrong`, `bình thườngbình thường` | Chuẩn hóa tách từ ở tiền xử lý; encoder subword bền. |
| Phủ định phạm vi danh sách | "Không buồn nôn, hay nôn, đổ mồ hôi" | Negation scope qua dấu phẩy (đã làm trong `assertions.py`). |
| Pseudo-negation | "không **đặc hiệu**", "không **xác định**" | Danh sách loại trừ (đã làm). |
| "Tiền sử" ≠ historical | "Tiền sử **bệnh hiện tại**" = HPI | Phân biệt theo section (đã làm: `HIST_SECTION_EXCLUDE`). |
| Family ≠ người nhà quan sát | "người nhà nhận thấy…" (không phải bệnh của người nhà) | Family regex theo ranh giới từ + yêu cầu quan hệ sở hữu bệnh (đã làm). |
| Mã 3 ký tự vs mã lá | GERD → K21 vs K21.0/K21.9 | Mở rộng parent→children (đã làm). |

---

## 8. Suy luận quan hệ (Ontological Reasoning) — nhóm (B)

Ngoài 3 assertion, quan hệ giữa khái niệm (triệu chứng→chẩn đoán, thuốc→điều trị bệnh, xét nghiệm→kết quả) được LLM 7B suy luận trên **đồ thị khái niệm của một văn bản**: input là danh sách concept đã chuẩn hóa + ngữ cảnh; output là các cạnh quan hệ (JSON-constrained). Vì chỉ thao tác trên concept đã neo KB nên vẫn không sinh mã mới. (Tầng này tùy phạm vi chấm điểm của BTC.)

---

## 9. Đánh giá

- **NER**: span-level P/R/F1; type accuracy.
- **Assertion**: F1 từng nhãn (isNegated/isFamily/isHistorical).
- **Linking**: accuracy@k, MRR so với gold code (lưu ý §4 — cần thống nhất KB chấm).
- **End-to-end**: khớp đầy đủ {text, type, position, assertions, candidates}.
- Dev set: giữ lại phần gold do người gán để đo thật, tránh chỉ đo trên silver.

---

## 10. Trạng thái hiện thực trong repo

| Thành phần | Trạng thái |
|---|---|
| KB ICD/RxNorm/gazetteer builders | ✅ Hoàn chỉnh, đã build |
| Linker deterministic (exact+fuzzy+leaf, salt/strength) | ✅ Production-grade, đã test |
| Assertion (negation/family/historical + pseudo/section) | ✅ Đã test 6/6 case |
| NER baseline (drug gazetteer + lab pattern + dx/sym) | ✅ Chạy 100 file, 0 lỗi schema — *dùng như bootstrap/silver-labeler* |
| Pipeline + validate schema + run_all | ✅ Hoàn chỉnh |
| Encoder NER fine-tuned / retriever / LLM reranker | ◻ Thiết kế + kế hoạch data (mục 2, 6) — bước tiếp theo |

Baseline rule hiện là **điểm khởi đầu & bộ gán nhãn bạc**, không phải lời giải cuối. Điểm mạnh đã sẵn sàng production là **tầng KB + linker + assertion + ràng buộc schema** — chính là phần bảo chứng "EXACT, không hallucination".
