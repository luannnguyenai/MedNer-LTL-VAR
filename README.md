# Med-NER-VN — Chuẩn hóa khái niệm y khoa tiếng Việt (self-host, ≤9B, no-hallucination)

Hệ thống phát hiện + chuẩn hóa khái niệm y khoa từ văn bản tự do tiếng Việt, ánh xạ
**CHẨN_ĐOÁN → ICD-10** và **THUỐC → RxNorm**, suy luận **isNegated / isFamily / isHistorical**.

Xem `ARCHITECTURE.md` cho thiết kế production (model ≤9B + chiến lược dữ liệu).
Phần dưới là cách chạy **tầng KB + linker + assertion + pipeline baseline** đã hiện thực.

## Cài đặt
```bash
pip install openpyxl          # phụ thuộc DUY NHẤT (đọc ICD .xlsx). Linker/NER thuần Python.
```

## Xây KB (1 lần) từ dữ liệu được cấp
```bash
python kb/build_icd.py            /path/to/ICD10.xlsx
python kb/build_rxnorm.py         /path/to/RXNCONSO.RRF
python kb/build_drug_gazetteer.py /path/to/RXNCONSO.RRF
```
Tạo `kb/icd_index.json`, `kb/rxnorm_index.json`, `kb/drug_gazetteer.json`.

## Chạy trên toàn bộ test
```bash
cd src
python run_all.py /path/to/input /path/to/output   # mỗi N.txt -> N.json
```
Bản chạy mẫu (100 file) nằm ở `out/`.

## Thử nhanh từng phần
```bash
cd src
python vntext.py       # chuẩn hóa tiếng Việt
python linker.py       # linker ICD + RxNorm (có ví dụ)
python assertions.py   # phủ định / gia đình / tiền sử
python extract.py      # NER trên câu ví dụ đề bài
python pipeline.py /path/to/1.txt   # JSON đầy đủ 1 file
```

## Cấu trúc
```
kb/    build_icd.py  build_rxnorm.py  build_drug_gazetteer.py  (+ *.json sau khi build)
src/   vntext.py  linker.py  assertions.py  extract.py  pipeline.py  run_all.py
out/   1.json … 100.json   (kết quả baseline)
ARCHITECTURE.md   README.md
```

## Đảm bảo "EXACT — không hallucination"
Mọi mã trong `candidates` đều lấy từ **retrieval trên KB** (`src/linker.py`), không do model
sinh tự do; tầng `run_all.py::validate` kiểm tra cứng schema + ràng buộc phạm vi. Nếu không đủ
tự tin, linker trả `[]` thay vì đoán mã.

## Lưu ý quan trọng
File `RXNCONSO.RRF` được cấp là **phiên bản khác** bộ tạo đáp án mẫu (một số RXCUI trong
ví dụ đề không tồn tại trong file). Hệ thống map theo KB được cấp; xem chi tiết & khuyến nghị
hỏi BTC ở `ARCHITECTURE.md` §4.
