"""
build_drug_gazetteer.py — Trích danh sách HOẠT CHẤT (ingredient) & brand từ RxNorm
để làm gazetteer nhận diện THUỐC trong văn bản.

Chỉ lấy TTY in {IN, PIN, MIN, BN} và độ dài hợp lý -> tránh nhiễu.
Đầu ra: kb/drug_gazetteer.json = {"names": [name_lowercase, ...]}  (đã khử trùng, sort theo độ dài giảm dần để match longest-first)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vntext import norm_key  # noqa

KEEP = {"IN", "PIN", "MIN", "BN"}
STOP = {"water", "air", "oxygen", "alcohol", "starch", "glucose", "honey",
        "menthol", "camphor", "caffeine"}  # từ quá phổ thông, dễ FP (giữ caffeine? bỏ)


def build(rrf, out):
    names = set()
    with open(rrf, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) < 15:
                continue
            lat, sab, tty, s = p[1], p[11], p[12], p[14]
            if lat != "ENG" or sab != "RXNORM" or tty not in KEEP:
                continue
            k = norm_key(s)
            # bỏ tên quá ngắn (<=3) hoặc chỉ số
            if len(k) < 4:
                continue
            if k in STOP:
                continue
            # bỏ tên có ngoặc/nhiều thành phần rất dài (MIN) để tránh over-match
            if k.count(" ") > 5:
                continue
            names.add(k)
    names = sorted(names, key=lambda x: (-len(x), x))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"names": names}, f, ensure_ascii=False)
    print(f"[gazetteer] drug names={len(names)} -> {out}")


if __name__ == "__main__":
    rrf = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/RXNCONSO.RRF"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "drug_gazetteer.json")
    build(rrf, out)
