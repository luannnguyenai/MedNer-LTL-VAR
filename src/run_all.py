"""
import re
run_all.py — Chạy pipeline trên toàn bộ input/*.txt -> out/*.json và kiểm tra schema.

Dùng:
  python run_all.py <input_dir> <output_dir>
"""
import json
import os
import sys
import time
from pipeline import Pipeline

VALID_TYPES = {"TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
               "CHẨN_ĐOÁN", "THUỐC"}
VALID_ASSERT = {"isNegated", "isFamily", "isHistorical"}


def validate(items, text):
    errs = []
    for i, it in enumerate(items):
        for key in ("text", "position", "type", "assertions", "candidates"):
            if key not in it:
                errs.append(f"item{i}: thiếu '{key}'")
        if it.get("type") not in VALID_TYPES:
            errs.append(f"item{i}: type sai '{it.get('type')}'")
        p = it.get("position")
        if not (isinstance(p, list) and len(p) == 2 and 0 <= p[0] <= p[1] <= len(text)):
            errs.append(f"item{i}: position sai {p}")
        elif text[p[0]:p[1]] != it["text"]:
            errs.append(f"item{i}: text!=slice")
        a = it.get("assertions", [])
        if len(a) > 3 or any(x not in VALID_ASSERT for x in a):
            errs.append(f"item{i}: assertions sai {a}")
        # ràng buộc phạm vi
        if it.get("type") not in ("CHẨN_ĐOÁN", "THUỐC") and it.get("candidates"):
            errs.append(f"item{i}: candidates không được có cho {it.get('type')}")
        if it.get("type") not in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG") and a:
            errs.append(f"item{i}: assertions không được có cho {it.get('type')}")
    return errs


def main():
    indir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/work/input"
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "..", "out")
    os.makedirs(outdir, exist_ok=True)

    print("Loading pipeline (KB + gazetteer)...")
    t0 = time.time()
    pipe = Pipeline()
    print(f"  loaded in {time.time()-t0:.1f}s")

    files = sorted([f for f in os.listdir(indir) if f.endswith(".txt")],
                   key=lambda x: int(re.sub(r"\D", "", x) or 0))
    total_items = 0
    total_err = 0
    by_type = {}
    with_icd = with_rx = 0
    t0 = time.time()
    for fn in files:
        with open(os.path.join(indir, fn), encoding="utf-8") as f:
            txt = f.read()
        items = pipe.process(txt)
        errs = validate(items, txt)
        total_err += len(errs)
        if errs:
            print(f"  [!] {fn}: {errs[:3]}")
        total_items += len(items)
        for it in items:
            by_type[it["type"]] = by_type.get(it["type"], 0) + 1
            if it["type"] == "CHẨN_ĐOÁN" and it["candidates"]:
                with_icd += 1
            if it["type"] == "THUỐC" and it["candidates"]:
                with_rx += 1
        out_fn = fn.replace(".txt", ".json")
        with open(os.path.join(outdir, out_fn), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    print(f"\nProcessed {len(files)} files in {dt:.1f}s "
          f"({dt/max(len(files),1)*1000:.0f} ms/file)")
    print(f"Total concepts: {total_items}  | schema errors: {total_err}")
    print("By type:")
    for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"   {k:20s} {v}")
    print(f"CHẨN_ĐOÁN có mã ICD: {with_icd} | THUỐC có mã RxNorm: {with_rx}")


if __name__ == "__main__":
    import re
    main()
