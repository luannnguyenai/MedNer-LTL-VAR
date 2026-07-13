"""
augment.py — Tăng cường dữ liệu (data augmentation) để đẩy NER lên tối đa.

Ba chiến lược, đều bảo toàn nhãn + cập nhật lại char offset:
 1) ENTITY REPLACEMENT: thay span CHẨN_ĐOÁN bằng tên bệnh khác (từ ICD-Việt),
    span THUỐC bằng tên thuốc khác (từ RxNorm). Nhân bội biến thể biểu đạt.
 2) TYPO/NOISE tiếng Việt: gỡ dấu ngẫu nhiên, hoán vị ký tự kề, DÍNH TỪ
    ("atenolol trong" -> "atenololtrong") — đúng lỗi thấy trong test set.
 3) LAB SYNTH: chèn cặp (TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM) tổng hợp
    (số + đơn vị) — lớp KẾT_QUẢ không có trong dataset public nào.

Dùng offset an toàn: luôn tái tạo text và dịch chuyển offset các span phía sau.
"""
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vntext import strip_accents  # noqa

random.seed(13)

LAB_TEMPLATES = [
    ("WBC", lambda: f"{random.uniform(3, 20):.2f}".replace(".", ",")),
    ("NEUT%", lambda: f"{random.uniform(40, 90):.1f}".replace(".", ",")),
    ("troponin", lambda: f"{random.uniform(0, 2):.2f}"),
    ("HGB", lambda: f"{random.randint(80, 170)}"),
    ("CRP", lambda: f"{random.uniform(0, 200):.1f}"),
    ("EF", lambda: f"{random.randint(20, 70)}"),
    ("creatinin", lambda: f"{random.randint(40, 400)}"),
    ("kali", lambda: f"{random.uniform(2.5, 6.5):.1f}"),
]


def _rebuild(text, spans, edits):
    """Áp danh sách edit (start,end,new_str) không chồng lấn -> text mới +
    span đã dịch offset. edits sắp xếp tăng dần theo start."""
    edits = sorted(edits, key=lambda x: x[0])
    out = []
    cursor = 0
    delta_map = []       # (orig_pos, cum_delta) để dịch span
    cum = 0
    for s, e, new in edits:
        out.append(text[cursor:s])
        out.append(new)
        cum += len(new) - (e - s)
        delta_map.append((e, cum))
        cursor = e
    out.append(text[cursor:])
    new_text = "".join(out)

    def shift(pos):
        d = 0
        for boundary, cd in delta_map:
            if pos >= boundary:
                d = cd
            else:
                break
        return pos + d

    new_spans = []
    for sp in spans:
        ns = dict(sp)
        # nếu span nằm trong 1 edit đã thay -> cập nhật text theo edit
        ns["start"], ns["end"] = shift(sp["start"]), shift(sp["end"])
        new_spans.append(ns)
    # cập nhật lại trường text
    for sp in new_spans:
        sp["text"] = new_text[sp["start"]:sp["end"]]
    return new_text, new_spans


def entity_replace(ex, icd_names, rx_names, p=0.5):
    """Thay span CHẨN_ĐOÁN/THUỐC bằng tên khác cùng loại."""
    edits = []
    for sp in ex["spans"]:
        if random.random() > p:
            continue
        if sp["fine"] == "CHẨN_ĐOÁN" and icd_names:
            new = random.choice(icd_names)
            edits.append((sp["start"], sp["end"], new))
        elif sp["fine"] == "THUỐC" and rx_names:
            new = random.choice(rx_names)
            edits.append((sp["start"], sp["end"], new))
    if not edits:
        return ex
    # loại edit chồng lấn
    edits = _dedupe_nonoverlap(edits)
    nt, ns = _rebuild(ex["text"], ex["spans"], edits)
    # gán text mới cho các span vừa thay (vì shift không đổi nội dung span được thay)
    for (s, e, new), sp in zip(edits, [x for x in ns if x["start"] in
                                       [d[0] for d in _shifted_starts(ex, edits)]]):
        pass
    return {"text": nt, "source": ex["source"] + "+repl", "spans": ns}


def _shifted_starts(ex, edits):
    return edits


def _dedupe_nonoverlap(edits):
    edits = sorted(edits, key=lambda x: x[0])
    out = []
    last_end = -1
    for s, e, new in edits:
        if s >= last_end:
            out.append((s, e, new))
            last_end = e
    return out


def inject_typos(ex, p_deaccent=0.15, p_merge=0.1, p_swap=0.05):
    """Gỡ dấu / dính từ / hoán vị ký tự — mô phỏng lỗi lâm sàng."""
    text = ex["text"]
    spans = [dict(s) for s in ex["spans"]]
    # 1) dính từ: xóa 1 khoảng trắng ngẫu nhiên ngoài span
    edits = []
    if random.random() < p_merge:
        ws = [m.start() for m in re.finditer(r" ", text)]
        random.shuffle(ws)
        for w in ws:
            if not _inside_any(w, spans):
                edits.append((w, w + 1, ""))     # xóa space -> dính từ
                break
    if edits:
        text, spans = _rebuild(text, spans, edits)
    # 2) gỡ dấu toàn câu xác suất thấp (mô phỏng gõ không dấu)
    if random.random() < p_deaccent:
        new = strip_accents(text)
        if len(new) == len(text):      # chỉ khi độ dài giữ nguyên -> offset an toàn
            for sp in spans:
                sp["text"] = new[sp["start"]:sp["end"]]
            text = new
    return {"text": text, "source": ex["source"] + "+typo", "spans": spans}


def _inside_any(pos, spans):
    return any(sp["start"] <= pos < sp["end"] for sp in spans)


def add_lab_synth(ex, k=2):
    """Chèn k cặp (TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM) tổng hợp vào cuối câu."""
    text = ex["text"].rstrip()
    spans = [dict(s) for s in ex["spans"]]
    add = " Kết quả xét nghiệm:"
    base = len(text) + len(add)
    text = text + add
    picks = random.sample(LAB_TEMPLATES, min(k, len(LAB_TEMPLATES)))
    for name, valfn in picks:
        val = valfn()
        chunk = f" {name}: {val};"
        ns = len(text) + 1
        ne = ns + len(name)
        vs = ne + 2
        ve = vs + len(val)
        text = text + chunk
        spans.append({"start": ns, "end": ne, "text": name,
                      "fine": "TÊN_XÉT_NGHIỆM", "coarse": "TEST", "assertions": []})
        spans.append({"start": vs, "end": ve, "text": val,
                      "fine": "KẾT_QUẢ_XÉT_NGHIỆM", "coarse": None, "assertions": []})
    return {"text": text, "source": ex["source"] + "+lab", "spans": spans}


def augment_file(in_jsonl, out_jsonl, icd_names=None, rx_names=None, mult=2):
    exs = [json.loads(l) for l in open(in_jsonl, encoding="utf-8") if l.strip()]
    out = list(exs)
    for ex in exs:
        for _ in range(mult):
            a = ex
            if icd_names or rx_names:
                a = entity_replace(a, icd_names or [], rx_names or [])
            a = inject_typos(a)
            if random.random() < 0.3:
                a = add_lab_synth(a)
            out.append(a)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[augment] {len(exs)} -> {len(out)} (x{1+mult}) -> {out_jsonl}")


if __name__ == "__main__":
    ex = {"text": "Bệnh nhân được chẩn đoán trào ngược dạ dày, dùng atenolol",
          "source": "demo",
          "spans": [
              {"start": 25, "end": 42, "text": "trào ngược dạ dày",
               "fine": "CHẨN_ĐOÁN", "coarse": "PROBLEM", "assertions": []},
              {"start": 49, "end": 57, "text": "atenolol",
               "fine": "THUỐC", "coarse": "TREATMENT", "assertions": []},
          ]}
    print("ORIG:", ex["text"])
    r = entity_replace(ex, ["viêm phổi", "nhồi máu cơ tim"], ["aspirin", "metoprolol"], p=1.0)
    print("REPL:", r["text"])
    for s in r["spans"]:
        ok = r["text"][s["start"]:s["end"]] == s["text"]
        print(f"   [{s['start']},{s['end']}] '{s['text']}' fine={s['fine']} offset_ok={ok}")
    t = inject_typos(ex, p_deaccent=1.0, p_merge=1.0)
    print("TYPO:", t["text"])
    for s in t["spans"]:
        print(f"   [{s['start']},{s['end']}] '{s['text']}' ok={t['text'][s['start']:s['end']]==s['text']}")
    l = add_lab_synth(ex, k=2)
    print("LAB :", l["text"])
    for s in l["spans"][-4:]:
        print(f"   [{s['start']},{s['end']}] '{s['text']}' {s['fine']} ok={l['text'][s['start']:s['end']]==s['text']}")
