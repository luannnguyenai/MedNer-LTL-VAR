"""
adapters.py — Chuyển các dataset public (định dạng CoNLL / spans-JSON) về
ĐỊNH DẠNG HỢP NHẤT (JSONL) có CHAR OFFSET, áp bảng ánh xạ trong schema.py.

Định dạng hợp nhất (mỗi dòng 1 câu):
{
  "text": "...",                       # surface text (đã dựng lại từ token)
  "source": "phoner",
  "spans": [
     {"start": int, "end": int, "text": "...",
      "fine": "CHẨN_ĐOÁN"|null, "coarse":"PROBLEM"|null,
      "assertions": ["isNegated", ...]}    # nếu nguồn có
  ]
}

Vì lưu CHAR OFFSET nên độc lập tokenizer: lúc train/infer ta re-tokenize bằng
fast-tokenizer của model và căn theo offset_mapping -> giữ đúng vị trí ký tự
mà đề bài yêu cầu.

Ghi chú word-segmentation: PhoNER/ViHealthBERT dùng token nối bằng '_'. Ta dựng
surface text bằng cách thay '_'->' ' (xấp xỉ raw tiếng Việt) và tính offset trên
surface đó. Model học phát hiện span trên surface; lúc suy luận offset lấy từ
offset_mapping trên RAW input thật.
"""
import json
import re
from schema import map_entity


def read_conll(path, sep=None):
    """Đọc file CoNLL: mỗi dòng 'token<sep>TAG', dòng trống = hết câu.
    sep=None -> tách theo khoảng trắng/tab."""
    sents = []
    toks, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if toks:
                    sents.append((toks, tags))
                    toks, tags = [], []
                continue
            parts = line.split(sep) if sep else line.split()
            if len(parts) < 2:
                continue
            toks.append(parts[0])
            tags.append(parts[-1])          # cột cuối là nhãn BIO
    if toks:
        sents.append((toks, tags))
    return sents


def _surface(tok):
    return tok.replace("_", " ")


def conll_to_unified(tokens, tags, source):
    """Dựng surface text + char span, áp ánh xạ fine/coarse theo nguồn."""
    text_parts = []
    offsets = []           # (start,end) của từng token trên surface text
    pos = 0
    for i, tok in enumerate(tokens):
        s = _surface(tok)
        if i > 0:
            text_parts.append(" ")
            pos += 1
        start = pos
        text_parts.append(s)
        pos += len(s)
        offsets.append((start, pos))
    text = "".join(text_parts)

    # gom BIO -> span (theo nhãn gốc)
    spans = []
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        if tag == "O" or tag == "0":
            i += 1
            continue
        m = re.match(r"^([BI])-(.+)$", tag)
        if not m:
            i += 1
            continue
        lbl = m.group(2)
        j = i + 1
        while j < n and re.match(rf"^I-{re.escape(lbl)}$", tags[j]):
            j += 1
        s_char = offsets[i][0]
        e_char = offsets[j - 1][1]
        fine = map_entity(source, lbl, "fine")
        coarse = map_entity(source, lbl, "coarse")
        if fine is not None or coarse is not None:
            spans.append({"start": s_char, "end": e_char,
                          "text": text[s_char:e_char],
                          "fine": fine, "coarse": coarse, "assertions": []})
        i = j
    return {"text": text, "source": source, "spans": spans}


def load_spans_json(path, source, key_text="text", key_spans="entities",
                    k_start="start", k_end="end", k_label="label",
                    k_assert=None):
    """Nạp dataset đã ở dạng char-span JSON (mỗi dòng/obj 1 câu)."""
    out = []
    with open(path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()] if path.endswith(
            ".jsonl") else json.load(f)
    for obj in data:
        text = obj[key_text]
        spans = []
        for e in obj.get(key_spans, []):
            lbl = e[k_label]
            fine = map_entity(source, lbl, "fine")
            coarse = map_entity(source, lbl, "coarse")
            if fine is None and coarse is None:
                continue
            asrt = []
            if k_assert and e.get(k_assert):
                a = e[k_assert]
                asrt = [a] if isinstance(a, str) else list(a)
            spans.append({"start": e[k_start], "end": e[k_end],
                          "text": text[e[k_start]:e[k_end]],
                          "fine": fine, "coarse": coarse, "assertions": asrt})
        out.append({"text": text, "source": source, "spans": spans})
    return out


def write_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[adapters] wrote {len(examples)} -> {path}")


def convert_phoner(conll_path, out_path):
    sents = read_conll(conll_path)
    ex = [conll_to_unified(t, g, "phoner") for t, g in sents]
    write_jsonl(ex, out_path)


def convert_vimedner(conll_path, out_path):
    sents = read_conll(conll_path)
    ex = [conll_to_unified(t, g, "vimedner") for t, g in sents]
    write_jsonl(ex, out_path)


if __name__ == "__main__":
    # SMOKE TEST bằng câu CoNLL tổng hợp (không cần tải dataset)
    demo = """Bệnh_nhân O
ho B-SYMPTOM
đờm I-SYMPTOM
xanh I-SYMPTOM
, O
được O
chẩn_đoán O
trào_ngược B-DISEASE_NAME
dạ_dày I-DISEASE_NAME
- I-DISEASE_NAME
thực_quản I-DISEASE_NAME
. O
"""
    open("/tmp/demo.conll", "w", encoding="utf-8").write(demo)
    sents = read_conll("/tmp/demo.conll")
    for t, g in sents:
        u = conll_to_unified(t, g, "vimedner")
        print("TEXT:", u["text"])
        for s in u["spans"]:
            assert u["text"][s["start"]:s["end"]] == s["text"], "offset sai!"
            print(f"   [{s['start']:>3},{s['end']:>3}] fine={s['fine']:<12} "
                  f"coarse={s['coarse']:<8} '{s['text']}'")
    print("offset khớp text — OK")
