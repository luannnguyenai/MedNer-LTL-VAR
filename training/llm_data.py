"""
llm_data.py — Dựng dữ liệu FINE-TUNE (LoRA) cho LLM đa vai trò từ JSONL hợp nhất.

LLM (Qwen2.5-7B) được LoRA-tune đồng thời nhiều "role", mỗi role 1 template
instruction → output. Dùng CHUNG nguồn dữ liệu 5-type (silver + ViMedNER +
pseudo_fine + gold + augmented) nên KHÔNG cần dữ liệu mới.

Roles:
  R1 NER-instruct : text -> JSON [{text,type}]      (second-opinion / hard-case NER)
  R2 assertion    : (ngữ cảnh, mention, loại) -> assertions[]   (LLM mạnh suy luận này)
  R3 rerank       : (mention, ngữ cảnh, ứng viên) -> index đúng (dạy CHỌN, khớp rerank.py)
  R4 normalize    : mention viết tắt/nhiễu + ngữ cảnh -> dạng chuẩn (tùy chọn)

Xuất chat-format {"messages":[{role,content}...]} tương thích SFT (TRL/axolotl).
"""
import json
import random

random.seed(7)

SYS = "Bạn là trợ lý trích xuất & mã hóa khái niệm y khoa tiếng Việt. Trả lời đúng định dạng yêu cầu, không giải thích thừa."


def _msg(user, assistant):
    return {"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


# ---- R1: NER-instruct ------------------------------------------------------
def make_ner(ex):
    ents = [{"text": s["text"], "type": s["fine"]}
            for s in ex["spans"] if s.get("fine")]
    if not ents:
        return None
    user = ("Trích xuất các khái niệm y khoa (TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, "
            "KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC) từ đoạn sau. Trả JSON list "
            '{"text","type"}.\n\n"' + ex["text"] + '"')
    return _msg(user, json.dumps(ents, ensure_ascii=False))


# ---- R2: assertion ---------------------------------------------------------
ASSERTABLE = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}


def make_assertion(ex):
    out = []
    for s in ex["spans"]:
        if s.get("fine") not in ASSERTABLE:
            continue
        user = ('Xác định ngữ cảnh của khái niệm trong câu. Chọn 0..3 nhãn trong '
                '[isNegated, isFamily, isHistorical]. Trả JSON list.\n\n'
                f'Câu: "{ex["text"]}"\nKhái niệm: "{s["text"]}" (loại: {s["fine"]})')
        out.append(_msg(user, json.dumps(s.get("assertions", []),
                                         ensure_ascii=False)))
    return out


# ---- R3: rerank (cần linker để sinh ứng viên) ------------------------------
def make_rerank(ex, icd=None, rx=None, topk=6):
    """Với mỗi span CHẨN_ĐOÁN/THUỐC có gold code, tạo ví dụ CHỌN index đúng.
    gold code lấy từ span['candidates'] (nếu dữ liệu có) HOẶC bỏ qua."""
    out = []
    for s in ex["spans"]:
        gold = s.get("candidates") or []
        if s.get("fine") == "CHẨN_ĐOÁN" and icd is not None:
            cand_codes = icd.link(s["text"], topk=topk) or []
            names = {c: icd._name(c) for c in cand_codes}
        elif s.get("fine") == "THUỐC" and rx is not None:
            cand_codes = rx.link(s["text"], topk=topk) or []
            names = {c: rx.by_rxcui.get(c, "") for c in cand_codes}
        else:
            continue
        if not gold or not cand_codes:
            continue
        # trộn thêm gold vào ứng viên nếu thiếu (để có nhãn dương)
        for g in gold:
            if g not in cand_codes:
                cand_codes.append(g)
                names[g] = names.get(g, "")
        random.shuffle(cand_codes)
        cand_pairs = [(c, names.get(c, "")) for c in cand_codes]
        gold_idx = [i for i, (c, _) in enumerate(cand_pairs) if c in set(gold)]
        block = "\n".join(f"[{i}] {c} - {n}" for i, (c, n) in enumerate(cand_pairs))
        user = (f'Chọn (các) mã đúng nhất cho cụm "{s["text"]}" (loại {s["fine"]}).'
                f'\nNgữ cảnh: "{ex["text"][:200]}"\nỨng viên:\n{block}\n'
                'Trả JSON list index.')
        out.append(_msg(user, json.dumps(gold_idx)))
    return out


def build(in_jsonl, out_jsonl, roles=("ner", "assertion"), icd=None, rx=None):
    rows = [json.loads(l) for l in open(in_jsonl, encoding="utf-8") if l.strip()]
    ex_out = []
    for r in rows:
        if "ner" in roles:
            m = make_ner(r)
            if m:
                ex_out.append(m)
        if "assertion" in roles:
            ex_out += make_assertion(r)
        if "rerank" in roles:
            ex_out += make_rerank(r, icd, rx)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for e in ex_out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[llm_data] {len(rows)} câu -> {len(ex_out)} ví dụ SFT ({roles}) -> {out_jsonl}")


if __name__ == "__main__":
    ex = {"text": "Bệnh nhân không sốt, được chẩn đoán viêm phổi, tiền sử dùng aspirin",
          "source": "demo",
          "spans": [
              {"text": "sốt", "fine": "TRIỆU_CHỨNG", "coarse": "PROBLEM",
               "assertions": ["isNegated"]},
              {"text": "viêm phổi", "fine": "CHẨN_ĐOÁN", "coarse": "PROBLEM",
               "assertions": [], "candidates": ["J18.9"]},
              {"text": "aspirin", "fine": "THUỐC", "coarse": "TREATMENT",
               "assertions": ["isHistorical"], "candidates": ["1191"]},
          ]}
    print("=== R1 NER-instruct ===")
    print(json.dumps(make_ner(ex), ensure_ascii=False, indent=2)[:400], "...")
    print("\n=== R2 assertion (3 ví dụ) ===")
    for m in make_assertion(ex):
        u = m["messages"][1]["content"].split("Khái niệm:")[1].strip()
        print(f"  {u}  -> {m['messages'][2]['content']}")
    print("\n=== R3 rerank (mock ứng viên, không cần linker) ===")

    class MockICD:
        def link(self, t, topk=6): return ["J18.9", "J15.9", "J12.9"]
        def _name(self, c): return {"J18.9": "Viêm phổi kxđ",
                                    "J15.9": "Viêm phổi vi khuẩn kxđ",
                                    "J12.9": "Viêm phổi virus kxđ"}.get(c, "")
    for m in make_rerank(ex, icd=MockICD()):
        print("  USER:", m["messages"][1]["content"].replace("\n", " ")[:150], "...")
        print("  GOLD index:", m["messages"][2]["content"])
    print("\nĐịnh dạng SFT chat OK — dùng chung nguồn 5-type, không cần data mới")
