"""
rerank.py — Rerank ứng viên bằng LLM có RÀNG BUỘC (cổng chống hallucination).

Vai trò: sau khi linker (+retriever) trả TẬP ứng viên mã (ICD/RxNorm), LLM CHỌN
tập con đúng nhất dựa trên mention + ngữ cảnh. NHƯNG:
  - LLM chỉ được trả về INDEX trong danh sách ứng viên (constrained decoding / grammar).
  - Parser ép: mọi mã xuất ra PHẢI thuộc tập ứng viên (giao với KB). Bất kỳ mã "lạ"
    nào (LLM bịa) đều bị LOẠI. => giữ cam kết EXACT tuyệt đối dù có LLM trong vòng.

`llm_fn` được tiêm vào (dependency injection) nên test được bằng LLM giả, và khi
chạy thật chỉ cần bọc vLLM/transformers.generate.
"""
import json
import re


PROMPT = """Bạn là trợ lý mã hóa y khoa. Cho một cụm khái niệm và ngữ cảnh, hãy chọn (các) mã ĐÚNG NHẤT trong DANH SÁCH ứng viên bên dưới.
QUY TẮC:
- CHỈ chọn từ danh sách. Nếu không mã nào phù hợp, trả về [].
- Trả về DUY NHẤT một JSON list các số thứ tự (index) đã chọn, không giải thích.

Cụm: "{mention}"  (loại: {ctype})
Ngữ cảnh: "{context}"

Ứng viên:
{cand_block}

JSON index đã chọn:"""


def build_prompt(mention, ctype, context, candidates):
    """candidates: list (code, name). Trả prompt + map index->code."""
    lines = []
    for i, (code, name) in enumerate(candidates):
        nm = (name or "").strip()
        lines.append(f"[{i}] {code} - {nm}" if nm else f"[{i}] {code}")
    block = "\n".join(lines) if lines else "(không có ứng viên)"
    prompt = PROMPT.format(mention=mention, ctype=ctype,
                           context=context[:300], cand_block=block)
    return prompt


def parse_indices(text, n):
    """Trích list index hợp lệ (0..n-1) từ output LLM, bỏ mọi thứ khác."""
    # tìm đoạn JSON list đầu tiên
    m = re.search(r"\[[^\]]*\]", text)
    idxs = []
    if m:
        try:
            arr = json.loads(m.group())
            for x in arr:
                if isinstance(x, int) and 0 <= x < n:
                    idxs.append(x)
        except Exception:
            pass
    if not idxs:  # fallback: quét số rời
        for tok in re.findall(r"-?\d+", text):
            v = int(tok)
            if 0 <= v < n:
                idxs.append(v)
    # khử trùng giữ thứ tự
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i); out.append(i)
    return out


def rerank(mention, ctype, context, candidates, llm_fn, topk=3):
    """
    candidates: list (code, name)  — TẬP ứng viên từ KB (linker/retriever).
    llm_fn: callable(prompt:str)->str  (LLM sinh; tiêm vào để test/serve).
    Trả list code ĐÃ CHỌN, đảm bảo ⊆ candidates (không hallucination).
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return [candidates[0][0]]
    prompt = build_prompt(mention, ctype, context, candidates)
    raw = llm_fn(prompt)
    idxs = parse_indices(raw, len(candidates))
    codes = [candidates[i][0] for i in idxs][:topk]
    # CỔNG CUỐI: chỉ giữ code thực sự nằm trong tập ứng viên
    allowed = {c for c, _ in candidates}
    return [c for c in codes if c in allowed]


if __name__ == "__main__":
    cands = [("K21.0", "Bệnh trào ngược dạ dày-thực quản có viêm thực quản"),
             ("K21.9", "Bệnh trào ngược dạ dày-thực quản không viêm thực quản"),
             ("K20", "Viêm thực quản")]

    # LLM giả 1: chọn đúng [0,1]
    def good_llm(p):
        return "Tôi chọn: [0, 1]"
    print("good ->", rerank("trào ngược dạ dày-thực quản", "CHẨN_ĐOÁN",
                            "bệnh nhân ợ hơi, nóng rát sau xương ức", cands, good_llm))

    # LLM giả 2: cố bịa mã ngoài danh sách + index sai
    def rogue_llm(p):
        return 'Chọn mã K99.9 và [5], [1]'   # K99.9 không trong list, 5 out-of-range
    print("rogue ->", rerank("trào ngược dạ dày-thực quản", "CHẨN_ĐOÁN",
                             "ợ hơi", cands, rogue_llm),
          "(mã bịa K99.9 & index 5 bị loại; chỉ còn 1)")

    # LLM giả 3: trả rỗng (không phù hợp)
    def empty_llm(p):
        return "[]"
    print("empty ->", rerank("abcxyz", "CHẨN_ĐOÁN", "", cands, empty_llm))

    # 1 ứng viên -> khỏi gọi LLM
    print("single ->", rerank("aspirin", "THUỐC", "", [("1191", "aspirin")], good_llm))
    print("\nCổng chống hallucination OK — output luôn ⊆ tập ứng viên KB")
