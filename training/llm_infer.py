"""
llm_infer.py — Nạp Qwen2.5-7B base + LoRA adapter (Phase 2) và cấp:
  - make_llm_fn(): callable(prompt)->str  để tiêm vào rerank.rerank()
  - ner(text): generative-NER (role R1, second-opinion cho câu khó)
  - assertion(context, mention, ctype): role R2

Mặc định dùng transformers.generate (portable). Sản xuất: thay bằng vLLM
(+ LoRA) để nhanh hơn — interface make_llm_fn giữ nguyên.

Yêu cầu: transformers, peft, torch, accelerate (+ bitsandbytes nếu 4-bit).
"""
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class MedLLM:
    def __init__(self, base="Qwen/Qwen2.5-7B-Instruct", adapter=None,
                 load_4bit=True, max_new_tokens=256):
        quant = None
        if load_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=quant, device_map="auto",
            torch_dtype=torch.bfloat16)
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def chat(self, user, system=None, max_new_tokens=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            ids, max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=self.tok.eos_token_id)
        gen = out[0][ids.shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)

    # ---- llm_fn cho rerank.py ----
    def make_llm_fn(self):
        SYS = "Bạn là trợ lý mã hóa y khoa. Chỉ trả JSON list index, không giải thích."
        return lambda prompt: self.chat(prompt, system=SYS, max_new_tokens=32)

    # ---- role R1: generative NER ----
    def ner(self, text):
        SYS = "Bạn là trợ lý trích xuất khái niệm y khoa tiếng Việt."
        user = ("Trích xuất khái niệm (TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, "
                "KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC). Trả JSON list "
                '{"text","type"}.\n\n"' + text + '"')
        raw = self.chat(user, system=SYS, max_new_tokens=512)
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            return json.loads(m.group()) if m else []
        except Exception:
            return []

    # ---- role R2: assertion ----
    def assertion(self, context, mention, ctype):
        SYS = "Bạn xác định ngữ cảnh khái niệm y khoa."
        user = ('Chọn 0..3 nhãn trong [isNegated, isFamily, isHistorical]. Trả JSON list.\n\n'
                f'Câu: "{context}"\nKhái niệm: "{mention}" (loại: {ctype})')
        raw = self.chat(user, system=SYS, max_new_tokens=32)
        m = re.search(r"\[.*?\]", raw, re.S)
        try:
            out = json.loads(m.group()) if m else []
            valid = {"isNegated", "isFamily", "isHistorical"}
            return [a for a in out if a in valid][:3]
        except Exception:
            return []


if __name__ == "__main__":
    # Không tải model ở đây (cần GPU/mạng) — chỉ minh hoạ nối với rerank.
    print(__doc__)
    print(">> Ví dụ dùng thật:")
    print("""
    from llm_infer import MedLLM
    from rerank import rerank
    llm = MedLLM(base="Qwen/Qwen2.5-7B-Instruct", adapter="ckpt/qwen-medner-lora")
    llm_fn = llm.make_llm_fn()
    codes = rerank("trào ngược dạ dày-thực quản", "CHẨN_ĐOÁN", context,
                   [("K21.0","..."),("K21.9","...")], llm_fn)   # -> ⊆ ứng viên
    """)
