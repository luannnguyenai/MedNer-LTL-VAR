"""
infer_worker.py — Chạy pipeline MAX cho MỘT shard file trên MỘT GPU.
Được infer_parallel.py gọi (mỗi GPU 1 tiến trình, CUDA_VISIBLE_DEVICES đã set).

Ghép: encoder NER -> (LLM assertion) -> (linker ∪ retriever) -> rerank ràng buộc
-> validate schema -> ghi JSON.

Cờ --no_llm / --no_retriever để chạy nhẹ hơn (chỉ encoder+linker).
"""
import argparse
import json
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_stack(args):
    from predict import NERPredictor
    from linker import ICDLinker, RxNormLinker
    stack = {"ner": NERPredictor(args.ner_ckpt, args.encoder),
             "icd": ICDLinker(), "rx": RxNormLinker(),
             "llm": None, "llm_fn": None, "sidx": None}
    if not args.no_llm:
        from llm_infer import MedLLM
        stack["llm"] = MedLLM(args.llm_base, adapter=args.llm_adapter,
                              load_4bit=True)
        stack["llm_fn"] = stack["llm"].make_llm_fn()
    if not args.no_retriever:
        from retriever_infer import SemanticIndex
        s = SemanticIndex(args.retriever)
        s.build_icd(); s.build_rxnorm()
        stack["sidx"] = s
    return stack


def process_one(text, stack, args):
    from rerank import rerank
    try:
        from retriever_infer import augment_candidates
    except Exception:
        augment_candidates = None

    items = stack["ner"].predict(text)
    icd, rx = stack["icd"], stack["rx"]
    llm, llm_fn, sidx = stack["llm"], stack["llm_fn"], stack["sidx"]

    for it in items:
        t, ty = it["text"], it["type"]
        # assertion bằng LLM (override nháp encoder) nếu bật
        if llm is not None and ty in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
            a = llm.assertion(text, t, ty)
            if a:
                it["assertions"] = a
        # ứng viên: linker (∪ retriever) rồi rerank ràng buộc
        if ty == "CHẨN_ĐOÁN":
            cands = icd.link(t, topk=3) or []
            if sidx is not None and augment_candidates:
                cands = augment_candidates(cands, sidx.search(t, topk=5))
            if llm_fn is not None and len(cands) > 1:
                pairs = [(c, icd._name(c)) for c in cands]
                it["candidates"] = rerank(t, ty, text, pairs, llm_fn)
            else:
                it["candidates"] = cands[:3]
        elif ty == "THUỐC":
            cands = rx.link(t, topk=3) or []
            if sidx is not None and augment_candidates:
                cands = augment_candidates(cands, sidx.search(t, topk=5))
            if llm_fn is not None and len(cands) > 1:
                pairs = [(c, rx.by_rxcui.get(c, "")) for c in cands]
                it["candidates"] = rerank(t, ty, text, pairs, llm_fn)
            else:
                it["candidates"] = cands[:3]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ner_ckpt", default="ckpt/nerA-xlmr")
    ap.add_argument("--encoder", default="xlm-roberta-large")
    ap.add_argument("--llm_base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--llm_adapter", default="ckpt/qwen-medner-lora")
    ap.add_argument("--retriever", default="ckpt/bge-m3-medvn")
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--no_retriever", action="store_true")
    args = ap.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[worker gpu={gpu}] nạp model... ({len(args.files)} file)")
    stack = load_stack(args)

    for i, fp in enumerate(args.files):
        try:
            text = open(fp, encoding="utf-8").read()
            items = process_one(text, stack, args)
            name = os.path.splitext(os.path.basename(fp))[0] + ".json"
            with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"[worker gpu={gpu}] {i+1}/{len(args.files)} {name} ({len(items)} ent)")
        except Exception as e:
            print(f"[worker gpu={gpu}] LỖI {fp}: {e}")
    print(f"[worker gpu={gpu}] xong.")


if __name__ == "__main__":
    main()
