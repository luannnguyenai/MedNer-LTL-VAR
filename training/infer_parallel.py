"""
infer_parallel.py — Chia 100 file thành N shard, chạy SONG SONG trên N GPU.
Mỗi GPU 1 tiến trình worker (infer_worker.py) với CUDA_VISIBLE_DEVICES riêng.

Với 8×A100: 100 file / 8 ≈ 13 file/GPU -> tất cả xong gần như đồng thời.

Dùng:
  python infer_parallel.py --input ../input --out_dir ../out_max --gpus 0,1,2,3,4,5,6,7
  # nhẹ (chỉ encoder+linker, không LLM/retriever):
  python infer_parallel.py --input ../input --out_dir ../out_fast --gpus 0,1,2,3,4,5,6,7 --no_llm --no_retriever
"""
import argparse
import glob
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="thư mục chứa file .txt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--ner_ckpt", default="ckpt/nerA-xlmr")
    ap.add_argument("--encoder", default="xlm-roberta-large")
    ap.add_argument("--llm_base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--llm_adapter", default="ckpt/qwen-medner-lora")
    ap.add_argument("--retriever", default="ckpt/bge-m3-medvn")
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--no_retriever", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(args.input, "*")))
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n = len(gpus)
    os.makedirs(args.out_dir, exist_ok=True)

    # chia đều (round-robin để cân tải nếu file dài ngắn khác nhau)
    shards = [[] for _ in range(n)]
    for i, f in enumerate(files):
        shards[i % n].append(f)

    print(f"[parallel] {len(files)} file -> {n} GPU ({args.gpus}); "
          f"~{len(files)//n}-{-(-len(files)//n)} file/GPU")

    procs = []
    t0 = time.time()
    worker = os.path.join(os.path.dirname(__file__), "infer_worker.py")
    for gi, gpu in enumerate(gpus):
        if not shards[gi]:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [sys.executable, worker,
               "--files", *shards[gi],
               "--out_dir", args.out_dir,
               "--ner_ckpt", args.ner_ckpt, "--encoder", args.encoder,
               "--llm_base", args.llm_base, "--llm_adapter", args.llm_adapter,
               "--retriever", args.retriever]
        if args.no_llm:
            cmd.append("--no_llm")
        if args.no_retriever:
            cmd.append("--no_retriever")
        procs.append(subprocess.Popen(cmd, env=env))

    codes = [p.wait() for p in procs]
    dt = time.time() - t0
    ok = sum(1 for c in codes if c == 0)
    done = len(glob.glob(os.path.join(args.out_dir, "*.json")))
    print(f"[parallel] xong {ok}/{len(procs)} worker OK, "
          f"{done} file JSON, {dt:.1f}s tổng")


if __name__ == "__main__":
    main()
