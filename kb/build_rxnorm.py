"""
build_rxnorm.py — Xây lexicon RxNorm từ RXNCONSO.RRF.

Định dạng RRF (|-separated), cột quan trọng:
  0 RXCUI | 1 LAT | 11 SAB | 12 TTY | 14 STR

Ta CHỈ giữ SAB == 'RXNORM' (nguồn chuẩn, tránh nhiễu từ MTHSPL trùng lặp),
và các TTY hữu ích cho ánh xạ thuốc lâm sàng. Với mỗi tên (STR) lưu RXCUI + TTY
để linker có thể ưu tiên loại phù hợp.

Ưu tiên TTY (cao -> thấp) khi phải chọn 1 mã đại diện:
  SCD  Semantic Clinical Drug        (ingredient+strength+form, "prescribable")
  SBD  Semantic Branded Drug
  SCDC Semantic Clinical Drug Comp   (ingredient+strength) -> khớp "X 0.4 MG/ML"
  SBDC
  GPCK/BPCK  packs
  SCDF/SBDF  drug+form
  IN   Ingredient
  PIN  Precise Ingredient
  MIN  Multiple Ingredients
  BN   Brand Name
  PSN/SY/TMSY  tên hiển thị/đồng nghĩa

Đầu ra: kb/rxnorm_index.json
  entries : [{rxcui, str, key, tty}]
  exact   : {norm_key -> [rxcui...]} (giữ thứ tự xuất hiện, ưu tiên khử trùng)
  by_rxcui: {rxcui -> preferred_name}
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vntext import norm_key  # noqa

KEEP_TTY = {
    "SCD", "SBD", "SCDC", "SBDC", "GPCK", "BPCK", "SCDF", "SBDF",
    "SCDG", "SBDG", "IN", "PIN", "MIN", "BN", "PSN", "SY", "TMSY",
    "SCDGP", "SBDFP", "SCDFP",
}
TTY_RANK = {t: i for i, t in enumerate([
    "SCD", "SBD", "SCDC", "SBDC", "GPCK", "BPCK", "SCDF", "SBDF",
    "SCDG", "SBDG", "MIN", "IN", "PIN", "BN", "PSN", "TMSY", "SY",
    "SCDGP", "SBDFP", "SCDFP",
])}


def build(rrf_path: str, out_path: str, keep_sab=("RXNORM",)):
    entries = []
    exact = {}
    by_rxcui = {}
    best_tty = {}  # rxcui -> rank of best tty seen (for preferred name)

    with open(rrf_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) < 15:
                continue
            rxcui, lat, sab, tty, s = p[0], p[1], p[11], p[12], p[14]
            if lat != "ENG":
                continue
            if keep_sab and sab not in keep_sab:
                continue
            if tty not in KEEP_TTY:
                continue
            if not s:
                continue
            k = norm_key(s)
            if not k:
                continue
            entries.append({"rxcui": rxcui, "str": s, "key": k, "tty": tty})
            exact.setdefault(k, [])
            if rxcui not in exact[k]:
                exact[k].append(rxcui)
            # preferred name = tên có TTY xếp hạng cao nhất
            rank = TTY_RANK.get(tty, 999)
            if rxcui not in best_tty or rank < best_tty[rxcui]:
                best_tty[rxcui] = rank
                by_rxcui[rxcui] = s

    out = {"entries": entries, "exact": exact, "by_rxcui": by_rxcui}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[RxNorm] entries={len(entries)}  exact_keys={len(exact)}  "
          f"rxcuis={len(by_rxcui)}  -> {out_path}")


if __name__ == "__main__":
    rrf = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/RXNCONSO.RRF"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "rxnorm_index.json")
    build(rrf, out)
