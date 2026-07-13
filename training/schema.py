"""
schema.py — Không gian nhãn hợp nhất + BẢNG ÁNH XẠ giữa các bộ dữ liệu.

Vấn đề cốt lõi: mỗi dataset public có schema KHÁC nhau và thô hơn 5-type của đề.
Đặc biệt PhoNER / i2b2 gộp "SYMPTOM_AND_DISEASE" / "problem" -> KHÔNG tách sẵn
TRIỆU_CHỨNG vs CHẨN_ĐOÁN. Giải pháp: HAI đầu (head) học đa nhiệm chung encoder:

  HEAD A (TARGET, 5 lớp)  — sinh output cuộc thi. Chỉ train trên dữ liệu có phân
      biệt 5 chiều: silver (rule pipeline), ViMedNER, gold người gán, i2b2 test.
  HEAD B (AUX, thô)       — train trên MỌI nguồn với nhãn thô {PROBLEM, TEST,
      TREATMENT, OTHER}. Dạy encoder nhận BIÊN thực thể + biểu diễn miền y khoa.

Chuyển thô->tinh cho span PROBLEM: dùng ICD-linker (đã có) — link được ICD mạnh
  => CHẨN_ĐOÁN, ngược lại => TRIỆU_CHỨNG (pseudo-label cho HEAD A).
"""

# ---- 5 nhãn cuộc thi (HEAD A) ---------------------------------------------
TYPES = ["TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
         "CHẨN_ĐOÁN", "THUỐC"]

# ---- nhãn thô (HEAD B) -----------------------------------------------------
COARSE = ["PROBLEM", "TEST", "TREATMENT", "OTHER"]

# ---- assertion (HEAD C, multi-label nhị phân trên token thực thể) ----------
ASSERTIONS = ["isNegated", "isFamily", "isHistorical"]


def bio_labels(types):
    out = ["O"]
    for t in types:
        out += [f"B-{t}", f"I-{t}"]
    return out


BIO_FINE = bio_labels(TYPES)      # HEAD A: 11 nhãn
BIO_COARSE = bio_labels(COARSE)   # HEAD B: 9 nhãn
FINE2ID = {l: i for i, l in enumerate(BIO_FINE)}
COARSE2ID = {l: i for i, l in enumerate(BIO_COARSE)}

# ===========================================================================
# BẢNG ÁNH XẠ TỪ SCHEMA GỐC -> (fine 5-type | None), (coarse | None)
# None nghĩa là bỏ khỏi head đó.
# ===========================================================================

# PhoNER_COVID19 (10 loại) — chỉ phần y khoa hữu ích
PHONER_MAP = {
    "SYMPTOM_AND_DISEASE": {"fine": None,       "coarse": "PROBLEM"},
    # các loại phi khái-niệm-y-khoa: dùng làm OTHER cho HEAD B (học biên/ngữ cảnh)
    "PATIENT_ID":  {"fine": None, "coarse": "OTHER"},
    "PERSON_NAME": {"fine": None, "coarse": "OTHER"},
    "AGE":         {"fine": None, "coarse": "OTHER"},
    "GENDER":      {"fine": None, "coarse": "OTHER"},
    "OCCUPATION":  {"fine": None, "coarse": "OTHER"},
    "LOCATION":    {"fine": None, "coarse": "OTHER"},
    "ORGANIZATION":{"fine": None, "coarse": "OTHER"},
    "TRANSPORTATION": {"fine": None, "coarse": "OTHER"},
    "DATE":        {"fine": None, "coarse": "OTHER"},
}

# ViMedNER (5 loại) — GẦN với đề nhất
VIMEDNER_MAP = {
    "DISEASE_NAME": {"fine": "CHẨN_ĐOÁN",       "coarse": "PROBLEM"},
    "SYMPTOM":      {"fine": "TRIỆU_CHỨNG",      "coarse": "PROBLEM"},
    "DIAGNOSTIC":   {"fine": "TÊN_XÉT_NGHIỆM",   "coarse": "TEST"},
    "TREATMENT":    {"fine": None,               "coarse": "TREATMENT"},  # treatment≠thuốc chắc chắn -> để linker/regex lọc thuốc
    "CAUSE":        {"fine": None,               "coarse": None},
}

# VietBioNER (5 loại)
VIETBIONER_MAP = {
    "SYMPTOM_AND_DISEASE": {"fine": None,             "coarse": "PROBLEM"},
    "DIAGNOSTIC_PROCEDURE":{"fine": "TÊN_XÉT_NGHIỆM", "coarse": "TEST"},
    "ORGANISATION": {"fine": None, "coarse": "OTHER"},
    "LOCATION":     {"fine": None, "coarse": "OTHER"},
    "DATE_AND_TIME":{"fine": None, "coarse": "OTHER"},
}

# ViMQ (medical question) — thực thể chủ yếu là triệu chứng/bệnh
VIMQ_MAP = {
    "SYMPTOM_AND_DISEASE": {"fine": None,       "coarse": "PROBLEM"},
    "MEDICAL_ENTITY":      {"fine": None,       "coarse": "PROBLEM"},
}

# i2b2 2010 (dịch sang tiếng Việt) — problem/test/treatment + assertion
I2B2_MAP = {
    "problem":   {"fine": None,             "coarse": "PROBLEM"},
    "test":      {"fine": "TÊN_XÉT_NGHIỆM", "coarse": "TEST"},
    "treatment": {"fine": None,             "coarse": "TREATMENT"},
}
# assertion i2b2 -> assertion đề bài
I2B2_ASSERT_MAP = {
    "absent":                        "isNegated",
    "associated_with_someone_else":  "isFamily",
    # present/possible/conditional/hypothetical -> không map cứng;
    # isHistorical lấy từ i2b2-2012 temporal HOẶC section-cue (xem README_TRAIN §4)
}

SOURCE_MAPS = {
    "phoner": PHONER_MAP, "vimedner": VIMEDNER_MAP,
    "vietbioner": VIETBIONER_MAP, "vimq": VIMQ_MAP, "i2b2": I2B2_MAP,
}


def map_entity(source: str, label: str, which: str):
    """label gốc -> nhãn (fine|coarse) theo nguồn. Trả None nếu bỏ."""
    m = SOURCE_MAPS.get(source, {})
    ent = m.get(label.upper() if source != "i2b2" else label)
    if not ent:
        return None
    return ent.get(which)


if __name__ == "__main__":
    print("HEAD A (fine):", BIO_FINE)
    print("HEAD B (coarse):", BIO_COARSE)
    print("assertions:", ASSERTIONS)
    print()
    for src in SOURCE_MAPS:
        print(f"[{src}]")
        for lbl in SOURCE_MAPS[src]:
            f = map_entity(src, lbl, "fine")
            c = map_entity(src, lbl, "coarse")
            print(f"   {lbl:24s} fine={f}  coarse={c}")
