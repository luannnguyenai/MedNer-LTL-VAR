"""
assertions.py — Suy luận ngữ cảnh: isNegated / isFamily / isHistorical.

Chỉ áp dụng cho CHẨN_ĐOÁN / THUỐC / TRIỆU_CHỨNG (theo đề bài).

Cách tiếp cận (rule-based, precision-oriented — dùng cho baseline & silver label;
bản production thay bằng classifier fine-tuned, xem ARCHITECTURE.md):
  - Xét cửa sổ trái (left context) trước span + nhãn "section" (mục 1/2/3 của bệnh án).
  - Có danh sách CUE + danh sách PSEUDO (loại trừ) để tránh dương tính giả.
"""
import re
from vntext import strip_accents

# ---- NEGATION -------------------------------------------------------------
NEG_CUES = [
    "khong", "chua", "khong co", "khong ghi nhan", "khong phat hien",
    "khong thay", "khong con", "loai tru", "am tinh", "phu nhan",
    "khong bi", "khong dau", "khong kem", "khong xuat hien",
]
# "không xác định", "không đặc hiệu"... là bổ nghĩa, KHÔNG phải phủ định khái niệm
NEG_PSEUDO_NEXT = {
    "xac", "dac", "ro", "do", "dang", "doi", "the", "lien", "on",
    "on dinh", "can", "kiem", "tu chu", "duoc",  # "không tự chủ", "không được"
}

# ---- FAMILY ---------------------------------------------------------------
# Match theo RANH GIỚI TỪ (\b) để tránh "ong" trong "khong". Precision-oriented:
# chỉ nhận khi có người thân RÕ RÀNG trong ngữ cảnh trái gần span.
FAMILY_WORDS = [
    r"gia dinh", r"tien su gia dinh", r"di truyen", r"ho hang",
    r"bo(?: benh nhan| ruot|,| )", r"me(?: benh nhan| ruot|,| )",
    r"cha(?: benh nhan| ruot|,| )", r"anh (?:trai|ruot)", r"chi (?:gai|ruot)",
    r"em (?:trai|gai|ruot)", r"ong (?:noi|ngoai)", r"ba (?:noi|ngoai)",
    r"con (?:trai|gai) benh nhan", r"nguoi than trong gia dinh",
]
FAMILY_RE = re.compile(r"\b(" + "|".join(FAMILY_WORDS) + r")")

# ---- HISTORICAL -----------------------------------------------------------
HIST_CUES = [
    "tien su", "tien can", "truoc day", "da tung", "cach day", "cach ",
    "man tinh", "mang tinh", "truoc do", " cu", "benh ly nen", "benh nen",
    "da phau thuat", "da dieu tri", "tu nho", "nhieu nam",
]
# Section headers => historical (past medical/surgical history)
HIST_SECTION_KEYS = [
    "tien su benh noi khoa", "tien su benh ly", "tien su benh",
    "tien su phau thuat", "cac benh ly man tinh", "cac benh ly nen",
    "benh su", "tien su san khoa",
]
# NHƯNG "tiền sử bệnh hiện tại" = HPI = hiện tại (không historical)
HIST_SECTION_EXCLUDE = ["hien tai", "benh su hien tai", "tien su benh hien tai"]


def _left_window(text, start, n=60):
    return strip_accents(text[max(0, start - n):start].lower())


def _clause_left(text, start, n=120):
    """Ngữ cảnh trái tới dấu ngắt câu gần nhất (. ; xuống dòng)."""
    seg = text[max(0, start - n):start]
    # cắt tại dấu chấm/; hoặc newline cuối cùng
    m = list(re.finditer(r"[.;\n]", seg))
    if m:
        seg = seg[m[-1].end():]
    return strip_accents(seg.lower())


def detect_negation(text, start, end):
    left = _clause_left(text, start)
    la = strip_accents(text[max(0, start - 30):end].lower())
    for cue in NEG_CUES:
        # cue xuất hiện ở cuối ngữ cảnh trái (gần span)
        idx = left.rfind(cue)
        if idx == -1:
            continue
        after = left[idx + len(cue):].strip()
        first = after.split()[:2]
        firstw = " ".join(first)
        # loại trừ pseudo-negation
        if first and (first[0] in NEG_PSEUDO_NEXT or firstw in NEG_PSEUDO_NEXT):
            continue
        # cue phải khá gần span (trong ~40 ký tự) để đúng scope
        if len(left) - idx <= 45:
            return True
    return False


def detect_family(text, start, end):
    # xét cả cụm chứa span (người thân có thể đứng ngay trước bệnh)
    ctx = strip_accents(text[max(0, start - 70):end].lower())
    return bool(FAMILY_RE.search(ctx))


def detect_historical(text, start, end, section_label=""):
    sec = strip_accents((section_label or "").lower())
    # section-based
    if sec and not any(x in sec for x in HIST_SECTION_EXCLUDE):
        if any(k in sec for k in HIST_SECTION_KEYS):
            return True
    left = _clause_left(text, start, 140)
    # inline cue "cũ" thường đứng NGAY SAU khái niệm ("nhồi máu cơ tim cũ")
    right = strip_accents(text[end:end + 8].lower())
    if right.strip().startswith("cu"):
        return True
    for cue in HIST_CUES:
        if cue in left:
            return True
    return False


def assertions_for(text, start, end, section_label="", ctype=None):
    """Trả list assertions (<=3) cho span, theo loại khái niệm."""
    if ctype not in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
        return []
    out = []
    if detect_negation(text, start, end):
        out.append("isNegated")
    if detect_family(text, start, end):
        out.append("isFamily")
    if detect_historical(text, start, end, section_label):
        out.append("isHistorical")
    return out[:3]


if __name__ == "__main__":
    samples = [
        ("Bệnh nhân không ho, không sốt.", "ho"),
        ("Không buồn nôn, hay nôn, đổ mồ hôi", "nôn"),
        ("Có tiền sử hen suyễn nhiều năm.", "hen suyễn"),
        ("Nhồi máu cơ tim cũ trên điện tâm đồ", "Nhồi máu cơ tim"),
        ("bệnh thận mạn, không đặc hiệu", "bệnh thận mạn"),
        ("Bố bệnh nhân bị đái tháo đường", "đái tháo đường"),
    ]
    for txt, term in samples:
        s = txt.find(term); e = s + len(term)
        print(f"{txt!r}")
        print(f"   [{term}] neg={detect_negation(txt,s,e)} "
              f"fam={detect_family(txt,s,e)} hist={detect_historical(txt,s,e)}")
