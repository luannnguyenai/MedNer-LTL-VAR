"""
vntext.py — Chuẩn hóa văn bản tiếng Việt (và thuật ngữ y khoa Anh–Việt).

Mục tiêu: tạo các "khóa" (keys) ổn định để index & so khớp KB, đồng thời
giữ nguyên văn bản gốc cho tính toán `position` theo ký tự.

KHÔNG phụ thuộc thư viện ngoài — chạy được offline hoàn toàn.
"""
import re
import unicodedata

# ---- Bảng gỡ dấu tiếng Việt (đầy đủ, kể cả đ/Đ) --------------------------
_VN_MAP = str.maketrans(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD",
)


def strip_accents(s: str) -> str:
    """Gỡ toàn bộ dấu tiếng Việt -> ASCII thường dùng để so khớp mờ."""
    return s.translate(_VN_MAP)


# Khoảng trắng thừa, ký tự lạ
_WS = re.compile(r"\s+")
# Ký tự giữ lại khi tạo khóa so khớp (chữ, số, vài dấu ngăn cách)
_NONKEY = re.compile(r"[^a-z0-9%/+.\- ]")


def clean_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


# Chuẩn hóa spacing hàm lượng: "325mg" -> "325 mg", "0.4mg/ml" -> "0.4 mg/ml"
_STRENGTH_SPACE = re.compile(r"(\d)\s*(mg/ml|mcg/ml|mg/actuat|mg|mcg|ml|iu|unit|%)\b")


def norm_key(s: str) -> str:
    """
    Khóa so khớp CHÍNH: lowercase + gỡ dấu + bỏ ký tự lạ + gộp khoảng trắng.
    Dùng cho cả tên bệnh (VN/EN) và tên thuốc.
    """
    s = s.lower()
    s = strip_accents(s)
    s = s.replace("–", "-").replace("—", "-").replace("_", " ")
    s = _NONKEY.sub(" ", s)
    s = _STRENGTH_SPACE.sub(r"\1 \2", s)   # tách số và đơn vị hàm lượng
    s = _WS.sub(" ", s).strip()
    return s


def norm_key_loose(s: str) -> str:
    """Khóa lỏng: như norm_key nhưng bỏ luôn số/đơn vị & dấu câu -> chỉ còn chữ.
    Hữu ích để so khớp tên thuốc khi bỏ hàm lượng."""
    s = norm_key(s)
    s = re.sub(r"[0-9]+([.,][0-9]+)?", " ", s)          # bỏ số
    s = s.replace("mg", " ").replace("ml", " ").replace("mcg", " ")
    s = s.replace("/", " ").replace("%", " ").replace("-", " ")
    s = _WS.sub(" ", s).strip()
    return s


# Token hóa đơn giản cho char n-gram / word overlap
def tokens(s: str):
    return norm_key(s).split()


def char_ngrams(s: str, n: int = 3):
    s = norm_key(s).replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


if __name__ == "__main__":
    tests = [
        "bệnh trào ngược dạ dày - thực quản",
        "Chlorpheniramine 0.4 MG/ML",
        "Đái tháo đường típ 2",
        "NEUT% (Tỷ lệ % bạch cầu trung tính)",
    ]
    for t in tests:
        print(repr(t))
        print("  key      :", norm_key(t))
        print("  loose    :", norm_key_loose(t))
        print("  accents  :", strip_accents(t))
