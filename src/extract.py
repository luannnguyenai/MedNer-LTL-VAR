"""
extract.py — Bộ trích xuất khái niệm y tế (NER) BASELINE / BOOTSTRAP.

Đây là hệ thống chạy-được-ngay VÀ là silver-labeler để sinh dữ liệu huấn luyện
cho model NER fine-tuned (xem ARCHITECTURE.md). Ưu tiên PRECISION cho THUỐC và
KẾT_QUẢ_XÉT_NGHIỆM (bắt được chắc chắn), CHẨN_ĐOÁN neo vào ICD.

Thành phần:
  - Drug matcher: gazetteer hoạt chất (RxNorm IN/PIN/BN) + bắt hàm lượng đi kèm.
  - Lab matcher : mẫu "TÊN: GIÁ_TRỊ", "TÊN GIÁ_TRỊ", số + đơn vị.
  - Dx/Sym      : tách cụm lâm sàng theo cấu trúc bệnh án; neo ICD để phân biệt
                  CHẨN_ĐOÁN vs TRIỆU_CHỨNG.
  - Section parser: gán nhãn mục 1/2/3 -> phục vụ assertion isHistorical.

MỌI mã (candidates) đều lấy từ linker (KB), không sinh tự do -> không hallucination.
"""
import json
import os
import re

from vntext import norm_key
from linker import ICDLinker, RxNormLinker

# ----------------------------------------------------------------------------
DRUG_GAZ = os.path.join(os.path.dirname(__file__), "..", "kb", "drug_gazetteer.json")

# Từ khóa gợi ý CHẨN_ĐOÁN (nếu cụm chứa & link được ICD -> disease)
DX_KEYS = ["benh ", "viem", "suy tim", "suy than", "suy gan", "suy ho hap",
           "ung thu", "nhoi mau", "xo gan", "xo vua", "xo hoa",
           "hoi chung", "roi loan", "tang huyet ap", "dai thao duong",
           "gay xuong", "gay co", "khoi u", "u ac", "u tuyen", "u lanh",
           "thieu mau", "nhiem trung", "nhiem khuan", "loet",
           "hep ", "phinh ", "trao nguoc", "soi than", "soi mat", "lao phoi",
           "hen suyen", "hen phe quan", "dot quy", "tai bien", "rung nhi",
           "block", "ngoai tam thu", "nhip nhanh", "nhip cham",
           "tran dich", "tran khi", "co that", "viem phoi", "copd"]

# Cụm nên loại (không phải khái niệm y tế cần trả)
DROP_PHRASES = {"n/a", "na", "khong", "binh thuong", "on dinh", "khong ro",
                "khong xac dinh", "khong co gi dang chu y", "khong ghi nhan gi bat thuong"}

# Tiêu đề mục / dòng hành chính -> loại (khớp toàn bộ hoặc tiền tố)
HEADER_STOP = {
    "tien su benh", "tien su benh noi khoa", "tien su benh ly",
    "tien su benh hien tai", "tien su phau thuat thu thuat",
    "tien su phau thuat", "benh su", "benh su hien tai",
    "cac benh ly man tinh", "cac benh ly nen", "cac yeu to nguy co lien quan",
    "yeu to nguy co", "thuoc truoc khi nhap vien",
    "thuoc truoc khi nhap vien lan nay", "trieu chung hien tai",
    "cac trieu chung hien tai", "dac diem trieu chung",
    "ly do nhap vien", "ly do vao vien", "thoi diem khoi phat trieu chung",
    "danh gia tai benh vien", "ket qua kham lam sang", "ket qua xet nghiem",
    "ket qua chan doan hinh anh", "ket qua hinh anh",
    "cac su kien truoc khi nhap vien", "cac dien bien truoc khi nhap vien",
    "cac ket qua chan doan khac", "cac phat hien chan doan khac",
    "thu thuat thuc hien", "cac thu thuat da thuc hien",
    "tinh trang truoc nhap vien", "trieu chung khi nhap vien",
    "dien bien benh", "kham lam sang", "can lam sang", "vi tri",
    "muc do nghiem trong", "thoi gian", "tan suat", "chieu xa",
    "cac yeu to lam nang them", "cac yeu to lam giam",
    "cac trieu chung lien quan", "dau hieu sinh ton",
    "chua phat hien benh ly bat thuong", "cac tap kinh lam sang truoc day",
    "dien bien", "tinh trang", "dieu tri", "chan doan",
}

# Đơn vị / mẫu kết quả xét nghiệm
LAB_UNIT = r"(?:mg/dl|mmol/l|g/dl|g/l|u/l|iu/l|ng/ml|pg/ml|µmol/l|umol/l|mmhg|" \
           r"lần/phút|l/ph|/µl|/ul|10\^\d|%|bpm|mm|cm|ml|kg|°c|mcg|mg)"
# "tên: 12,3" hoặc "tên 12.3 unit"
RE_LAB_COLON = re.compile(
    r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9%()\/\s\.\-]{0,40}?)\s*[:=]\s*"
    r"([<>]?\s*\d+(?:[.,]\d+)?\s*" + LAB_UNIT + r"?)", re.I)
# Tên xét nghiệm phổ biến đứng trước 1 số không có dấu ':' -> "troponin 0.01", "EF30"
LAB_NAMES = (r"troponin|tropo|wbc|rbc|hgb|hct|plt|neut|lymph|lyph|mono|eos|baso|"
             r"cea|psa|crp|bnp|nt-probnp|inr|pt|aptt|glucose|hba1c|creatinin|"
             r"ure|ast|alt|ggt|bilirubin|natri|kali|clo|canxi|ef|spo2|tro")
RE_LAB_NAMED = re.compile(r"\b(" + LAB_NAMES + r")\b\s*[:=]?\s*"
                          r"(\d+(?:[.,]\d+)?\s*%?)", re.I)
# giá trị số trần kèm đơn vị (đứng độc lập)
RE_NUM_UNIT = re.compile(r"\b(\d+(?:[.,]\d+)?\s*" + LAB_UNIT + r")\b", re.I)

# hàm lượng thuốc theo sau tên: "25mg", "0.4 MG/ML", "325 mg po bid"
RE_DOSE = re.compile(
    r"\s*\d+(?:[.,]\d+)?\s*(?:mg/ml|mcg/ml|mg|mcg|ml|g|iu|unit|đơn vị)"
    r"(?:\s*/\s*ml)?(?:\s+(?:po|iv|im|sc|bid|tid|qid|qd|daily|prn|"
    r"uống|tiêm|ngậm|dán|bôi|x\s*\d+))*", re.I)

SECTION_RE = re.compile(r"^\s*(\d)\s*\.\s*(.{0,60})", re.M)


class Extractor:
    def __init__(self):
        self.icd = ICDLinker()
        self.rx = RxNormLinker()
        with open(DRUG_GAZ, encoding="utf-8") as f:
            self.drugs = json.load(f)["names"]
        self.drugset = set(self.drugs)
        # regex 1 lần cho toàn bộ hoạt chất (word boundary), longest-first đã sort
        # dùng alternation cho các tên >=5 ký tự để giảm FP
        big = [re.escape(n) for n in self.drugs if len(n) >= 5]
        self.drug_re = re.compile(r"(?<![a-z])(" + "|".join(big) + r")(?![a-z])", re.I)

    # ---- section map -------------------------------------------------------
    def section_spans(self, text):
        spans = []
        for m in SECTION_RE.finditer(text):
            spans.append((m.start(), m.group(2).strip()))
        spans.sort()
        return spans

    def section_of(self, pos, spans):
        lab = ""
        for s, name in spans:
            if s <= pos:
                lab = name
            else:
                break
        return lab

    # ---- drug extraction ---------------------------------------------------
    def find_drugs(self, text):
        out = []
        for m in self.drug_re.finditer(text):
            s, e = m.start(), m.end()
            # mở rộng sang phần hàm lượng ngay sau
            dm = RE_DOSE.match(text, e)
            if dm and dm.end() > e:
                e = dm.end()
            span_text = text[s:e].strip()
            # co lại nếu có khoảng trắng cuối do regex
            e = s + len(text[s:e].rstrip())
            out.append((s, e, text[s:e], "THUỐC"))
        return out

    # ---- lab extraction ----------------------------------------------------
    def find_labs(self, text):
        out = []
        for m in RE_LAB_COLON.finditer(text):
            name = m.group(1).strip(" -\t")
            val = m.group(2).strip()
            nk = norm_key(name)
            if not nk or nk in DROP_PHRASES or len(nk) < 2:
                continue
            # bỏ nếu "name" thực ra là cả câu dài (nhiều từ chức năng)
            if len(name) > 45:
                continue
            ns = m.start(1) + (len(m.group(1)) - len(m.group(1).lstrip()))
            ne = ns + len(name)
            vs = m.start(2) + (len(m.group(2)) - len(m.group(2).lstrip()))
            ve = vs + len(val)
            out.append((ns, ne, text[ns:ne], "TÊN_XÉT_NGHIỆM"))
            out.append((vs, ve, text[vs:ve], "KẾT_QUẢ_XÉT_NGHIỆM"))
        # mẫu "tên_lab số" không dấu hai chấm
        for m in RE_LAB_NAMED.finditer(text):
            ns, ne = m.start(1), m.end(1)
            vs, ve = m.start(2), m.end(2)
            out.append((ns, ne, text[ns:ne], "TÊN_XÉT_NGHIỆM"))
            out.append((vs, ve, text[vs:ve].strip(), "KẾT_QUẢ_XÉT_NGHIỆM"))
        return out

    # ---- diagnosis / symptom from clinical phrases -------------------------
    # tiền tố cần bóc để lấy tên bệnh sạch
    DX_PREFIX = re.compile(
        r"^(?:duoc |da |di |dang )?(?:chan doan (?:mac|xac dinh|la)?|"
        r"xac dinh|ket luan|nghi ngo|theo doi|tinh trang|phat hien(?: co)?|"
        r"mac |bi )\s*", re.I)

    def clinical_phrases(self, text):
        """Tách cụm lâm sàng theo: xuống dòng, ';', ',', ranh giới câu '. '."""
        phrases = []
        for m in re.finditer(r"[^\n;,]+", text):
            seg0, base0 = m.group(), m.start()
            for sub in re.finditer(r"[^.]*(?:\.(?=\s)|\.$|$)", seg0):
                stext = sub.group()
                if stext.strip():
                    self._emit_phrase(text, base0 + sub.start(), stext, phrases)
        return phrases

    def _emit_phrase(self, text, base, seg, phrases):
        # bỏ tiền tố gạch đầu dòng / số thứ tự
        lead = re.match(r"^[\s\-\*•\d\.\)]+", seg)
        off = lead.end() if lead else 0
        # "Nhãn: nội dung" -> lấy phần sau ':' làm cụm chính
        colon = seg.find(":")
        if 0 <= colon < 30 and colon + 1 < len(seg):
            off = colon + 1
            lead2 = re.match(r"^\s+", seg[off:])
            if lead2:
                off += lead2.end()
        if off >= len(seg) or not seg[off:].strip():
            return
        s = base + off
        raw = seg[off:].rstrip()
        # bỏ dấu câu cuối
        raw = re.sub(r"[.,;:\s\-]+$", "", raw)
        if not raw:
            return
        e = s + len(raw)
        phrases.append((s, e, text[s:e]))

    # mô tả bệnh nhân / hành chính -> bỏ
    PATIENT_DESC = re.compile(
        r"benh nhan (nam|nu)|(\d+ tuoi)|bi benh \d+|vao vien|nhap vien vi|"
        r"ly do (vao|nhap)|theo loi|dia chi|so dien thoai|ho ten", re.I)

    def classify_phrase(self, ptext):
        # bóc tiền tố "được chẩn đoán mắc ..." -> lấy tên bệnh sạch + offset
        nk_raw = norm_key(ptext)
        pm = self.DX_PREFIX.match(nk_raw)
        off = 0
        forced_dx = False
        if pm:
            # tính offset ký tự tương ứng trong ptext gốc (xấp xỉ theo số từ)
            forced_dx = True
            # cắt cùng số "từ" ở bản gốc
            nwords = len(pm.group().split())
            parts = ptext.split()
            if nwords < len(parts):
                clean = " ".join(parts[nwords:])
                off = ptext.find(parts[nwords]) if parts[nwords:] else 0
            else:
                clean = ptext
            ptext_eff = clean
        else:
            ptext_eff = ptext

        nk = norm_key(ptext_eff)
        if nk in DROP_PHRASES or len(nk) < 3:
            return None
        if nk in HEADER_STOP:
            return None
        # dòng hành chính / thời gian thuần -> bỏ
        if self.PATIENT_DESC.search(nk):
            return None
        # phi lâm sàng rõ ràng
        if re.search(r"mat viec|ca phe|caffeine|cong viec|xe cuu thuong|"
                     r"cuu thuong|the chien|dong ho|gio |phut\b", nk):
            return None
        if len(nk) > 60:
            return None
        is_dx = forced_dx or any(k in nk for k in DX_KEYS)
        codes = self.icd.link(ptext_eff, topk=3) if is_dx else []
        if is_dx:
            return ("CHẨN_ĐOÁN", codes, off, len(ptext_eff.rstrip()))
        if len(nk.split()) <= 8:
            return ("TRIỆU_CHỨNG", [], off, len(ptext_eff.rstrip()))
        return None

    # ---- orchestrate -------------------------------------------------------
    def extract(self, text):
        secs = self.section_spans(text)
        spans = []          # (s,e,text,type,candidates)
        occupied = []       # để tránh chồng lấn thô

        def overlaps(s, e):
            return any(not (e <= a or s >= b) for a, b in occupied)

        # 1) drugs (ưu tiên cao)
        for s, e, t, ty in self.find_drugs(text):
            if overlaps(s, e):
                continue
            cands = self.rx.link(t, topk=3)
            spans.append([s, e, t, ty, cands])
            occupied.append((s, e))

        # 2) labs
        for s, e, t, ty in self.find_labs(text):
            if overlaps(s, e):
                continue
            spans.append([s, e, t, ty, []])
            occupied.append((s, e))

        # 3) dx / symptoms từ cụm lâm sàng
        for s, e, t in self.clinical_phrases(text):
            if overlaps(s, e):
                continue
            res = self.classify_phrase(t)
            if not res:
                continue
            ty, cands, off, ln = res
            s2 = s + off
            e2 = s2 + ln
            if overlaps(s2, e2) or e2 <= s2:
                continue
            spans.append([s2, e2, text[s2:e2], ty, cands])
            occupied.append((s2, e2))

        spans.sort(key=lambda x: x[0])
        return spans, secs


if __name__ == "__main__":
    ex = Extractor()
    sample = ('Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, '
              'đau thượng vị, ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - '
              'thực quản. Bệnh nhân có tiền sử sử dụng Chlorpheniramine 0.4 MG/ML, '
              'Capsaicin 0.38 MG/ML, WBC:14,43; troponin 0.01')
    spans, secs = ex.extract(sample)
    for s, e, t, ty, c in spans:
        print(f"[{ty:16s}] '{t}'  pos=({s},{e})  cand={c}")
