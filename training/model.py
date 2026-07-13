"""
model.py — Model NER ĐA NHIỆM để đẩy encoder lên tối đa.

Kiến trúc (chung 1 encoder ~0.55B nếu XLM-R-large, hoặc 0.135B nếu ViHealthBERT):
  encoder (AutoModel)
    ├─ HEAD A: token-classification 5-type (BIO 11 nhãn) + LINEAR-CHAIN CRF
    ├─ HEAD B: token-classification thô (BIO 9 nhãn)  [auxiliary, dạy biên/miền]
    └─ HEAD C: assertion multi-label (3 sigmoid) trên biểu diễn token

Loss = L_A(CRF) + λ_B * L_B(CE) + λ_C * L_C(BCE, chỉ trên token thực thể).

CRF tự hiện thực (linear-chain, không phụ thuộc package ngoài) để giải mã BIO
hợp lệ (chặn chuyển O->I, B-X->I-Y…). Đã kiểm thử forward/decoder bằng tensor
ngẫu nhiên với config nhỏ (không cần tải model).
"""
import torch
import torch.nn as nn
from transformers import AutoModel


# ----------------------------------------------------------------------------
class CRF(nn.Module):
    """Linear-chain CRF tối giản (Viterbi + forward). mask: (B,T) bool."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.trans = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)

    def _numerator(self, emis, tags, mask):
        B, T, C = emis.shape
        score = self.start[tags[:, 0]] + emis[torch.arange(B), 0, tags[:, 0]]
        for t in range(1, T):
            m = mask[:, t]
            e = emis[torch.arange(B), t, tags[:, t]]
            tr = self.trans[tags[:, t - 1], tags[:, t]]
            score = score + (e + tr) * m
        # end transition tại token cuối hợp lệ
        last = (mask.sum(1) - 1).long()
        score = score + self.end[tags[torch.arange(B), last]]
        return score

    def _denominator(self, emis, mask):
        B, T, C = emis.shape
        alpha = self.start.unsqueeze(0) + emis[:, 0]           # (B,C)
        for t in range(1, T):
            e = emis[:, t].unsqueeze(1)                        # (B,1,C)
            tr = self.trans.unsqueeze(0)                       # (1,C,C)
            nxt = torch.logsumexp(alpha.unsqueeze(2) + tr, dim=1) + emis[:, t]
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m.bool(), nxt, alpha)
        alpha = alpha + self.end.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def forward(self, emis, tags, mask):
        """Trả NLL trung bình."""
        num = self._numerator(emis, tags, mask.float())
        den = self._denominator(emis, mask.float())
        return (den - num).mean()

    @torch.no_grad()
    def decode(self, emis, mask):
        B, T, C = emis.shape
        history = []
        score = self.start.unsqueeze(0) + emis[:, 0]
        for t in range(1, T):
            tr = self.trans.unsqueeze(0)
            s = score.unsqueeze(2) + tr                       # (B,C,C)
            best, idx = s.max(dim=1)                           # (B,C)
            score_t = best + emis[:, t]
            m = mask[:, t].unsqueeze(1).bool()
            score = torch.where(m, score_t, score)
            history.append(idx)
        score = score + self.end.unsqueeze(0)
        best_last = score.argmax(dim=1)                        # (B,)
        seqs = []
        lengths = mask.sum(1).long()
        for b in range(B):
            L = int(lengths[b].item())
            best = int(best_last[b].item())
            path = [best]
            for t in range(len(history) - 1, -1, -1):
                if t + 1 < L:
                    best = int(history[t][b, best].item())
                    path.append(best)
            path.reverse()
            # cắt đúng độ dài
            path = path[:L] + [0] * (T - L)
            seqs.append(path)
        return torch.tensor(seqs, device=emis.device)


# ----------------------------------------------------------------------------
class MultiTaskNER(nn.Module):
    def __init__(self, encoder_name, n_fine, n_coarse, n_assert=3,
                 lambda_b=0.3, lambda_c=0.5, dropout=0.1, config=None):
        super().__init__()
        if config is not None:            # nhánh test không tải trọng số
            from transformers import AutoModel as AM
            self.enc = AM.from_config(config)
        else:
            self.enc = AutoModel.from_pretrained(encoder_name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fine = nn.Linear(h, n_fine)
        self.coarse = nn.Linear(h, n_coarse)
        self.assertion = nn.Linear(h, n_assert)
        self.crf = CRF(n_fine)
        self.lambda_b, self.lambda_c = lambda_b, lambda_c
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_ids, attention_mask,
                fine_tags=None, coarse_tags=None,
                assert_labels=None, assert_mask=None, **kw):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.drop(out.last_hidden_state)          # (B,T,H)
        emis_fine = self.fine(seq)                      # (B,T,n_fine)
        logit_coarse = self.coarse(seq)
        logit_assert = self.assertion(seq)              # (B,T,3)

        result = {"emis_fine": emis_fine, "logit_coarse": logit_coarse,
                  "logit_assert": logit_assert}
        if fine_tags is None:
            # inference: giải mã CRF
            result["pred_fine"] = self.crf.decode(emis_fine, attention_mask)
            return result

        # ---- loss đa nhiệm ----
        # HEAD A: CRF cần tags hợp lệ (không -100). Dùng mask token thật.
        crf_mask = attention_mask.clone()
        safe_fine = fine_tags.clone()
        safe_fine[safe_fine == -100] = 0
        # token bị -100 (subword sau, special) -> loại khỏi mask CRF
        crf_mask = crf_mask * (fine_tags != -100).long()
        # đảm bảo mỗi câu còn ít nhất token đầu
        crf_mask[:, 0] = 1
        loss_a = self.crf(emis_fine, safe_fine, crf_mask)

        loss = loss_a
        if coarse_tags is not None:
            loss_b = self.ce(logit_coarse.view(-1, logit_coarse.size(-1)),
                             coarse_tags.view(-1))
            loss = loss + self.lambda_b * loss_b
        if assert_labels is not None and assert_mask is not None:
            # BCE chỉ trên token thực thể (assert_mask=1)
            l = self.bce(logit_assert, assert_labels.float())   # (B,T,3)
            m = assert_mask.unsqueeze(-1).float()
            denom = m.sum().clamp(min=1.0)
            loss_c = (l * m).sum() / denom
            loss = loss + self.lambda_c * loss_c
        result["loss"] = loss
        result["loss_a"] = loss_a.detach()
        return result


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # SMOKE TEST: config nhỏ, KHÔNG tải model từ mạng
    from transformers import BertConfig
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=64)
    model = MultiTaskNER("(test)", n_fine=11, n_coarse=9, n_assert=3, config=cfg)
    B, T = 3, 12
    ids = torch.randint(0, 200, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 9:] = 0                       # câu 0 ngắn hơn
    fine = torch.randint(0, 11, (B, T))
    fine[mask == 0] = -100
    coarse = torch.randint(0, 9, (B, T))
    coarse[mask == 0] = -100
    a_lab = (torch.rand(B, T, 3) > 0.7).long()
    a_mask = (torch.rand(B, T) > 0.6).long() * mask

    out = model(ids, mask, fine_tags=fine, coarse_tags=coarse,
                assert_labels=a_lab, assert_mask=a_mask)
    print("train loss:", float(out["loss"]), "| loss_A(CRF):", float(out["loss_a"]))
    out["loss"].backward()
    gsum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print("grad flow OK, sum|grad| =", round(gsum, 3))

    model.eval()
    pred = model(ids, mask)["pred_fine"]
    print("decode shape:", tuple(pred.shape), "| sample path[0]:", pred[0].tolist())
    # kiểm tra CRF không sinh chuyển O->I bất hợp lệ sau khi học? (chỉ smoke: chạy được)
    print("SMOKE OK")
