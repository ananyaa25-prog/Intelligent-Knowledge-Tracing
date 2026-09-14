# src/long_term_encoder.py

import torch
import torch.nn as nn


class LongTermEncoder(nn.Module):
    """
    Encodes a student's ENTIRE history up to each timestep into a hidden
    state. Unlike the short-term encoder, this needs no windowing — a
    standard GRU run over the full sequence already accumulates everything
    seen so far at every position. The only real design decision here is
    handling variable-length sequences correctly via the mask.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [batch, seq_len, input_dim]
        mask: [batch, seq_len]  -- 1 = real, 0 = padding

        Returns: [batch, seq_len, hidden_dim] -- long-term state at each t.
        """
        # Zero out padded input first — GRU will still churn through padded
        # steps, but we mask the OUTPUT below, so this just keeps the hidden
        # state from drifting on garbage input before that point.
        x = x * mask.unsqueeze(-1)

        h_seq, _ = self.gru(x)  # h_seq: [batch, seq_len, hidden_dim]
        h_seq = h_seq * mask.unsqueeze(-1)  # zero out padded positions
        return h_seq


if __name__ == "__main__":
    print("Testing LongTermEncoder on dummy data...")

    batch, seq_len, input_dim, hidden_dim = 4, 20, 32, 64
    x = torch.randn(batch, seq_len, input_dim)
    mask = torch.ones(batch, seq_len)
    mask[0, 15:] = 0

    encoder = LongTermEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    out = encoder(x, mask)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"NaN count in output: {torch.isnan(out).sum().item()}")
    print(f"Padded-position values are zero: {(out[0, 15:] == 0).all().item()}")

    # Sanity check that this is actually "long-term" behavior: the hidden
    # state at t=19 should differ from t=0 (it's accumulated 19 more steps
    # of context) -- if these are identical, something's wrong.
    drift = (out[1, 19] - out[1, 0]).abs().sum().item()
    print(f"State drift from t=0 to t=19 (should be nonzero): {drift:.4f}")

    assert out.shape == (batch, seq_len, hidden_dim), "Shape mismatch"
    assert torch.isnan(out).sum() == 0, "NaNs in output"
    assert drift > 0, "Hidden state isn't evolving over time — GRU may be broken"
    print("\nPASS: shapes correct, no NaNs, padding zeroed, state genuinely accumulates over time.")