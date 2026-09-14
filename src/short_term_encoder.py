# src/short_term_encoder.py

import torch
import torch.nn as nn


class ShortTermEncoder(nn.Module):
    """
    Encodes a student's recent behavior (last k interactions) into a
    hidden state. This is NOT a continuously-running GRU over the full
    history — it's deliberately re-run on a fresh, short window at every
    timestep, because a GRU fed the whole sequence would still "remember"
    everything from step 1 regardless of what you call it.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, window_size: int = 10):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [batch, seq_len, input_dim]  -- per-timestep input embeddings
        mask: [batch, seq_len]             -- 1 = real, 0 = padding

        Returns: [batch, seq_len, hidden_dim] -- short-term state at each t,
        computed ONLY from the last `window_size` steps up to and including t.
        """
        batch, seq_len, dim = x.shape
        k = self.window_size

        # Pad the front so every position has a full window to look back on
        pad = torch.zeros(batch, k - 1, dim, device=x.device)
        x_padded = torch.cat([pad, x], dim=1)  # [batch, seq_len + k - 1, dim]

        # Build all windows at once via unfold — avoids a slow per-timestep loop
        windows = x_padded.unfold(1, k, 1)          # [batch, seq_len, dim, k]
        windows = windows.permute(0, 1, 3, 2)         # [batch, seq_len, k, dim]
        windows = windows.reshape(batch * seq_len, k, dim)

        _, h_n = self.gru(windows)      # h_n: [1, batch*seq_len, hidden_dim]
        h_short = h_n.squeeze(0).view(batch, seq_len, self.hidden_dim)

        # Zero out padded positions so they don't pollute downstream gating
        h_short = h_short * mask.unsqueeze(-1)
        return h_short


if __name__ == "__main__":
    print("Testing ShortTermEncoder on dummy data...")

    batch, seq_len, input_dim, hidden_dim = 4, 20, 32, 64
    x = torch.randn(batch, seq_len, input_dim)
    mask = torch.ones(batch, seq_len)
    mask[0, 15:] = 0  # simulate one student with a shorter real history

    encoder = ShortTermEncoder(input_dim=input_dim, hidden_dim=hidden_dim, window_size=10)
    out = encoder(x, mask)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"NaN count in output: {torch.isnan(out).sum().item()}")
    print(f"Padded-position values are zero: {(out[0, 15:] == 0).all().item()}")

    assert out.shape == (batch, seq_len, hidden_dim), "Shape mismatch"
    assert torch.isnan(out).sum() == 0, "NaNs in output"
    print("\nPASS: shapes correct, no NaNs, padding correctly zeroed.")