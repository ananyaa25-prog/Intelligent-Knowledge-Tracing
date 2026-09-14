# src/metadata_encoder.py

import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    """
    Turns per-interaction metadata (response_time, attempt_count, hint_used)
    into a learned "confidence/reliability" vector.

    Normalization uses FIXED constants (passed in at construction, computed
    once from the training set) rather than per-batch statistics. Per-batch
    normalization is nondeterministic (same input -> different output
    depending on batch composition) and breaks entirely on batch size 1 --
    both are unacceptable for a model you're trying to evaluate reproducibly.
    """

    def __init__(self, hidden_dim: int = 16, rt_log_mean: float = 8.5, rt_log_std: float = 1.2):
        super().__init__()
        # Defaults above are reasonable ballpark values for ASSISTments
        # response times in ms (log-scale) -- replace with the ACTUAL
        # mean/std computed from your training split before real training.
        self.rt_log_mean = rt_log_mean
        self.rt_log_std = rt_log_std

        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, response_time: torch.Tensor, attempt_count: torch.Tensor,
                hint_used: torch.Tensor) -> torch.Tensor:
        rt = torch.log1p(response_time.clamp(min=0))
        rt = (rt - self.rt_log_mean) / (self.rt_log_std + 1e-6)  # fixed constants, not batch stats

        ac = (attempt_count.clamp(min=1, max=10) - 1).float()
        hu = hint_used.float()

        feats = torch.stack([rt, ac, hu], dim=-1)
        return self.net(feats)


if __name__ == "__main__":
    print("Testing MetadataEncoder on dummy data...")

    batch, seq_len = 4, 20
    response_time = torch.randint(500, 30000, (batch, seq_len)).float()
    attempt_count = torch.randint(1, 5, (batch, seq_len)).float()
    hint_used = torch.randint(0, 2, (batch, seq_len)).float()

    encoder = MetadataEncoder(hidden_dim=16)
    out = encoder(response_time, attempt_count, hint_used)

    print(f"Output shape: {out.shape}")
    print(f"NaN count: {torch.isnan(out).sum().item()}")

    confident = encoder(
        torch.tensor([[1000.0]]), torch.tensor([[1.0]]), torch.tensor([[0.0]])
    )
    struggling = encoder(
        torch.tensor([[25000.0]]), torch.tensor([[4.0]]), torch.tensor([[1.0]])
    )
    diff = (confident - struggling).abs().sum().item()
    print(f"Confident vs struggling embedding difference (should be clearly nonzero): {diff:.4f}")

    assert out.shape == (batch, seq_len, 16), "Shape mismatch"
    assert torch.isnan(out).sum() == 0, "NaNs in output"
    assert diff > 0.5, "Encoder isn't meaningfully distinguishing confident vs struggling interactions"
    print("\nPASS: shapes correct, no NaNs, encoder differentiates confident vs struggling answers.")