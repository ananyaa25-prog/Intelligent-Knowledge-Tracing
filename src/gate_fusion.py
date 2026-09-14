# src/gate_fusion.py

import torch
import torch.nn as nn


class GateFusion(nn.Module):
    """
    The core contribution: a learned gate that decides, per-timestep and
    per-dimension, how much to trust short-term state vs. long-term state,
    conditioned on graph context and metadata confidence.

    g is a VECTOR (same size as the hidden states), not a scalar -- this
    lets the model weight different dimensions of "knowledge" differently
    (e.g. trust recent behavior more for volatile skills, long-term more
    for stable ones) rather than one blanket trust percentage for
    everything at once.
    """

    def __init__(self, hidden_dim: int = 64, graph_dim: int = 64, meta_dim: int = 16):
        super().__init__()
        gate_input_dim = hidden_dim + hidden_dim + graph_dim + meta_dim
        self.gate_net = nn.Linear(gate_input_dim, hidden_dim)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_short: torch.Tensor, h_long: torch.Tensor,
                h_graph: torch.Tensor, h_meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h_short: [batch, seq_len, hidden_dim]
        h_long:  [batch, seq_len, hidden_dim]
        h_graph: [batch, seq_len, graph_dim]   -- graph-informed embedding
                 for the skill at each timestep (gathered per-question, NOT
                 the full 124-skill graph -- see note below)
        h_meta:  [batch, seq_len, meta_dim]    -- confidence signal

        Returns:
            pred: [batch, seq_len]  -- predicted P(correct)
            g:    [batch, seq_len, hidden_dim]  -- the gate values themselves,
                  kept for the interpretability figure your professor
                  asked about (visualize g over a student's sequence)
        """
        gate_input = torch.cat([h_short, h_long, h_graph, h_meta], dim=-1)
        g = torch.sigmoid(self.gate_net(gate_input))

        h_final = g * h_short + (1 - g) * h_long
        pred = torch.sigmoid(self.output_head(h_final)).squeeze(-1)

        return pred, g


if __name__ == "__main__":
    print("Testing GateFusion on dummy data...")

    batch, seq_len, hidden_dim, graph_dim, meta_dim = 4, 20, 64, 64, 16

    h_short = torch.randn(batch, seq_len, hidden_dim)
    h_long = torch.randn(batch, seq_len, hidden_dim)
    h_graph = torch.randn(batch, seq_len, graph_dim)
    h_meta = torch.randn(batch, seq_len, meta_dim)

    gate = GateFusion(hidden_dim=hidden_dim, graph_dim=graph_dim, meta_dim=meta_dim)
    pred, g = gate(h_short, h_long, h_graph, h_meta)

    print(f"Prediction shape: {pred.shape}")
    print(f"Gate shape: {g.shape}")
    print(f"NaN count in pred: {torch.isnan(pred).sum().item()}")
    print(f"NaN count in gate: {torch.isnan(g).sum().item()}")
    print(f"Prediction range: [{pred.min().item():.4f}, {pred.max().item():.4f}] (should be within [0,1])")
    print(f"Gate value range: [{g.min().item():.4f}, {g.max().item():.4f}] (should be within [0,1])")

    # Sanity check the gate isn't collapsed to a constant (e.g. always 0.5,
    # always saturated at 0 or 1 before any training has even happened) --
    # a healthy random-init gate should show real spread.
    gate_std = g.std().item()
    print(f"Gate value std across all positions (should be > 0.05, not near-zero): {gate_std:.4f}")

    assert pred.shape == (batch, seq_len), "Prediction shape mismatch"
    assert g.shape == (batch, seq_len, hidden_dim), "Gate shape mismatch"
    assert torch.isnan(pred).sum() == 0 and torch.isnan(g).sum() == 0, "NaNs present"
    assert 0 <= pred.min() and pred.max() <= 1, "Predictions out of valid probability range"
    assert gate_std > 0.05, "Gate has collapsed to a near-constant value even before training"
    print("\nPASS: shapes correct, no NaNs, predictions valid probabilities, gate shows real variation.")