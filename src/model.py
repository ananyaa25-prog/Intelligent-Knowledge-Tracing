# src/model.py

import pickle
import numpy as np
import torch
import torch.nn as nn

from short_term_encoder import ShortTermEncoder
from long_term_encoder import LongTermEncoder
from graph_encoder import GraphEncoder, adjacency_to_edge_index
from metadata_encoder import MetadataEncoder
from gate_fusion import GateFusion


# model.py — modify GatedKTModel to support ablation flags

class GatedKTModel(nn.Module):
    def __init__(self, num_skills: int, hidden_dim: int = 64,
                 window_size: int = 10, meta_dim: int = 16,
                 rt_log_mean: float = 8.5, rt_log_std: float = 1.2,
                 use_graph: bool = True, use_gate: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_graph = use_graph
        self.use_gate = use_gate

        self.skill_emb = nn.Embedding(num_skills + 1, hidden_dim, padding_idx=0)
        self.correct_emb = nn.Embedding(2, hidden_dim)

        self.short_term = ShortTermEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            window_size=window_size)
        self.long_term = LongTermEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.metadata_encoder = MetadataEncoder(hidden_dim=meta_dim,
                                                 rt_log_mean=rt_log_mean,
                                                 rt_log_std=rt_log_std)

        if use_graph:
            self.graph_encoder = GraphEncoder(emb_dim=hidden_dim, heads=1)
        else:
            self.graph_encoder = None  # DKT-only baseline: no graph at all

        graph_dim = hidden_dim  # gate_fusion still expects a graph_dim-sized tensor
        self.gate_fusion = GateFusion(hidden_dim=hidden_dim, graph_dim=graph_dim,
                                       meta_dim=meta_dim)

    def forward(self, skill_id, correct, mask, response_time, attempt_count,
                hint_used, edge_index=None, edge_weight=None):
        x = self.skill_emb(skill_id[:, :-1]) + self.correct_emb(correct[:, :-1].long())
        hist_mask = mask[:, :-1].float()

        h_short = self.short_term(x, hist_mask)
        h_long = self.long_term(x, hist_mask)

        h_meta = self.metadata_encoder(
            response_time[:, :-1], attempt_count[:, :-1], hint_used[:, :-1]
        )

        if self.use_graph:
            all_skill_ids = torch.arange(self.skill_emb.num_embeddings, device=skill_id.device)
            all_skill_emb = self.skill_emb(all_skill_ids)
            graph_emb_table = self.graph_encoder(all_skill_emb, edge_index, edge_weight)
            next_skill_id = skill_id[:, 1:]
            h_graph = graph_emb_table[next_skill_id]
        else:
            # DKT-only baseline: feed zeros in place of graph context, so
            # GateFusion's architecture stays identical across all three
            # conditions -- only the graph SIGNAL is removed, not the
            # network's capacity. This is important: if you shrank the
            # model too when removing the graph, a worse baseline result
            # wouldn't prove the graph helped, it'd just prove "smaller
            # model is worse," which is a different and useless claim.
            h_graph = torch.zeros(h_short.shape[0], h_short.shape[1],
                                   self.hidden_dim, device=h_short.device)

        if self.use_gate:
            pred, g = self.gate_fusion(h_short, h_long, h_graph, h_meta)
        else:
            # Genuine no-gate ablation: graph signal IS included in the
            # prediction, just combined by naive averaging instead of a
            # learned gate. This isolates what the GATE specifically buys
            # you, holding "graph information is present" constant across
            # both this and the full_gated condition.
            h_final = (h_short + h_long + h_graph) / 3.0
            pred = torch.sigmoid(self.gate_fusion.output_head(h_final)).squeeze(-1)
            g = torch.zeros_like(h_short)

        return pred, g


if __name__ == "__main__":
    print("Loading real data and graph for full integration test...")

    with open("data/processed/windowed_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("data/processed/skill_graph.pkl", "rb") as f:
        graph_data = pickle.load(f)

    num_skills = data["num_skills"]
    train = data["splits"]["train"]
    print(f"num_skills: {num_skills}")

    adj_matrix = torch.tensor(graph_data["adjacency_matrix"], dtype=torch.float32)
    print(f"adjacency_matrix shape: {adj_matrix.shape}")

    # Reuse graph_encoder.py's own conversion function -- this is the one
    # that correctly produces edge_weight alongside edge_index. Don't
    # reimplement this a second time with plain np.nonzero, which drops
    # the PPMI weights and silently treats every edge as equal strength.
    edge_index, edge_weight = adjacency_to_edge_index(adj_matrix)
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_weight shape: {edge_weight.shape}")

    print(f"Mismatch check: num_skills({num_skills}) + 1 vs adjacency rows({adj_matrix.shape[0]})")

    batch_size = 4
    skill_id = train["skill_seq"][:batch_size].long()
    correct = train["correct_seq"][:batch_size].long()
    mask = train["mask"][:batch_size].float()
    response_time = train["response_time_seq"][:batch_size].float()
    attempt_count = train["attempt_seq"][:batch_size].float()
    hint_used = train["hint_seq"][:batch_size].float()

    print(f"Real batch skill_id shape: {skill_id.shape}")
    print(f"Max skill_id in batch: {skill_id.max().item()} (must be <= num_skills = {num_skills})")
    print(f"Max edge_index value: {edge_index.max().item()} (must be < adjacency rows = {adj_matrix.shape[0]})")

    model = GatedKTModel(num_skills=num_skills)
    pred, g = model(skill_id, correct, mask, response_time, attempt_count,
                     hint_used, edge_index, edge_weight)

    print(f"\nPrediction shape: {pred.shape}")
    print(f"NaN count in prediction: {torch.isnan(pred).sum().item()}")
    print(f"Prediction range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    assert torch.isnan(pred).sum() == 0, "NaNs in prediction on REAL data"
    print("\nPASS: full model runs end-to-end on real ASSIST09 data, no NaNs.")