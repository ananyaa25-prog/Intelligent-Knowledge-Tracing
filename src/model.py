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
        # Difficulty bucket embedding: 0 = unknown/padding, 1 = easy, 2 = medium,
        # 3 = hard. Buckets are derived from TRAIN-only per-problem correctness.
        self.diff_emb = nn.Embedding(4, hidden_dim)

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
                hint_used, difficulty=None, edge_index=None, edge_weight=None):
        if difficulty is None:
            difficulty = torch.zeros_like(skill_id)

        x = (self.skill_emb(skill_id[:, :-1])
             + self.correct_emb(correct[:, :-1].long())
             + self.diff_emb(difficulty[:, :-1].long()))
        hist_mask = mask[:, :-1].float()

        h_short = self.short_term(x, hist_mask)
        h_long = self.long_term(x, hist_mask)

        h_meta = self.metadata_encoder(
            response_time[:, :-1], attempt_count[:, :-1], hint_used[:, :-1]
        )

        next_skill_id = skill_id[:, 1:]
        # Raw target representation: the target skill plus its difficulty bucket.
        target_emb = self.skill_emb(next_skill_id) + self.diff_emb(difficulty[:, 1:].long())

        if self.use_graph:
            all_skill_ids = torch.arange(self.skill_emb.num_embeddings, device=skill_id.device)
            all_skill_emb = self.skill_emb(all_skill_ids)
            graph_emb_table = self.graph_encoder(all_skill_emb, edge_index, edge_weight)
            h_graph = graph_emb_table[next_skill_id]
            # Target-aware head input: graph-refined target skill + difficulty.
            h_target = h_graph + self.diff_emb(difficulty[:, 1:].long())
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
            # Even without a graph, the head still receives the raw target-skill
            # embedding so the ablation actually knows WHAT it is predicting.
            h_target = target_emb

        if self.use_gate:
            pred, g = self.gate_fusion(h_short, h_long, h_graph, h_meta, h_target)
        else:
            # Genuine no-gate ablation: graph signal IS included in the
            # prediction, just combined by naive averaging instead of a
            # learned gate. This isolates what the GATE specifically buys
            # you, holding "graph information is present" constant across
            # both this and the full_gated condition.
            h_final = (h_short + h_long + h_graph) / 3.0
            pred = torch.sigmoid(
                self.gate_fusion.output_head(torch.cat([h_final, h_target], dim=-1))
            ).squeeze(-1)
            g = torch.zeros_like(h_short)

        return pred, g

    def _final_states(self, skill_seq, correct_seq, mask, response_time,
                      attempt_count, hint_used, difficulty_seq):
        """Encode a single student's history and return the FINAL (most recent)
        real timestep's short-term, long-term, and metadata states, plus the
        length. Used by predict_candidates so it scores the NEXT question from
        the exact same encoder states forward() uses."""
        x = (self.skill_emb(skill_seq.long())
             + self.correct_emb(correct_seq.long())
             + self.diff_emb(difficulty_seq.long()))
        m = mask.float()
        h_short = self.short_term(x, m)
        h_long = self.long_term(x, m)
        h_meta = self.metadata_encoder(response_time, attempt_count, hint_used)

        lengths = mask.sum(dim=1).long()  # [batch]
        batch = h_short.shape[0]
        last = (lengths - 1).clamp(min=0)
        idx = torch.arange(batch, device=h_short.device)
        h_short_end = h_short[idx, last]   # [batch, hidden]
        h_long_end = h_long[idx, last]
        h_meta_end = h_meta[idx, last]
        return h_short_end, h_long_end, h_meta_end

    def predict_candidates(self, skill_seq, correct_seq, mask, response_time,
                           attempt_count, hint_used, difficulty_seq, candidates,
                           edge_index=None, edge_weight=None):
        """
        Scores candidate NEXT questions. Each candidate is a (skill_id, difficulty)
        tuple using the model's internal (reindexed) skill id and difficulty bucket.

        skill_seq etc. are [1, L] tensors describing the history so far.
        candidates is a list of (skill_id, difficulty_bucket) pairs.
        Returns a tensor of shape [num_candidates] with P(correct) for each.
        """
        h_short_end, h_long_end, h_meta_end = self._final_states(
            skill_seq, correct_seq, mask, response_time, attempt_count,
            hint_used, difficulty_seq
        )  # each [1, hidden]

        num_candidates = len(candidates)
        if num_candidates == 0:
            return torch.zeros(0)

        cand_skill = torch.tensor([c[0] for c in candidates], dtype=torch.long,
                                  device=h_short_end.device)
        cand_diff = torch.tensor([c[1] for c in candidates], dtype=torch.long,
                                 device=h_short_end.device)

        if self.use_graph:
            all_skill_ids = torch.arange(self.skill_emb.num_embeddings, device=h_short_end.device)
            all_skill_emb = self.skill_emb(all_skill_ids)
            graph_emb_table = self.graph_encoder(all_skill_emb, edge_index, edge_weight)
            h_graph = graph_emb_table[cand_skill]              # [K, hidden]
            h_target = h_graph + self.diff_emb(cand_diff)      # [K, hidden]
        else:
            h_graph = torch.zeros(num_candidates, self.hidden_dim, device=h_short_end.device)
            h_target = self.skill_emb(cand_skill) + self.diff_emb(cand_diff)

        # Tile the single final state across all candidates.
        h_short_k = h_short_end.expand(num_candidates, -1)
        h_long_k = h_long_end.expand(num_candidates, -1)
        h_meta_k = h_meta_end.expand(num_candidates, -1)

        if self.use_gate:
            pred, _ = self.gate_fusion(h_short_k, h_long_k, h_graph, h_meta_k, h_target)
        else:
            h_final = (h_short_k + h_long_k + h_graph) / 3.0
            pred = torch.sigmoid(
                self.gate_fusion.output_head(torch.cat([h_final, h_target], dim=-1))
            ).squeeze(-1)

        return pred


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
                     hint_used, edge_index=edge_index, edge_weight=edge_weight)

    print(f"\nPrediction shape: {pred.shape}")
    print(f"NaN count in prediction: {torch.isnan(pred).sum().item()}")
    print(f"Prediction range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    assert torch.isnan(pred).sum() == 0, "NaNs in prediction on REAL data"

    # ---- Cross-check forward() vs predict_candidates() ---------------------
    # Take student 0's sequence. The final real step t (predicting t+1) must be
    # reproducible by predict_candidates() fed only the history up to t.
    model.eval()
    with torch.no_grad():
        row_mask = mask[0].bool()
        real_len = int(row_mask.sum().item())
        # forward predicts correct[1:] so the last valid prediction is at t = real_len - 2
        t = real_len - 2
        pred_last = float(pred[0, t].item())

        hist_len = t + 1  # interactions 0..t inclusive are history
        cand_skill = int(skill_id[0, t + 1].item())
        cand_diff = 0  # windowed_data.pkl predates difficulty; use unknown bucket
        scores = model.predict_candidates(
            skill_id[0:1, :hist_len], correct[0:1, :hist_len], mask[0:1, :hist_len],
            response_time[0:1, :hist_len], attempt_count[0:1, :hist_len],
            hint_used[0:1, :hist_len], torch.zeros_like(skill_id[0:1, :hist_len]),
            [(cand_skill, cand_diff)], edge_index, edge_weight
        )
        pred_cand = float(scores[0].item())
        print(f"\nforward() last-step pred: {pred_last:.4f}")
        print(f"predict_candidates() pred for the same target: {pred_cand:.4f}")
        assert abs(pred_last - pred_cand) < 1e-3, \
            f"forward() and predict_candidates() disagree: {pred_last:.4f} vs {pred_cand:.4f}"

        # ---- Target-skill sensitivity check --------------------------------
        other_skills = [s for s in range(1, num_skills + 1) if s != cand_skill][:2]
        other_scores = model.predict_candidates(
            skill_id[0:1, :hist_len], correct[0:1, :hist_len], mask[0:1, :hist_len],
            response_time[0:1, :hist_len], attempt_count[0:1, :hist_len],
            hint_used[0:1, :hist_len], torch.zeros_like(skill_id[0:1, :hist_len]),
            [(s, cand_diff) for s in other_skills], edge_index, edge_weight
        )
        spread = max(float(other_scores[0].item()), float(other_scores[1].item())) - min(
            float(other_scores[0].item()), float(other_scores[1].item()))
        print(f"Prediction spread across 2 different target skills: {spread:.4f} (must be > 0)")
        assert spread > 0, "Prediction does not depend on the target skill"

    print("\nPASS: full model runs end-to-end on real ASSIST09 data, no NaNs; forward() and predict_candidates() agree; output depends on target skill.")