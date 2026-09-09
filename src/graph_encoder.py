"""
src/graph_encoder.py
----------------------
Stage 3: Graph Attention Encoder

Reads the PPMI adjacency matrix from graph_builder.py and refines skill
embeddings using message passing over the graph — related skills inform
each other's representation.

This file is written to be testable STANDALONE before it ever touches
the real model — run this file directly to sanity-check shapes and
catch bugs here, not three files downstream during model training.
"""

import pickle
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


GRAPH_PATH = "data/processed/skill_graph.pkl"


def adjacency_to_edge_index(adj_matrix: torch.Tensor, threshold: float = 0.0):
    """
    Convert a dense [num_skills, num_skills] adjacency matrix into the
    edge_index / edge_weight format PyG expects.
    """
    adj_matrix = adj_matrix.clone()
    adj_matrix[adj_matrix <= threshold] = 0
    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()   # [2, num_edges]
    edge_weight = adj_matrix[edge_index[0], edge_index[1]]
    return edge_index, edge_weight


class GraphEncoder(nn.Module):
    """
    Single-layer Graph Attention over the skill relation graph.
    Input: the full skill embedding table (not a batch — GAT needs the
    whole graph structure to propagate messages correctly).
    Output: the same shape, but each skill's embedding now incorporates
    information from its graph neighbors, weighted by learned attention.
    """
    def __init__(self, emb_dim: int, heads: int = 1):
        super().__init__()
        self.gat = GATConv(emb_dim, emb_dim, heads=heads, edge_dim=1)

    def forward(self, skill_embeddings: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None):
        edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
        return self.gat(skill_embeddings, edge_index, edge_attr=edge_attr)


def run_sanity_test():
    """
    Standalone test — run this file directly. Loads the REAL adjacency
    matrix you built in graph_builder.py, feeds it random embeddings
    (we're testing plumbing here, not real learned embeddings yet),
    and checks that:
      1. It runs without error
      2. Output shape matches input shape
      3. No NaNs appear in the output (a common silent-failure mode
         with GATs on disconnected or oddly-weighted graphs)
    """
    print("Loading real adjacency matrix from graph_builder.py output...")
    with open(GRAPH_PATH, "rb") as f:
        graph_data = pickle.load(f)

    adj_matrix = torch.tensor(graph_data["adjacency_matrix"], dtype=torch.float32)
    num_skills = adj_matrix.shape[0]
    print(f"Adjacency matrix shape: {adj_matrix.shape}")

    edge_index, edge_weight = adjacency_to_edge_index(adj_matrix)
    print(f"Converted to edge_index with {edge_index.shape[1]} directed edges")

    emb_dim = 64
    dummy_embeddings = torch.randn(num_skills, emb_dim)

    encoder = GraphEncoder(emb_dim=emb_dim)
    output = encoder(dummy_embeddings, edge_index, edge_weight)

    print(f"\nInput shape:  {dummy_embeddings.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_embeddings.shape, \
        "Shape mismatch! GAT output should match input shape exactly."

    num_nans = torch.isnan(output).sum().item()
    print(f"NaN count in output: {num_nans}")

    if num_nans > 0:
        print("\nWARNING: NaNs detected. This usually means some skill node has "
              "ZERO edges (fully isolated in the graph) and GAT's attention "
              "softmax has nothing to normalize over. Check for isolated "
              "nodes before proceeding to model training.")
        isolated = (adj_matrix.sum(dim=1) == 0).sum().item()
        print(f"Isolated skills (no edges at all): {isolated} out of {num_skills}")
    else:
        print("\nNo NaNs. Shapes match. Graph encoder plumbing is working correctly.")


if __name__ == "__main__":
    run_sanity_test()