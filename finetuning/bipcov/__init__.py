"""Bipartite Greedy Coverage (BipCov) baseline.

This method is designed to be *lightweight*:
  - Embeddings are produced externally (e.g., BGE/E5/Rep-Sim hidden states).
  - This module only performs selection/ranking given train/ref embeddings.

See `probe_bipcov_from_emb.py`.
"""
