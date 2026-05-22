"""LLM / fine-tuning oriented data selection utilities.

This subpackage is intentionally *standalone* (no OpenDataVal dependency),
so it can run on CPU for quick prototyping and later be swapped into
DATE-LM or other LLM pipelines.

The core abstraction is: given a training pool (instructions / examples)
and a small *reference / validation* set (target tasks), build a bipartite
graph using an embedding similarity threshold, then run greedy max-coverage
to produce a *selection sequence* and per-example scores.
"""
