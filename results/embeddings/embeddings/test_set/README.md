# Test-Set Structure Embeddings

This folder contains fixed test-set structure embeddings from the local pretrained ALIGNN model.

Files:
- `structure_embeddings.npz`: compressed embedding matrices keyed by embedding source.
- `structure_embedding_metadata.csv`: one row per structure per embedding source.
- `structure_embedding_metadata.parquet`: written only when a parquet engine is installed.

Embedding sources:
- `pre_head`: shape `[1726, 256]`
- `last_alignn_pool`: shape `[1726, 256]`
- `last_gcn_pool`: shape `[1726, 256]`
