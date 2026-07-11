# Multi-node validation (split by parquet ids, deterministic)

Image smolmo-olmocore-fast:1.1+ ; SMOLMO_NO_SHUFFLE drops the internal reshuffle (order-only).

## C1 — split is lossless & order-correct
node0 = shards {0,1}, node1 = shards {2,3} (each SMOLMO_NO_SHUFFLE=1, node-prefixed parts) vs
single-node over shards {0,1,2,3} (SMOLMO_NO_SHUFFLE=1). Concatenated streams:
| stream | tokens | single sha | multi sha | identical |
|---|---|---|---|---|
| token_ids   | 163,132,814 | cefcb9655a66 | cefcb9655a66 | YES |
| labels_mask | 163,132,814 | eaee7e74a2a5 | eaee7e74a2a5 | YES |
=> multi-node(parquet-id split) concatenates BYTE-IDENTICAL to single-node. No seams/drift.

## C2 — dropping shuffle only reorders (same document set as stock)
3-shard no-shuffle run vs the stock shuffle-ON run (out_glob): document multiset (sha per doc) compared.
noshuf3 docs=9000, shuffle-on docs=9000, SAME_SET=True.
=> every sequence identical to stock; only sequence ORDER differs (olmo-core reshuffles at train time).

## Combined guarantee
Single-node byte-identical to stock (RESULTS.md) + C1 + C2 => every training sequence is byte-identical
to the stock olmo_thinker output in ANY node configuration; multi-node differs only in document order.
