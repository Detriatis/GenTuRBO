# esm

Source: [esm](https://github.com/facebookresearch/esm).

Role: runtime.

ESM2 mean-pooled protein embeddings. fair-esm 2.0.0 is pinned by the protein extra and lock. Prefetch uses esm2_t33_650M_UR50D including its contact-regression weights.

Pinned distribution: `fair-esm==2.0.0`.

The source checkout is optional and ignored by Git. See the upstream LICENSE and notices at the pinned revision before redistributing source. This folder documents provenance; it does not replace upstream attribution.

Exact weight URLs, provider checksums and sizes are in
`src/genturbo/model_weights.json`. `setup --prefetch-models` verifies them
before use and records a SHA256 receipt in the Torch cache.
