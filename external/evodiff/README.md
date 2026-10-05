# evodiff

Source: [evodiff](https://github.com/microsoft/evodiff).

Role: runtime.

Frozen D3PM uniform 38M sequence generator. Installed by setup --protein; model weights are downloaded separately by --prefetch-models.

Pinned commit: `33206e99446f799ec11cf6e57d66ffaec837be91`. To inspect this exact version inside this folder:

```bash
git clone https://github.com/microsoft/evodiff.git external/evodiff/source
git -C external/evodiff/source checkout --detach 33206e99446f799ec11cf6e57d66ffaec837be91
```

The source checkout is optional and ignored by Git. See the upstream LICENSE and notices at the pinned revision before redistributing source. This folder documents provenance; it does not replace upstream attribution.

Exact weight URLs, provider checksums and sizes are in
`src/genturbo/model_weights.json`. `setup --prefetch-models` verifies them
before use and records a SHA256 receipt in the Torch cache.
