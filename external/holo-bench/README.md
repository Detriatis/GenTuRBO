# holo-bench

Source: [holo-bench](https://github.com/prescient-design/holo-bench).

Role: adapted-source.

The Holo DiscreteEvolution algorithm is adapted in src/baselines/holo_genetic_algorithm.py. Upstream source: holo/optim/_discrete_evolution.py. The adaptation adds finite paid budgets and prepaid initial labels; no external Holo installation is needed.

Pinned commit: `4d94731017e63a73008c94a99a1bee87585bf9b0`. To inspect this exact version inside this folder:

```bash
git clone https://github.com/prescient-design/holo-bench.git external/holo-bench/source
git -C external/holo-bench/source checkout --detach 4d94731017e63a73008c94a99a1bee87585bf9b0
```

The source checkout is optional and ignored by Git. See the upstream LICENSE and notices at the pinned revision before redistributing source. This folder documents provenance; it does not replace upstream attribution.
