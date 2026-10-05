# flexs

Source: [FLEXS](https://github.com/samsinai/FLEXS).

Role: adapted-source.

AdaLead semantics are adapted in the sequence/protein proposal policy and src/sokoban/adalead_official.py. The retained no-recombination variant uses threshold parents, root-anchored rollout and a fixed surrogate-query budget. No FLEXS runtime installation is needed.

Pinned commit: `dd409167b5575b51967d325b4e48aef3577a505f`. To inspect this exact version inside this folder:

```bash
git clone https://github.com/samsinai/FLEXS.git external/flexs/source
git -C external/flexs/source checkout --detach dd409167b5575b51967d325b4e48aef3577a505f
```

The source checkout is optional and ignored by Git. See the upstream LICENSE and notices at the pinned revision before redistributing source. This folder documents provenance; it does not replace upstream attribution.
