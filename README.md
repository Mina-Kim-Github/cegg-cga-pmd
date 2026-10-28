# CGA-PMD

Reference implementation of **CGA-PMD** from the NeurIPS 2026 paper
*Coupling-Aware Reinforcement Learning for Co-Evolving Graph Games*
(Mina Kim, Guanghui Lan, Benoit Montreuil).

A **Co-Evolving Graph Game (CEGG)** has two agents expanding two separate graphs:
a demand node starts diffusing only once it is placed *and* a structurally coupled
supply node has been placed as well. CGA-PMD pairs Φ-masked cross-graph attention
with a deep-RL realization of policy mirror descent in dual-averaging form.

Runs on CPU. No GPU required.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10.13.

## Contents

```
cegg/env.py     the CEGG environment, supply-graph and coupling construction,
                TNTP loader, BFS subsampling
cegg/model.py   GraphSAGE encoders, Φ-masked cross-graph attention,
                the CGA advantage network
cegg/train.py   train_cga_pmd (Algorithm 1 of the paper) and greedy evaluation
data/           the six TNTP road networks used in the paper
```

This release contains the method itself. The baselines, the experiment drivers
and the result files behind the paper's tables are not included.

## Use

```python
from cegg import (load_tntp_raw, bfs_subsample, create_supply_network,
                  create_coupling, train_cga_pmd, evaluate_cga_pmd)

env_params = dict(T=30, p=0.004, q=0.3, i=0.02,
                  carbon_saving=8500, c_hrs=1e6, c_hpp=5e5, c_transport=1e5)

G_D = bfs_subsample(load_tntp_raw('ChicagoSketch_net.tntp'), 50)
G_S, supply = create_supply_network(G_D, 16, seed=42)
phi = create_coupling(G_D, supply, max_dist=2)          # Φ: demand → supply partners

psi_D, psi_S, episodes = train_cga_pmd(G_D, G_S, phi, env_params)
reward, _ = evaluate_cga_pmd(psi_D, psi_S, G_D, G_S, phi, env_params)
```

`train_cga_pmd` follows Algorithm 1: it samples a joint rollout from
softmax(η_k ψ̃) with η_k = β_k/λ_k, β_k = k+1 and λ_k = λ(k+1)^{3/2}, then takes
one Adam step on the w_k-weighted policy gradient plus the return-regression term.
Training stops on a patience of 100 episodes, at most 1000.

## Citation

```bibtex
@inproceedings{kim2026coupling,
  title     = {Coupling-Aware Reinforcement Learning for Co-Evolving Graph Games},
  author    = {Kim, Mina and Lan, Guanghui and Montreuil, Benoit},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT for the code in this repository. The `.tntp` network files come from the
Transportation Networks for Research project and keep their own terms; see
`data/README.md`.
