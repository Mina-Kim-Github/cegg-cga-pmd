# Network data

The six `.tntp` files in this directory are road networks from the
**Transportation Networks for Research** repository, redistributed here so that
the experiments run without a network connection.

- Source: https://github.com/bstabler/TransportationNetworks

## Terms

The Transportation Networks for Research project states that its data sets are
for academic research purposes only, and that users must indicate the source of
any data set they use in any publication that relies on it. The project is not
responsible for the content or correctness of the data sets.

**This repository's MIT license covers our code only. The `.tntp` files keep the
terms of their own project.**

## How to cite

The project asks to be cited as:

> Transportation Networks for Research Core Team. *Transportation Networks for
> Research*. https://github.com/bstabler/TransportationNetworks.
> Accessed April 23, 2026.

| File | Network | Nodes used | Role in the paper |
|---|---|---|---|
| `ChicagoSketch_net.tntp` | Chicago Sketch | BFS subgraphs, N = 50–300 | controlled scaling sweep, coupling sweep, reward-domain control |
| `SiouxFalls_net.tntp` | Sioux Falls | 24 | real-network evaluation |
| `EMA_net.tntp` | Eastern Massachusetts | 74 | real-network evaluation |
| `Friedrichshain_net.tntp` | Berlin–Friedrichshain | 224 | real-network evaluation, physical-distance coupling |
| `Tiergarten_net.tntp` | Berlin–Tiergarten | 359 | real-network evaluation |
| `BerlinMitte_net.tntp` | Berlin–Mitte | 397 | real-network evaluation |

Node counts are the nodes that appear in the link table; isolated nodes listed
in the file header are not part of the graph the loader builds.
