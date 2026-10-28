"""CEGG environment: two coupled graphs, Bass diffusion on the demand graph.

A demand node becomes operational only once it is placed and at least one of
its coupling partners on the supply graph has been placed as well; diffusion
then runs on the edges whose two endpoints are both operational.
"""
import numpy as np
import networkx as nx
import torch
import os
from collections import deque

DEVICE = torch.device('cpu')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data') + os.sep

class CoupledEnv:
    def __init__(self, G_D, G_S, phi, T=30, p=0.004, q=0.3, i=0.02,
                 carbon_saving=8500, c_hrs=1e6, c_hpp=5e5, c_transport=1e5):
        self.G_D = G_D
        self.G_S = G_S
        self.phi = phi  # demand_node -> [supply_nodes]
        self.phi_inv = {}  # supply_node -> [demand_nodes]
        for d, s_list in phi.items():
            for s in s_list:
                if s not in self.phi_inv:
                    self.phi_inv[s] = []
                self.phi_inv[s].append(d)

        self.nodes_D = sorted(list(G_D.nodes()))
        self.nodes_S = sorted(list(G_S.nodes()))
        self.edges_D = sorted(list(G_D.edges()))
        self.N_D = len(self.nodes_D)
        self.N_S = len(self.nodes_S)
        self.node_idx_D = {n: i for i, n in enumerate(self.nodes_D)}
        self.node_idx_S = {n: i for i, n in enumerate(self.nodes_S)}
        self.T = T
        self.p_bass = p; self.q_bass = q; self.growth = i
        self.carbon_saving = carbon_saving
        self.c_hrs = c_hrs; self.c_hpp = c_hpp; self.c_transport = c_transport
        self.weights = np.array([
            G_D[u][v].get('weight', G_D[u][v].get('trips', 1.0))
            for u, v in self.edges_D
        ], dtype=float)
        self.weights = np.maximum(self.weights, 1.0)
        self.reset()

    def reset(self):
        self.t = 0
        self.mask_D = np.zeros(self.N_D, dtype=int)
        self.mask_S = np.zeros(self.N_S, dtype=int)
        self.fcet = np.zeros(len(self.edges_D), dtype=float)
        return self._get_state()

    def _get_state(self):
        return (self.t, self.mask_D.copy(), self.mask_S.copy(), self.fcet.copy())

    def _is_operational(self, demand_node_idx):
        if self.mask_D[demand_node_idx] == 0:
            return False
        node = self.nodes_D[demand_node_idx]
        for s_node in self.phi.get(node, []):
            s_idx = self.node_idx_S.get(s_node)
            if s_idx is not None and self.mask_S[s_idx] == 1:
                return True
        return False

    def step(self, action_D, action_S):
        n_new_hrs = 0; n_new_hpp = 0
        for idx in action_D:
            if 0 <= idx < self.N_D and self.mask_D[idx] == 0:
                self.mask_D[idx] = 1; n_new_hrs += 1
        for idx in action_S:
            if 0 <= idx < self.N_S and self.mask_S[idx] == 0:
                self.mask_S[idx] = 1; n_new_hpp += 1
        cap_t = self.weights * ((1 + self.growth) ** self.t)
        for e_idx, (u, v) in enumerate(self.edges_D):
            u_idx = self.node_idx_D[u]
            v_idx = self.node_idx_D[v]
            if self._is_operational(u_idx) and self._is_operational(v_idx):
                prev = self.fcet[e_idx]; cap = cap_t[e_idx]
                if cap > 0:
                    self.fcet[e_idx] = prev + (self.p_bass + self.q_bass * prev / cap) * (cap - prev)
                    self.fcet[e_idx] = min(max(self.fcet[e_idx], 0), cap)
        total_fcet = self.fcet.sum()
        r1 = self.carbon_saving * total_fcet - self.c_hrs * n_new_hrs
        r2 = total_fcet * 1000 - self.c_hpp * n_new_hpp - self.c_transport * n_new_hpp
        self.t += 1
        done = (self.t >= self.T)
        return self._get_state(), r1, r2, done, {}

def create_supply_network(G_D, n_supply, seed=42):
    np.random.seed(seed)
    degrees = dict(G_D.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    supply_nodes = sorted_nodes[:n_supply]
    G_S = nx.Graph()
    for n in supply_nodes:
        G_S.add_node(n)
    for i, u in enumerate(supply_nodes):
        for j, v in enumerate(supply_nodes):
            if i < j:
                try:
                    dist = nx.shortest_path_length(G_D, u, v)
                    if dist <= 3:
                        G_S.add_edge(u, v, weight=1.0/dist)
                except nx.NetworkXNoPath:
                    pass
    if len(G_S) > 1 and not nx.is_connected(G_S):
        comps = list(nx.connected_components(G_S))
        for i in range(len(comps)-1):
            G_S.add_edge(list(comps[i])[0], list(comps[i+1])[0], weight=0.1)
    return G_S, supply_nodes

def create_coupling(G_D, supply_nodes, max_dist=2):
    phi = {}
    for d_node in G_D.nodes():
        phi[d_node] = []
        for s_node in supply_nodes:
            try:
                dist = nx.shortest_path_length(G_D, d_node, s_node)
                if dist <= max_dist:
                    phi[d_node].append(s_node)
            except nx.NetworkXNoPath:
                pass
    return phi

def build_state_tensor(G, mask):
    nodes = sorted(G.nodes())
    n = len(nodes)
    features = []
    for i, node in enumerate(nodes):
        features.append([float(mask[i]), G.degree(node) / max(1, n), float(i) / n])
    x = torch.FloatTensor(features).to(DEVICE)
    adj = torch.zeros(n, n).to(DEVICE)
    node_idx = {node: i for i, node in enumerate(nodes)}
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            adj[node_idx[u], node_idx[v]] = 1.0
            if not G.is_directed():
                adj[node_idx[v], node_idx[u]] = 1.0
    return x, adj

def build_coupling_matrix(env):
    """Build Phi matrix: (N_D, N_S) binary matrix."""
    C = torch.zeros(env.N_D, env.N_S).to(DEVICE)
    for d_node in env.nodes_D:
        d_idx = env.node_idx_D[d_node]
        for s_node in env.phi.get(d_node, []):
            s_idx = env.node_idx_S.get(s_node)
            if s_idx is not None:
                C[d_idx, s_idx] = 1.0
    return C

def build_coupling_matrix_inv(env):
    """Build Phi_inv matrix: (N_S, N_D) binary matrix."""
    C = torch.zeros(env.N_S, env.N_D).to(DEVICE)
    for s_node in env.nodes_S:
        s_idx = env.node_idx_S[s_node]
        for d_node in env.phi_inv.get(s_node, []):
            d_idx = env.node_idx_D.get(d_node)
            if d_idx is not None:
                C[s_idx, d_idx] = 1.0
    return C

def load_tntp_raw(filename):
    G = nx.DiGraph()
    filepath = filename if os.path.isabs(filename) else DATA_DIR + filename
    if not os.path.exists(filepath): return None
    with open(filepath, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.startswith('~'): in_data = True; continue
            if not in_data or not line: continue
            parts = line.replace(';', '').split()
            if len(parts) >= 5:
                try:
                    o = int(parts[0]); d = int(parts[1])
                    cap = float(parts[2]); flow = float(parts[4]) if len(parts) > 4 else cap
                    G.add_edge(o, d, weight=max(flow, cap, 1.0), trips=max(flow, cap, 1.0))
                except (ValueError, IndexError): continue
    return nx.convert_node_labels_to_integers(G)

def bfs_subsample(G, max_nodes):
    if G.number_of_nodes() <= max_nodes:
        return nx.convert_node_labels_to_integers(G.copy())
    und = G.to_undirected()
    degs = dict(G.degree())
    start = max(degs, key=degs.get)
    visited = {start}; q = deque([start])
    while q and len(visited) < max_nodes:
        u = q.popleft()
        for v in und.neighbors(u):
            if v not in visited:
                visited.add(v)
                if len(visited) >= max_nodes: break
                q.append(v)
    sub = G.subgraph(visited).copy()
    return nx.convert_node_labels_to_integers(sub)
