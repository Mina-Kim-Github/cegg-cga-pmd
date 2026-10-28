"""CGA: GraphSAGE encoders with Phi-masked cross-graph attention.

Each demand node attends only to the supply nodes structurally coupled to it,
so the attention softmax is normalised over Phi(v) rather than over all of V2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cpu')

class GraphSAGELayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f * 2, out_f)
    def forward(self, x, adj):
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        neigh = torch.mm(adj, x) / deg
        return F.relu(self.linear(torch.cat([x, neigh], dim=1)))

class CrossGraphAttention(nn.Module):
    """
    Cross-Graph Attention via coupling function Phi.

    For each node v in the "self" network, attends to the coupled nodes
    in the "other" network (given by Phi or Phi_inv), producing a
    cross-network context vector that is concatenated with the node's
    own GNN embedding.

    This is what makes the agent AWARE of the other network's state.
    """
    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Query from self-network node embedding
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        # Key and Value from other-network node features
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        # Output projection: combines self-embedding with cross-attention output
        self.W_o = nn.Linear(hidden_dim * 2, hidden_dim)
        # Learnable "no coupling" vector for nodes with no Phi connections
        self.no_coupling = nn.Parameter(torch.randn(hidden_dim) * 0.01)

    def forward(self, h_self, h_other, coupling_matrix):
        """
        Args:
            h_self:  (N_self, hidden_dim) — GNN embeddings of self-network
            h_other: (N_other, hidden_dim) — GNN embeddings of other-network
            coupling_matrix: (N_self, N_other) — binary, 1 if Phi(v) contains u
        Returns:
            h_cross: (N_self, hidden_dim) — coupling-aware embeddings
        """
        N_self = h_self.size(0)
        N_other = h_other.size(0)

        Q = self.W_q(h_self)   # (N_self, hidden_dim)
        K = self.W_k(h_other)  # (N_other, hidden_dim)
        V = self.W_v(h_other)  # (N_other, hidden_dim)

        # Reshape for multi-head attention
        Q = Q.view(N_self, self.n_heads, self.head_dim).transpose(0, 1)   # (heads, N_self, head_dim)
        K = K.view(N_other, self.n_heads, self.head_dim).transpose(0, 1)  # (heads, N_other, head_dim)
        V = V.view(N_other, self.n_heads, self.head_dim).transpose(0, 1)  # (heads, N_other, head_dim)

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)  # (heads, N_self, N_other)

        # Mask: only attend to nodes connected via Phi
        mask = coupling_matrix.unsqueeze(0).expand(self.n_heads, -1, -1)  # (heads, N_self, N_other)
        scores = scores.masked_fill(mask == 0, -1e9)

        # Check which self-nodes have ANY coupling
        has_coupling = coupling_matrix.sum(dim=1) > 0  # (N_self,)

        attn = F.softmax(scores, dim=-1)  # (heads, N_self, N_other)
        # Zero out attention for nodes with no coupling (softmax gave uniform over -1e9)
        attn = attn * mask.float().clamp(max=1)

        context = torch.bmm(attn, V)  # (heads, N_self, head_dim)
        context = context.transpose(0, 1).contiguous().view(N_self, self.hidden_dim)  # (N_self, hidden_dim)

        # For nodes with no coupling, use learned default
        for i in range(N_self):
            if not has_coupling[i]:
                context[i] = self.no_coupling

        # Combine self-embedding with cross-network context
        h_cross = F.relu(self.W_o(torch.cat([h_self, context], dim=1)))

        return h_cross

class CGAAdvantageNet(nn.Module):
    """
    CGA-based advantage function approximator ψ(s,a;θ).
    Uses cross-graph attention to observe the other network via Phi.
    Outputs per-node advantage scores.
    """
    def __init__(self, in_f_self, in_f_other, hidden=64, n_layers=3, n_heads=4):
        super().__init__()
        # Own network GNN
        self.gnn_self = nn.ModuleList()
        self.gnn_self.append(GraphSAGELayer(in_f_self, hidden))
        for _ in range(n_layers - 1):
            self.gnn_self.append(GraphSAGELayer(hidden, hidden))
        # Other network GNN (lightweight)
        self.gnn_other = nn.ModuleList()
        self.gnn_other.append(GraphSAGELayer(in_f_other, hidden))
        self.gnn_other.append(GraphSAGELayer(hidden, hidden))
        # Cross-Graph Attention
        self.cga = CrossGraphAttention(hidden, n_heads=n_heads)
        # Advantage head
        self.adv_head = nn.Linear(hidden, 1)
        # Value head (for baseline subtraction)
        self.val_head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x_self, adj_self, x_other, adj_other, coupling_matrix):
        h = x_self
        for layer in self.gnn_self:
            h = layer(h, adj_self)
        h_other = x_other
        for layer in self.gnn_other:
            h_other = layer(h_other, adj_other)
        h_cross = self.cga(h, h_other, coupling_matrix)
        adv_scores = self.adv_head(h_cross).squeeze(-1)  # per-node advantages
        value = self.val_head(h_cross.mean(dim=0))         # state value
        return adv_scores, value
