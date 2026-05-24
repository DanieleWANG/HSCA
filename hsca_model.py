"""HSCA: Heterogeneous Stock Conditioned Attention.

Paper-aligned reference implementation. Layout follows Sec. 3 of the paper:

- ``AsymmetricDualStream`` (Sec. 3.1)        : stock stream + state stream encoders.
- ``StateAnchorEncoder``  (Sec. 3.1, Eq. 3)  : MLP_state producing M state anchors.
- ``StateConditionedHeteroAttention``        : Sec. 3.2 (State Gate) + Sec. 3.3
                                               (State-Anchored Heterogeneous Attention)
                                               + Sec. 3.4 (Alpha-Guard).
- ``HSCANetwork``                            : full forward graph; returns pred and
                                               the pre-softmax stock-stock affinity
                                               block used by the Structural Hinge
                                               Loss (Sec. 3.5, applied in base_model).
- ``HSCAModel``                              : SequenceModel wrapper (training loop).

Symbol map (paper -> code):
  m_t            -> ``market_last``
  A_t            -> ``A`` (state anchors, [M, d])
  z_t            -> ``z`` (state context, [d]); paper Eq. 4: z = mean(A)
  g_t            -> ``g`` (State Gate output, [d]); paper Eq. 5
  Q_t            -> ``q``; paper Eq. 6
  H_het          -> ``het_bank``; paper Eq. 7
  S^(stk)        -> ``raw_scores_stk``; paper Eq. 9 (stock-stock block)
  Alpha-Guard G  -> ``alpha_gate``; paper Eq. 12
  h_tilde        -> ``h_tilde``; paper Eq. 13
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import SequenceModel


# =============================================================================
# Temporal backbone (Sec. 3.1, stock stream)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class TemporalTransformerBlock(nn.Module):
    """Single Transformer encoder block (multi-head self-attention + FFN).

    Implements the Transformer encoder applied to the per-stock lookback window
    in Sec. 3.1 (Eq. 2).
    """

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList()
        if dropout > 0:
            for _ in range(nhead):
                self.attn_dropout.append(nn.Dropout(p=dropout))

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = self.d_model // self.nhead
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            attn_h = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if len(self.attn_dropout) > 0:
                attn_h = self.attn_dropout[i](attn_h)
            att_output.append(torch.matmul(attn_h, vh))
        att_output = torch.cat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (paper ref [28])."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# =============================================================================
# Sec. 3.1, Eq. 3 :  State Anchor Encoder
#
#     A_t = Reshape(MLP_state(m_t)),  with MLP_state : R^{F_m} -> R^{M*d}
#
# A single MLP maps the macro snapshot m_t to M*d activations, which are then
# reshaped into M anchor vectors. This matches the paper exactly. The previous
# implementation used M parallel encoders over disjoint slices of m_t; that is
# replaced here.
# =============================================================================

class StateAnchorEncoder(nn.Module):
    """MLP_state producing ``num_anchors`` state anchor vectors of dim ``d_model``.

    Parameters
    ----------
    state_dim : int
        Macro feature dimension F_m (e.g. 21 for one broad index, 42 for two).
    d_model : int
        Per-anchor dimension d.
    num_anchors : int
        Number of state anchors M (= number of broad-index components in X^(m)).
    hidden : int, optional
        Hidden width of MLP_state. Defaults to ``d_model``.
    """

    def __init__(self, state_dim, d_model, num_anchors, hidden=None):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model
        self.num_anchors = num_anchors
        hidden = hidden or d_model
        self.mlp_state = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_anchors * d_model),
        )

    def forward(self, m):
        """Compute A_t.

        Parameters
        ----------
        m : torch.Tensor
            Macro snapshot, shape ``[F_m]`` or ``[1, F_m]``.

        Returns
        -------
        A : torch.Tensor
            State anchors, shape ``[M, d]``.
        """
        if m.dim() == 2:
            m = m.squeeze(0)
        flat = self.mlp_state(m)  # [M*d]
        return flat.view(self.num_anchors, self.d_model)


# =============================================================================
# Sec. 3.2-3.4 :  State-Conditioned Heterogeneous Attention with Alpha-Guard
#
#   State Gate (Eq. 5-6):  g_t = sigma(W_g z_t + b_g)
#                          Q_t = (h_t W_Q) (.) g_t
#   Heterogeneous bank (Eq. 7): H_het = [A_t ; h_t]
#   Pre-softmax affinity (Eq. 9): S^(h)_{ij} = <q_i, k_j> / sqrt(d_h)
#   Alpha-Guard (Eq. 12-13):
#       G = sigma(W_G [h || o] + b_G)
#       h_tilde = RMSNorm(h + G (.) o)
# =============================================================================

class StateConditionedHeteroAttention(nn.Module):
    """State-conditioned heterogeneous attention with Alpha-Guard residual.

    Parameters
    ----------
    d_model : int
    nhead : int
    dropout : float
    use_alpha_guard : bool
        Enable the Alpha-Guard gated residual + RMSNorm (Sec. 3.4).
    use_state_anchor : bool
        Inject the M state anchors as virtual key-value nodes (Sec. 3.3, Eq. 7).
    use_state_gate : bool
        Apply the State Gate multiplicative modulation on Q (Sec. 3.2, Eq. 5-6).
    use_zero_init : bool
        Zero-initialise the output projection ``W_O`` for residual stability
        at training start. (Implementation detail; not paper-specified.)
    """

    def __init__(self, d_model, nhead=4, dropout=0.1,
                 use_alpha_guard=True, use_state_anchor=True,
                 use_state_gate=True, use_zero_init=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_alpha_guard = use_alpha_guard
        self.use_state_anchor = use_state_anchor
        self.use_state_gate = use_state_gate

        # Q, K, V projections
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # State Gate: g_t = sigma(W_g z_t + b_g),  W_g : R^d -> R^d
        # Paper Eq. 5 (no concat: input is z_t = mean(A_t), already in R^d).
        if self.use_state_gate:
            self.state_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )

        # Alpha-Guard: G = sigma(W_G [h || o] + b_G);  h_tilde = RMSNorm(h + G * o)
        if self.use_alpha_guard:
            self.alpha_guard_gate = nn.Linear(d_model * 2, d_model)
            self.alpha_guard_norm = RMSNorm(d_model)

        # Output projection W_O (Eq. 11)
        self.out_proj = nn.Linear(d_model, d_model)
        if use_zero_init:
            nn.init.constant_(self.out_proj.weight, 0)
            nn.init.constant_(self.out_proj.bias, 0)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, stock_feats, A, z):
        """Forward pass.

        Parameters
        ----------
        stock_feats : torch.Tensor
            Per-stock embeddings h_t, shape ``[N, d]``.
        A : torch.Tensor or None
            State anchors A_t, shape ``[M, d]``. Required when
            ``use_state_anchor=True``.
        z : torch.Tensor or None
            State context z_t = mean(A_t), shape ``[d]``. Required when
            ``use_state_gate=True``.

        Returns
        -------
        h_tilde : torch.Tensor
            Updated stock embeddings, shape ``[N, d]``.
        raw_scores_stk : torch.Tensor
            Head-averaged pre-softmax stock-stock affinity logits S^(stk),
            shape ``[N, N]``. Consumed by the Structural Hinge Loss.
        gate_mean : float
            Mean Alpha-Guard activation (diagnostic; 0.0 if disabled).
        attn_stk : torch.Tensor
            Head-averaged stock-stock attention weights, shape ``[N, N]``.
            (Anchor columns excluded.)
        """
        N, D = stock_feats.shape

        # ---- Eq. 7 : heterogeneous bank H_het = [A ; h] ----
        if self.use_state_anchor:
            assert A is not None, "use_state_anchor=True requires A."
            M = A.shape[0]
            het_bank = torch.cat([A, stock_feats], dim=0)  # [M+N, d]
        else:
            M = 0
            het_bank = stock_feats

        # ---- Eq. 5-6 : State Gate + state-modulated query ----
        q_lin = self.to_q(stock_feats)  # [N, d]
        if self.use_state_gate:
            assert z is not None, "use_state_gate=True requires z."
            g = self.state_gate(z)            # [d]
            q_mod = q_lin * g.unsqueeze(0)    # broadcast over N
        else:
            q_mod = q_lin

        # ---- Multi-head split ----
        q = q_mod.view(N, self.nhead, self.head_dim)
        k = self.to_k(het_bank).view(M + N, self.nhead, self.head_dim)
        v = self.to_v(het_bank).view(M + N, self.nhead, self.head_dim)

        # ---- Eq. 9 : pre-softmax affinity S^(h) ----
        scores = torch.einsum('nhd,mhd->nhm', q, k) * self.scale  # [N, H, M+N]
        raw_scores_full = scores.mean(dim=1)                      # [N, M+N], paper Eq. 16

        # Stock-stock block: discard the M anchor columns. This is the
        # quantity supervised by the Structural Hinge Loss (Sec. 3.5).
        raw_scores_stk = raw_scores_full[:, M:]                   # [N, N]

        # ---- Attention weights + value aggregation (Eq. 11) ----
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.einsum('nhm,mhd->nhd', attn, v).contiguous().view(N, D)
        output = self.out_proj(output)

        # ---- Eq. 12-13 : Alpha-Guard ----
        if self.use_alpha_guard:
            alpha_gate = torch.sigmoid(self.alpha_guard_gate(torch.cat([stock_feats, output], dim=-1)))
            h_tilde = self.alpha_guard_norm(stock_feats + alpha_gate * output)
            gate_mean = alpha_gate.mean().item()
        else:
            h_tilde = output
            gate_mean = 0.0

        attn_mean = attn.mean(dim=1)            # [N, M+N]
        attn_stk = attn_mean[:, M:]             # [N, N]
        return h_tilde, raw_scores_stk, gate_mean, attn_stk


# =============================================================================
# Sec. 3.1 :  Asymmetric Dual-Stream Encoder
#
# Stock stream: feature projection (Eq. 1) -> Transformer encoder (Eq. 2)
#               -> last-step state h_{t,i}
# State stream: macro snapshot m_t -> StateAnchorEncoder (Eq. 3) -> A_t
#               -> z_t = mean(A_t) (Eq. 4)
# =============================================================================

class AsymmetricDualStream(nn.Module):
    """The pair of encoders described in Sec. 3.1, exposed as a single module.

    The stock stream and the state stream have different architectures
    (a Transformer over a lookback window vs. an MLP on a snapshot), hence
    "asymmetric". This module owns both streams and exposes their outputs.
    """

    def __init__(self, d_feat, d_model, state_dim, num_anchors, dropout=0.1, nhead=4):
        super().__init__()
        # Stock stream (Eq. 1-2)
        self.feature_projection = nn.Sequential(
            nn.Linear(d_feat, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.Sequential(
            PositionalEncoding(d_model),
            TemporalTransformerBlock(d_model=d_model, nhead=nhead, dropout=dropout),
        )

        # State stream (Eq. 3-4)
        self.state_encoder = StateAnchorEncoder(
            state_dim=state_dim,
            d_model=d_model,
            num_anchors=num_anchors,
        )

    def forward(self, stock_feats_seq, market_last):
        """Forward both streams.

        Parameters
        ----------
        stock_feats_seq : torch.Tensor
            Per-stock raw lookback features, shape ``[N, L, F_s]``.
        market_last : torch.Tensor
            Macro snapshot m_t at the lookback end, shape ``[F_m]``.

        Returns
        -------
        h : torch.Tensor    Last-step stock embeddings, shape ``[N, d]``.
        A : torch.Tensor    State anchors A_t, shape ``[M, d]``.
        z : torch.Tensor    State context z_t = mean(A_t), shape ``[d]``.
        """
        # Stock stream
        e = self.feature_projection(stock_feats_seq)   # [N, L, d]
        H = self.temporal_encoder(e)                   # [N, L, d]
        h = H[:, -1, :]                                # [N, d]

        # State stream
        A = self.state_encoder(market_last)            # [M, d]
        z = A.mean(dim=0)                              # [d]   <-- Eq. 4 mean pool
        return h, A, z


# =============================================================================
# HSCA Network (full forward graph)
# =============================================================================

# Per-universe defaults: (state_dim, num_anchors)
#  - sp500            : single broad index (21 dims) -> M=1
#  - csi300 (default) : two broad indices (z300+z500, 42 dims) -> M=2
#  - csi300 + use_single_index : ablation toggle, drop z500 -> M=1
#  - csi800           : two broad indices -> M=2
_UNIVERSE_DEFAULTS = {
    'sp500':  (21, 1),
    'csi300': (42, 2),
    'csi800': (42, 2),
}


class HSCANetwork(nn.Module):
    """HSCA: Heterogeneous Stock Conditioned Attention.

    Implements the three coordinated mechanisms of Sec. 3:
      (1) Asymmetric Dual-Stream Encoder (Sec. 3.1)
      (2) State-Conditioned Heterogeneous Attention with Alpha-Guard
          (Sec. 3.2-3.4)
      (3) The Structural Hinge Loss (Sec. 3.5) is computed externally during
          training in ``base_model.SequenceModel.train_epoch``; this network
          exposes the pre-softmax stock-stock affinity block it consumes.

    Parameters
    ----------
    d_feat : int
        Per-timestep stock feature dimension F_s (typically 158, Alpha158).
    d_model : int
        Hidden dimension d.
    dropout : float
    universe : {'csi300', 'csi800', 'sp500'} or None
    use_single_index : bool
        CSI300 only. When True, use only the first 21 macro dims and set M=1
        (ablation knob; default operating mode for CSI300 is M=2).
    """

    def __init__(self, d_feat, d_model, dropout=0.1,
                 universe=None, use_single_index=False, nhead=4):
        super().__init__()
        self.universe = universe
        self.use_single_index = use_single_index

        # Resolve (state_dim, num_anchors) from universe
        if universe == 'csi300' and use_single_index:
            self.state_dim, self.num_anchors = 21, 1
        elif universe in _UNIVERSE_DEFAULTS:
            self.state_dim, self.num_anchors = _UNIVERSE_DEFAULTS[universe]
        else:
            # Fallback for unknown universe: assume two-index composite
            self.state_dim, self.num_anchors = 42, 2

        # (1) Asymmetric Dual-Stream Encoder
        self.dual_stream = AsymmetricDualStream(
            d_feat=d_feat,
            d_model=d_model,
            state_dim=self.state_dim,
            num_anchors=self.num_anchors,
            dropout=dropout,
            nhead=nhead,
        )

        # (2) State-Conditioned Heterogeneous Attention
        self.attention = StateConditionedHeteroAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            use_alpha_guard=True,
            use_state_anchor=True,
            use_state_gate=True,
            use_zero_init=True,
        )

        # Prediction head: y_hat = MLP(h_tilde)  (Eq. 14)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _slice_market(self, market_full):
        """Extract m_t from the raw market block.

        The market block per stock has shape ``[L, 63]`` (3 broad indices x 21
        rolling-stat dims). We take the lookback end (``t``) and the first
        ``state_dim`` dims of the macro feature block.
        """
        # market_full: [N, L, 63]; all stocks share the same market vector,
        # so index 0 along the stock axis is the canonical row.
        return market_full[0, -1, :self.state_dim]  # [F_m]

    def forward(self, x):
        # x: [N, L, F_s + 63]   (stock features concatenated with macro features)
        stock_feats_seq = x[..., :-63]     # [N, L, F_s]
        market_full = x[..., -63:]         # [N, L, 63]
        market_last = self._slice_market(market_full)  # [F_m]

        # (1) Encode both streams
        h, A, z = self.dual_stream(stock_feats_seq, market_last)

        # (2) State-conditioned heterogeneous attention
        h_tilde, raw_scores_stk, gate_mean, attn_stk = self.attention(h, A, z)

        # (3) Prediction head
        pred = self.prediction_head(h_tilde).squeeze(-1)

        debug_stats = {'alpha_gate_mean': gate_mean}
        return pred, raw_scores_stk, debug_stats, attn_stk


# =============================================================================
# Training-loop wrapper
# =============================================================================

class HSCAModel(SequenceModel):
    """HSCA model wrapper (training loop, eval, loss assembly).

    The Structural Hinge Loss (Sec. 3.5, Eq. 17) is assembled inside
    ``SequenceModel.train_epoch`` from the ``raw_scores_stk`` returned by
    :class:`HSCANetwork`. The hinge margin ``gamma`` and the hinge weight
    ``lambda`` are configured there.

    Parameters
    ----------
    d_feat : int
    d_model : int
    use_hinge_loss : bool
        If False, fall back to an MSE surrogate on the affinity block
        (ablation).
    universe : {'csi300', 'csi800', 'sp500'}
    use_single_index : bool
        CSI300-only ablation toggle (M=1 instead of M=2).
    """

    def __init__(self, d_feat, d_model, use_hinge_loss=True,
                 universe=None, use_single_index=False, **kwargs):
        super().__init__(use_hinge_loss=use_hinge_loss, **kwargs)
        self.model = HSCANetwork(
            d_feat=d_feat,
            d_model=d_model,
            universe=universe,
            use_single_index=use_single_index,
        )
        self.use_hinge_loss = use_hinge_loss
        self.init_model()

    def forward(self, x):
        pred, _, _, _ = self.model(x)
        return pred
