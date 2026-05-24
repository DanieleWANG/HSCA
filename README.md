# HSCA

Official code for **HSCA (Heterogeneous Stock Conditioned Attention)**.

This repository implements the three coordinated mechanisms described in
Sec. 3 of the paper:

1. **Asymmetric Dual-Stream Encoder** (Sec. 3.1) — a Transformer encoder over
   per-stock lookback features, paired with an MLP-based state-stream encoder
   that produces *M* State Anchors.
2. **State-Conditioned Heterogeneous Attention** (Sec. 3.2–3.4) — a State Gate
   modulates the stock query subspace; State Anchors extend the key-value bank
   into a heterogeneous node set; an Alpha-Guard residual update stabilises
   the per-stock representation.
3. **Structural Hinge Loss** (Sec. 3.5) — direct margin-based supervision on
   the pre-softmax stock-stock affinity logits, lifted from the same
   cross-sectional return target as the prediction loss.

## File layout

| File | Role |
|---|---|
| `hsca_model.py` | `HSCAModel`, `HSCANetwork`, `StateConditionedHeteroAttention`, `StateAnchorEncoder`, `AsymmetricDualStream`. |
| `base_model.py` | `SequenceModel` training loop + Structural Hinge Loss assembly (γ = 0.5, default λ = 0.3). |
| `main_preprocessed.py` | Multi-seed training entry point. |
| `PreprocessedTimeSeriesDataset.py` | Daily-batched dataset wrapper. |

## Symbol map (paper → code)

| Paper | Code |
|---|---|
| `m_t` (macro snapshot) | `market_last` |
| `A_t` (State Anchors) | `A` (`[M, d]`) |
| `z_t = mean(A_t)` (Eq. 4) | `z` (`[d]`) |
| `g_t` (State Gate, Eq. 5) | `g` |
| `Q_t` (Eq. 6) | `q_mod` |
| `H_het` (Eq. 7) | `het_bank` |
| `S^{(stk)}` (Eq. 9, stock block) | `raw_scores_stk` (`[N, N]`) |
| `G_{t,i}` (Alpha-Guard, Eq. 12) | `alpha_gate` |
| `h̃_{t,i}` (Eq. 13) | `h_tilde` |
| `T_t` (Eq. 15) | `target_adj` |
| `L_struct` (Eq. 17) | `_structural_loss` |
| margin `γ = 0.5` | `HINGE_MARGIN` |
| weight `λ = 0.3` | `graph_loss_weight` / `--loss` |

## Data

> The CSI300 and CSI800 datasets can be obtained from
> [MASTER](https://github.com/SJTU-DMTai/MASTER) (`opensource` directory).
>
> For the S&P 500 dataset, refer to
> [SPF](https://github.com/kijeong22/ijcai2025-spf). Download
> `baseline_sp500.npy` and `sp500_index.csv`, then compute the Alpha158
> features and market-related indicators following the MASTER preprocessing
> pipeline. Refer to `data/csi_market_information.csv` in the MASTER
> repository for details.
>
> The SPF authors provide several preprocessed hypergraph datasets; these are
> not required for replication and can be safely ignored.

## Train

```bash
# Single market
python main_preprocessed.py --universe csi300 --seeds 0,1,2,3,4

# Both Chinese markets
python main_preprocessed.py --auto_train_both --seeds 0,1,2,3,4

# Disable Structural Hinge Loss (ablation)
python main_preprocessed.py --universe csi300 --no_hinge_loss

# Sweep hinge weight λ
python main_preprocessed.py --universe csi300 --loss 0.5
```
