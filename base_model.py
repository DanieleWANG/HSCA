import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler


# =============================================================================
# Utilities
# =============================================================================

def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


def zscore(x):
    return (x - x.mean()).div(x.std())


def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025 * N)
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]


def drop_na(x):
    mask = ~x.isnan()
    return mask, x[mask]


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # Calculate number of samples in each daily batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


# =============================================================================
# SequenceModel: training loop + loss assembly
#
# Loss = MSE(pred, label) + lambda * L_struct
#
# L_struct (Structural Hinge Loss, paper Sec. 3.5, Eq. 17):
#     T_{ij} = sign(y_i * y_j)                            in {-1, +1}
#     L_struct = mean_{i != j} max(0, gamma - T_{ij} * S_{ij})
#
# where S = raw_scores_stk is the head-averaged pre-softmax stock-stock
# affinity block returned by HSCANetwork.
# =============================================================================

# Paper Sec. 3.5 / Sec. 4.4: gamma = 0.5 is treated as a structural constant
# (bounds the exp(2*gamma) softmax suppression ratio at ~2.72 under the
# sqrt(d_h)-scaled affinity).
HINGE_MARGIN = 0.5


class SequenceModel:
    def __init__(self, n_epochs, lr, GPU=None, seed=None,
                 train_stop_loss_thred=None, save_path='model/', save_prefix='',
                 use_hinge_loss=True, graph_loss_weight=0.3):
        self.n_epochs = n_epochs
        self.lr = lr
        if torch.cuda.is_available() and GPU is not None:
            self.device = torch.device(f"cuda:{GPU}")
        else:
            self.device = torch.device("cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.use_hinge_loss = use_hinge_loss
        self.graph_loss_weight = graph_loss_weight  # lambda in paper

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model.to(self.device)

        # Scheduler: 5-epoch linear warmup, then cosine annealing (paper Sec. 4).
        warmup_epochs = 5
        if self.n_epochs > warmup_epochs:
            scheduler1 = optim.lr_scheduler.LinearLR(
                self.train_optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            eta_min = self.lr * 0.35
            scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
                self.train_optimizer, T_max=self.n_epochs - warmup_epochs, eta_min=eta_min
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.train_optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs]
            )
        else:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.train_optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.n_epochs
            )

    # ---------------------------------------------------------------------
    # Loss components
    # ---------------------------------------------------------------------

    def loss_fn(self, pred, label):
        """Cross-sectional MSE on valid (non-NaN, finite) entries."""
        if pred.dim() > 1:
            pred = pred.squeeze()
        if label.dim() > 1:
            label = label.squeeze()
        if pred.device != label.device:
            label = label.to(pred.device)

        mask = ~torch.isnan(label)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_masked = pred[mask]
        label_masked = label[mask]

        valid_mask = torch.isfinite(pred_masked) & torch.isfinite(label_masked)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_masked = pred_masked[valid_mask]
        label_masked = label_masked[valid_mask]

        mse_loss = torch.mean((pred_masked - label_masked) ** 2)
        return torch.clamp(mse_loss, 0.0, 1e6)

    def _structural_loss(self, raw_scores_stk, label):
        """Compute L_struct on the stock-stock pre-softmax affinity block.

        Parameters
        ----------
        raw_scores_stk : torch.Tensor
            Head-averaged S^(stk), shape ``[N, N]`` (paper Eq. 16).
        label : torch.Tensor
            CSZ-normalised future returns y_{t+tau}, shape ``[N]``.

        Returns
        -------
        torch.Tensor
            Scalar loss; zero tensor if no valid entries.
        """
        # Eq. 15 : T_{ij} = sign(y_i * y_j) in {-1, +1}
        y = label.view(-1, 1)
        target_adj = torch.sign(y @ y.t()).to(raw_scores_stk.device)

        if raw_scores_stk.shape != target_adj.shape:
            return torch.tensor(0.0, device=raw_scores_stk.device, requires_grad=True)

        valid = torch.isfinite(raw_scores_stk) & torch.isfinite(target_adj)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=raw_scores_stk.device, requires_grad=True)

        t_valid = target_adj[valid]
        s_valid = raw_scores_stk[valid]

        if self.use_hinge_loss:
            # Eq. 17 : L_struct = mean max(0, gamma - T * S)
            hinge = F.relu(HINGE_MARGIN - t_valid * s_valid)
            graph_loss = hinge.mean()
        else:
            # Ablation: MSE on sigmoid(S) against {0, 1} target.
            t_mse = (t_valid + 1) / 2.0
            s_mse = torch.sigmoid(s_valid)
            graph_loss = ((s_mse - t_mse) ** 2).mean()

        return torch.clamp(graph_loss, 0.0, 1e6)

    # ---------------------------------------------------------------------
    # Training / evaluation loops
    # ---------------------------------------------------------------------

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            # DataLoader wraps batch as (1, N, T, F); squeeze to (N, T, F).
            data = torch.squeeze(data, dim=0)

            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # Extreme label trimming + CSZ-Score normalisation. Required when
            # using the opensource preprocessed data, which has not been
            # trimmed upstream.
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label)

            model_output = self.model(feature.float())

            # HSCANetwork returns (pred, raw_scores_stk, debug_stats, attn_stk).
            # Other backbones may return just pred or a shorter tuple.
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                pred = model_output[0]
                raw_scores_stk = model_output[1]

                main_loss = self.loss_fn(pred, label)

                # raw_scores_stk should already be the [N, N] stock-stock block.
                # Defensive: if a model returns a 3D tensor, average over heads.
                if raw_scores_stk is not None and raw_scores_stk.dim() == 3:
                    raw_scores_stk = raw_scores_stk.mean(dim=0)

                if (raw_scores_stk is not None
                        and raw_scores_stk.dim() == 2
                        and raw_scores_stk.shape[0] == raw_scores_stk.shape[1]):
                    graph_loss = self._structural_loss(raw_scores_stk, label)
                    loss = main_loss + self.graph_loss_weight * graph_loss
                    loss = torch.clamp(loss, 0.0, 1e6)
                else:
                    loss = main_loss
            else:
                pred = model_output[0] if isinstance(model_output, tuple) else model_output
                loss = self.loss_fn(pred, label)

            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)

            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            mask, label = drop_na(label)
            label = zscore(label)

            model_output = self.model(feature.float())
            pred = model_output[0] if isinstance(model_output, tuple) else model_output
            loss = self.loss_fn(pred[mask], label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        if not (hasattr(data, '__getitem__') and hasattr(data, 'get_index')):
            raise ValueError("Data must be a PreprocessedTimeSeriesDataset object")
        sampler = DailyBatchSamplerRandom(data, shuffle)
        return DataLoader(data, sampler=sampler, drop_last=drop_last)

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                _, metrics = self.predict(dl_valid)
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f."
                      % (step, train_loss, metrics['IC'], metrics['ICIR'], metrics['RIC'], metrics['RICIR']))
            else:
                print("Epoch %d, train_loss %.6f" % (step, train_loss))

    def predict(self, dl_test):
        if self.fitted < 0:
            raise ValueError("model is not fitted yet!")
        print('Epoch:', self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds, ic_list, ric_list = [], [], []
        self.model.eval()

        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]

            with torch.no_grad():
                model_output = self.model(feature.float())
                pred = model_output[0] if isinstance(model_output, tuple) else model_output
                pred = pred.detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic_list.append(daily_ic)
            ric_list.append(daily_ric)

        test_index = dl_test.get_index()
        predictions = pd.Series(np.concatenate(preds), index=test_index)

        metrics = {
            'IC': np.mean(ic_list),
            'ICIR': np.mean(ic_list) / np.std(ic_list),
            'RIC': np.mean(ric_list),
            'RICIR': np.mean(ric_list) / np.std(ric_list),
        }
        return predictions, metrics
