"""Train HSCA using preprocessed time series data.

Model:
- HSCA: Heterogeneous Stock Conditioned Attention (paper Sec. 3)
  - Asymmetric Dual-Stream Encoder (Sec. 3.1)
  - State-Conditioned Heterogeneous Attention with State Gate, State Anchors,
    and Alpha-Guard (Sec. 3.2-3.4)
  - Structural Hinge Loss applied in base_model (Sec. 3.5)
  - Supports CSI300, CSI800, and S&P 500
"""
import argparse
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from PreprocessedTimeSeriesDataset import PreprocessedTimeSeriesDataset


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Train HSCA Model')
parser.add_argument('--universe', type=str, default=None, choices=['csi300', 'csi800', 'sp500'],
                    help='Market universe: csi300, csi800, or sp500')
parser.add_argument('--auto_train_both', action='store_true', default=False,
                    help='Automatically train both csi300 and csi800 (ignores --universe)')
parser.add_argument('--no_auto_train_both', dest='auto_train_both', action='store_false',
                    help='Disable auto training both markets, only train specified --universe')
parser.add_argument('--seeds', type=str, default='0,1,2,3,4,5',
                    help='Comma-separated list of seeds to train (default: 0,1,2,3,4,5)')
parser.add_argument('--start_seed', type=int, default=None,
                    help='Start training from specified seed (overrides --seeds)')
parser.add_argument('--end_seed', type=int, default=None,
                    help='End training at specified seed (requires --start_seed)')
parser.add_argument('--use_single_index', action='store_true', default=False,
                    help='CSI300 only: use M=1 (z300, 21 dims) instead of M=2 (z300+z500, 42 dims)')
parser.add_argument('--use_hinge_loss', action='store_true', default=True,
                    help='Use Structural Hinge Loss for L_struct (default: True)')
parser.add_argument('--no_hinge_loss', dest='use_hinge_loss', action='store_false',
                    help='Disable Hinge Loss; use MSE surrogate for L_struct (ablation)')
parser.add_argument('--loss', type=float, default=0.3, dest='graph_loss_weight',
                    help='Structural loss weight lambda (default: 0.3)')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU device ID (default: 0, uses CPU if CUDA unavailable)')
parser.add_argument('--n_epoch', type=int, default=50,
                    help='Number of training epochs (default: 50)')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='Learning rate (default: 1e-5)')
args = parser.parse_args()

# Validate GPU parameter
if torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        args.gpu = 0

# Import model class
from hsca_model import HSCAModel

# -----------------------------------------------------------------------------
# Market list
# -----------------------------------------------------------------------------

if args.auto_train_both:
    if args.universe is not None:
        print(f"[!] Warning: --universe {args.universe} specified but --auto_train_both is True, ignoring --universe")
    universes = ['csi300', 'csi800']
    print(f"[Auto Mode] Will train: {', '.join([u.upper() for u in universes])}")
else:
    if args.universe is None:
        args.universe = 'csi300'
        print(f"[Default] No --universe specified, defaulting to CSI300")
    universes = [args.universe]
    print(f"[Single Market Mode] Will train: {args.universe.upper()}")

# Data directory resolution
script_dir = Path(__file__).parent.absolute()
if (script_dir / 'data').exists():
    data_ts_dir = str(script_dir / 'data')
else:
    parent_data_dir = script_dir.parent / 'data'
    if parent_data_dir.exists():
        data_ts_dir = str(parent_data_dir)
    else:
        data_ts_dir = 'data'
        print(f"[Warning] Data directory not found, using relative path: {data_ts_dir}")

# Seed list
if args.start_seed is not None:
    if args.end_seed is not None:
        seeds = list(range(args.start_seed, args.end_seed + 1))
    else:
        seeds = list(range(args.start_seed, 6))
    print(f"Training from seed {args.start_seed}, seed list: {seeds}")
else:
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    print(f"Using seed list: {seeds}")

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

all_swa_ic = []
all_swa_icir = []
all_swa_ric = []
all_swa_ricir = []

for universe_idx, universe in enumerate(universes):
    print(f"\n{'='*80}")
    print(f"Training Market {universe.upper()} ({universe_idx + 1}/{len(universes)})")
    print(f"{'='*80}\n")

    print(f"Dataset: {universe.upper()}")
    print("Model: HSCA (Full - State Anchors + State Gate + Alpha-Guard + Structural Hinge Loss)")

    # Macro feature configuration display
    if universe == 'csi300':
        if args.use_single_index:
            print("  State Anchors: M=1 (z300 only, 21 dims)")
        else:
            print("  State Anchors: M=2 (z300+z500, 42 dims)")
    elif universe == 'csi800':
        print("  State Anchors: M=2 (z300+z500, 42 dims)")
    elif universe == 'sp500':
        print("  State Anchors: M=1 (single broad index, 21 dims)")

    if args.use_hinge_loss:
        print(f"  Structural Loss: Hinge (lambda={args.graph_loss_weight}, gamma=0.5)")
    else:
        print(f"  Structural Loss: MSE surrogate (lambda={args.graph_loss_weight})  [ablation]")

    # Load data
    try:
        dl_train = PreprocessedTimeSeriesDataset(f'{data_ts_dir}/{universe}_dl_train_ts.pkl')
        dl_valid = PreprocessedTimeSeriesDataset(f'{data_ts_dir}/{universe}_dl_valid_ts.pkl')
        dl_test = PreprocessedTimeSeriesDataset(f'{data_ts_dir}/{universe}_dl_test_ts.pkl')
    except Exception as e:
        print(f"[Error] Failed to load {universe.upper()} data: {e}")
        print(f"Skipping {universe.upper()} market, continuing to next market...")
        continue

    print(f"Training set: {len(dl_train)} samples, shape: {dl_train.data.shape}")
    print(f"Validation set: {len(dl_valid)} samples, shape: {dl_valid.data.shape}")
    print(f"Test set: {len(dl_test)} samples, shape: {dl_test.data.shape}")

    train_dates = dl_train.index.get_level_values('datetime')
    valid_dates = dl_valid.index.get_level_values('datetime')
    test_dates = dl_test.index.get_level_values('datetime')
    print(f"Training date range: {train_dates.min()} to {train_dates.max()}")
    print(f"Validation date range: {valid_dates.min()} to {valid_dates.max()}")
    print(f"Test date range: {test_dates.min()} to {test_dates.max()}")
    print("Data loading completed.\n")

    d_feat = 158
    d_model = 256

    n_epoch = args.n_epoch
    lr = args.lr
    GPU = args.gpu
    train_stop_loss_thred = 0.95

    # SWA result collectors
    swa_ic, swa_icir, swa_ric, swa_ricir = [], [], [], []

    for seed in seeds:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f'{log_dir}/hsca-full-{universe.upper()}-seed{seed}.txt'

        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        log_file = open(log_filename, 'w', encoding='utf-8')
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)

        try:
            model_kwargs = {
                'd_feat': d_feat,
                'd_model': d_model,
                'n_epochs': n_epoch,
                'lr': lr,
                'GPU': GPU,
                'seed': seed,
                'train_stop_loss_thred': train_stop_loss_thred,
                'save_path': 'model/',
                'save_prefix': universe,
                'use_hinge_loss': args.use_hinge_loss,
                'graph_loss_weight': args.graph_loss_weight,
                'universe': universe,
            }
            if universe == 'csi300':
                model_kwargs['use_single_index'] = args.use_single_index

            model = HSCAModel(**model_kwargs)

            print(f"\n{'='*60}")
            print(f"Start training: Seed {seed}")
            print(f"{'='*60}")

            start = time.time()
            train_loader = model._init_data_loader(dl_train, shuffle=True, drop_last=True)

            # SWA (Stochastic Weight Averaging) over the last 25% of training.
            swa_start_epoch = int(n_epoch * 0.75)
            swa_model_state = None
            swa_n_models = 0

            for step in range(n_epoch):
                epoch_start = time.time()

                train_start = time.time()
                train_loss = model.train_epoch(train_loader)
                if hasattr(model, 'scheduler'):
                    model.scheduler.step()
                    current_lr = model.scheduler.get_last_lr()[0]
                else:
                    current_lr = lr
                train_time = time.time() - train_start

                model.fitted = step

                eval_start = time.time()
                _, val_metrics = model.predict(dl_valid)
                _, test_metrics = model.predict(dl_test)
                eval_time = time.time() - eval_start

                epoch_time = time.time() - epoch_start
                print(f"Epoch {step:3d} | lr={current_lr:.2e} | train_loss={train_loss:.6f} | "
                      f"val IC={val_metrics['IC']:.4f}, RIC={val_metrics['RIC']:.4f} | "
                      f"test IC={test_metrics['IC']:.4f}, RIC={test_metrics['RIC']:.4f} | "
                      f"t_train={train_time:.1f}s, t_eval={eval_time:.1f}s, t_total={epoch_time:.1f}s")

                # SWA snapshot
                if step >= swa_start_epoch:
                    current_state = {k: v.clone().detach() for k, v in model.model.state_dict().items()}
                    if swa_model_state is None:
                        swa_model_state = current_state
                        swa_n_models = 1
                    else:
                        for key in swa_model_state:
                            swa_model_state[key] = (swa_model_state[key] * swa_n_models +
                                                    current_state[key]) / (swa_n_models + 1)
                        swa_n_models += 1

            running_time = time.time() - start

            print(f"\n{'='*60}")
            print(f"Seed {seed} Training Completed:")
            print(f"  Training time: {running_time:.2f} seconds ({running_time/60:.2f} minutes)")

            swa_test_metrics = None
            if swa_model_state is not None and swa_n_models > 0:
                model.model.load_state_dict(swa_model_state)
                os.makedirs('model', exist_ok=True)
                model_filename = f'model/{universe}_hsca_{seed}.pkl'
                torch.save(swa_model_state, model_filename)
                print(f"  Model saved: {model_filename}")

                print(f"\n[SWA Model Performance] (averaged weights from last {swa_n_models} epochs):")
                _, swa_test_metrics = model.predict(dl_test)
                print(f"  IC: {swa_test_metrics['IC']:.4f}, ICIR: {swa_test_metrics['ICIR']:.4f}, "
                      f"RankIC: {swa_test_metrics['RIC']:.4f}, RankICIR: {swa_test_metrics['RICIR']:.4f}")
            else:
                print(f"\n[SWA Model]: Not generated (insufficient training epochs)")

            print(f"{'='*60}\n")

            if swa_test_metrics:
                swa_ic.append(swa_test_metrics['IC'])
                swa_icir.append(swa_test_metrics['ICIR'])
                swa_ric.append(swa_test_metrics['RIC'])
                swa_ricir.append(swa_test_metrics['RICIR'])

        finally:
            sys.stdout = original_stdout
            log_file.close()
            print(f"Log saved to: {log_filename}")

    # Market-level summary
    print(f"\n{'='*80}")
    print(f"[{universe.upper()} Market Results Summary]")
    print(f"{'='*80}")

    if len(swa_ic) > 0:
        print("\n[SWA Model] (Stochastic Weight Averaging):")
        print("  IC:       {:.4f} ± {:.4f}".format(np.mean(swa_ic), np.std(swa_ic)))
        print("  ICIR:     {:.4f} ± {:.4f}".format(np.mean(swa_icir), np.std(swa_icir)))
        print("  RankIC:   {:.4f} ± {:.4f}".format(np.mean(swa_ric), np.std(swa_ric)))
        print("  RankICIR: {:.4f} ± {:.4f}".format(np.mean(swa_ricir), np.std(swa_ricir)))

        all_swa_ic.extend(swa_ic)
        all_swa_icir.extend(swa_icir)
        all_swa_ric.extend(swa_ric)
        all_swa_ricir.extend(swa_ricir)
    else:
        print("\n[SWA Model]: Not generated (insufficient training epochs)")

    print(f"{'='*80}\n")

# Cross-market summary
if len(universes) > 1:
    print("\n" + "=" * 80)
    print("[Summary Results Across All Markets]")
    print("=" * 80)
    if len(all_swa_ic) > 0:
        print("\n[SWA Model] (Stochastic Weight Averaging, across all markets):")
        print("  IC:       {:.4f} ± {:.4f}".format(np.mean(all_swa_ic), np.std(all_swa_ic)))
        print("  ICIR:     {:.4f} ± {:.4f}".format(np.mean(all_swa_icir), np.std(all_swa_icir)))
        print("  RankIC:   {:.4f} ± {:.4f}".format(np.mean(all_swa_ric), np.std(all_swa_ric)))
        print("  RankICIR: {:.4f} ± {:.4f}".format(np.mean(all_swa_ricir), np.std(all_swa_ricir)))
    print("=" * 80)
