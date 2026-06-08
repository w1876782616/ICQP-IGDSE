import os
import glob
from numpy.random import permutation
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import FLAGS, TARGETS
import sys
from saver import saver
from utils import MLP, OurTimer, get_save_path, _get_y_with_target, get_root_path
import programl_data
from torch_geometric.utils import degree
from model import Net
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, mean_absolute_percentage_error, classification_report, confusion_matrix, root_mean_squared_error
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from collections import defaultdict

def custom_collate(batch):
    from torch_geometric.data import Data
    points = []
    clean_batch = []
    for data in batch:
        point_val = getattr(data, 'point', None)
        points.append(point_val)
        new_data = Data()
        for key, value in data.__dict__.items():
            if isinstance(value, dict):
                continue
            if isinstance(value, (list, tuple)):
                if any((isinstance(item, dict) for item in value)):
                    continue
            if isinstance(value, (torch.Tensor, list, tuple, int, float, str, bool, type(None))):
                try:
                    setattr(new_data, key, value)
                except:
                    pass
            elif isinstance(value, np.ndarray):
                try:
                    setattr(new_data, key, torch.from_numpy(value))
                except:
                    pass
        clean_batch.append(new_data)
    batch_obj = Batch.from_data_list(clean_batch)
    batch_obj.point = points
    return batch_obj
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, mean_absolute_percentage_error
from scipy.stats import rankdata, kendalltau, pearsonr
from torch.nn import Sequential, Linear, ReLU
from tqdm import tqdm
from os.path import join
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
from src.causal_data_utils import hamming_distance, create_intervention_pairs
from src.parameter import DesignPoint
from typing import List, Tuple, Optional, Dict
out_dim = FLAGS.out_dim
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen', 'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d', 'correlation', 'covariance', 'syrk']

def compute_causal_loss(model, data, code, design_points: Optional[List[DesignPoint]], target_list: List[str], flags, out_dict=None, intervention_delta_accumulator: Optional[Dict[str, List[Tuple[float, float]]]]=None, max_pairs_override: Optional[int]=None) -> Optional[torch.Tensor]:
    if design_points is None or len(design_points) == 0:
        return None
    if not flags.use_causal:
        return None
    valid_indices = [i for i, dp in enumerate(design_points) if dp is not None]
    if len(valid_indices) < 2:
        return None
    from torch_geometric.data import Batch
    if not isinstance(data, Batch):
        return None
    batch_size = data.num_graphs if hasattr(data, 'num_graphs') else len(design_points)
    if batch_size < 2:
        return None
    pred_source = None
    if hasattr(model, '_last_causal_pred') and getattr(model, '_last_causal_pred', None) is not None:
        pred_source = model._last_causal_pred
    elif out_dict is not None:
        pred_source = out_dict
    else:
        return None
    true_values = {}
    for target in target_list:
        y = _get_y_with_target(data, target)
        if y is not None:
            if flags.task == 'regression':
                true_values[target] = y.view((len(y), flags.out_dim))
            else:
                true_values[target] = y.view(len(y))
        else:
            return None
    valid_design_points = [(i, design_points[i]) for i in valid_indices]
    pairs = []
    debug_logged = getattr(compute_causal_loss, '_debug_logged', False)
    if not debug_logged:
        from saver import saver
        saver.log_info(f'[Causal Loss Debug] Starting pair creation: {len(valid_design_points)} valid design_points in batch')
        compute_causal_loss._debug_logged = True
    for i in range(len(valid_design_points)):
        for j in range(i + 1, len(valid_design_points)):
            idx1, dp1 = valid_design_points[i]
            idx2, dp2 = valid_design_points[j]
            dist = hamming_distance(dp1, dp2)
            if 1 <= dist <= 2:
                pairs.append((idx1, idx2))
                if not debug_logged and len(pairs) <= 3:
                    saver.log_info(f'[Causal Loss Debug] Pair {len(pairs)}: idx1={idx1}, idx2={idx2}, dist={dist}')
    if len(pairs) == 0:
        if not debug_logged:
            saver.log_info(f'[Causal Loss Debug] No valid pairs found (need Hamming distance 1-2)')
        return None
    train_cap = getattr(flags, 'causal_max_pairs_per_batch', 10)
    if max_pairs_override is not None and max_pairs_override > 0:
        cap = max_pairs_override
        eval_trim = True
    else:
        cap = train_cap
        eval_trim = False
    n_pairs_found = len(pairs)
    if n_pairs_found > cap:
        if eval_trim:
            pairs = sorted(pairs)[:cap]
        else:
            import random
            pairs = random.sample(pairs, cap)
        if not debug_logged:
            saver.log_info(f'[Causal Loss Debug] Capped intervention pairs: use {cap} of {n_pairs_found} (deterministic={eval_trim})')
    if not debug_logged:
        saver.log_info(f'[Causal Loss Debug] Computing causal loss for {len(pairs)} pairs')
    device = next(model.parameters()).device
    total_causal_loss = 0.0
    valid_pairs = 0
    try:
        for idx1, idx2 in pairs:
            pred_diff = {}
            true_diff = {}
            for target in target_list:
                if target not in pred_source or target not in true_values:
                    continue
                pred1 = pred_source[target][idx1:idx1 + 1]
                pred2 = pred_source[target][idx2:idx2 + 1]
                true1 = true_values[target][idx1:idx1 + 1]
                true2 = true_values[target][idx2:idx2 + 1]
                pred_diff[target] = pred2 - pred1
                true_diff[target] = true2 - true1
            if intervention_delta_accumulator is not None:
                for target in target_list:
                    if target in pred_diff and target in true_diff:
                        with torch.no_grad():
                            td = true_diff[target].detach().cpu().float().reshape(-1).mean().item()
                            pd = pred_diff[target].detach().cpu().float().reshape(-1).mean().item()
                        intervention_delta_accumulator[target].append((td, pd))
            if len(pred_diff) == 0:
                continue
            target_losses = []
            for target in target_list:
                if target in pred_diff and target in true_diff:
                    diff = pred_diff[target] - true_diff[target]
                    if flags.task == 'regression' and flags.out_dim > 1:
                        loss_target = torch.mean(diff ** 2)
                    else:
                        loss_target = torch.mean(diff ** 2)
                    target_losses.append(loss_target)
            if len(target_losses) > 0:
                pair_loss = torch.stack(target_losses).mean()
                total_causal_loss += pair_loss
                valid_pairs += 1
        if valid_pairs == 0:
            if not debug_logged:
                saver.log_info(f'[Causal Loss Debug] No valid pairs after loss computation')
            return None
        avg_loss = total_causal_loss / valid_pairs
        if not debug_logged:
            saver.log_info(f'[Causal Loss Debug] Computed causal loss: {avg_loss.item():.6f} (from {valid_pairs} pairs)')
        return avg_loss
    except Exception as e:
        if not debug_logged:
            from saver import saver
            saver.warning(f'[Causal Loss Debug] Exception during causal loss computation: {e}')
            import traceback
            saver.warning(traceback.format_exc())
        return None
        
def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = [data for data, _ in d['pred']]
    pred = [data for _, data in d['pred']]
    target_names = ['invalid', 'valid']
    saver.info('classification report')
    saver.log_info(classification_report(labels, pred, target_names=target_names))
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    saver.info(f'Confusion matrix:\n{cm}')

def _report_rmse_etc(points_dict, label, print_result=True):
    if print_result:
        saver.log_info(label)
    data = defaultdict(list)
    tot_mape, tot_rmse, tot_mse, tot_mae, tot_max_err, tot_tau, tot_std = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    num_data = None
    try:
        for target_name, d in points_dict.items():
            true_li = [data for data, _ in d['pred']]
            pred_li = [data for _, data in d['pred']]
            num_data = len(true_li)
            mape = mean_absolute_percentage_error(true_li, pred_li)
            rmse = root_mean_squared_error(true_li, pred_li)
            mse = mean_squared_error(true_li, pred_li)
            mae = mean_absolute_error(true_li, pred_li)
            max_err = max_error(true_li, pred_li)
            true_rank = rankdata(true_li)
            pred_rank = rankdata(pred_li)
            tau = kendalltau(true_rank, pred_rank)[0]
            data['target'].append(target_name)
            data['mape'].append(mape)
            data['rmse'].append(rmse)
            data['mse'].append(mse)
            data['mae'].append(mae)
            data['max_err'].append(max_err)
            data['tau'].append(tau)
            tot_mape += mape
            tot_rmse += rmse
            tot_mse += mse
            tot_mae += mae
            tot_max_err += max_err
            tot_tau += tau
            pred_std = d.get('pred_std')
            if pred_std is not None:
                assert type(pred_std) is np.ndarray, f'{type(pred_std)}'
                pred_std = np.mean(pred_std)
                data['pred_std'].append(pred_std)
                tot_std += pred_std
        data['target'].append('tot/avg')
        data['mape'].append(tot_mape)
        data['rmse'].append(tot_rmse)
        data['mse'].append(tot_mse)
        data['mae'].append(tot_mae)
        data['max_err'].append(tot_max_err)
        data['tau'].append(tot_tau / len(points_dict))
        if 'pred_std' in data:
            data['pred_std'].append(tot_std / len(points_dict))
    except ValueError as v:
        saver.log_info(f'Error {v}')
        data = defaultdict(list)
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    if print_result:
        saver.log_info(num_data)
        saver.log_info(df.round(4))
    return df

def inference(dataset):
    gp, cp = dataset.processed_file_names
    gpr = [f'{ke} graph nums: {len(p)}' for ke, p in gp.items()]
    cpr = [f'{ke} code nums: {len(p)}' for ke, p in cp.items()]
    gpr_sum = sum([len(p) for p in gp.values()])
    cpr_sum = sum([len(p) for p in cp.values()])
    for i in gpr:
        print(i)
    for j in cpr:
        print(j)
    print(f'graph total number: {gpr_sum}, code total number: {cpr_sum}')
    print(f'machsuite len: {len(MACHSUITE_KERNEL)}, poly len: {len(poly_KERNEL)}')
    minx, pinx = (5, 10)
    li_g = []
    li_c = []
    li_points = []
    points_cache = {}
    global_points_list = getattr(dataset, '_points_list', None)
    global_idx = 0

    def _extract_point_from_path(file_path, local_i):
        if not isinstance(file_path, str):
            return None
        norm_path = file_path.replace('\\', '/')
        parts = norm_path.split('/')
        benchmark = None
        kernel_name = None
        for idx, part in enumerate(parts):
            if part in ['machsuite', 'poly']:
                benchmark = part
                if idx + 1 < len(parts):
                    kernel_name = parts[idx + 1]
                break
        if benchmark is None or kernel_name is None:
            return None
        points_file = os.path.join(get_root_path(), 'two_tower_dataset', 'points', benchmark, kernel_name, 'points_list.pkl')
        cache_key = f'{benchmark}/{kernel_name}'
        if cache_key not in points_cache:
            if not os.path.exists(points_file):
                points_cache[cache_key] = None
            else:
                try:
                    import pickle
                    with open(points_file, 'rb') as f:
                        points_cache[cache_key] = pickle.load(f)
                except Exception:
                    points_cache[cache_key] = None
        plist = points_cache.get(cache_key, None)
        if plist is None:
            return None
        if 0 <= local_i < len(plist):
            return plist[local_i]
        return None
    for m in tqdm(MACHSUITE_KERNEL[minx:], position=0, total=len(MACHSUITE_KERNEL[:minx]), file=sys.stdout):
        gpt = gp[m]
        for i in range(1, len(gpt)):
            g, c = dataset.get_data(i, m)
            li_g.append(g)
            li_c.append(c)
            point = getattr(g, 'point', None)
            if point is None and i < len(gpt):
                point = _extract_point_from_path(gpt[i], i)
            if point is None and global_points_list is not None and (global_idx < len(global_points_list)):
                point = global_points_list[global_idx]
            li_points.append(point)
            global_idx += 1
    for p in tqdm(poly_KERNEL[pinx:], position=0, total=len(poly_KERNEL[:pinx]), file=sys.stdout):
        gpt = gp[p]
        for i in range(1, len(gpt)):
            g, c = dataset.get_data(i, p)
            li_g.append(g)
            li_c.append(c)
            point = getattr(g, 'point', None)
            if point is None and i < len(gpt):
                point = _extract_point_from_path(gpt[i], i)
            if point is None and global_points_list is not None and (global_idx < len(global_points_list)):
                point = global_points_list[global_idx]
            li_points.append(point)
            global_idx += 1
    li_len = len(li_g)
    if FLAGS.inference_use_all_data:
        li_r = [[], [], list(range(li_len))]
    else:
        l_t, l_v = (int(li_len * 0.7), int(li_len * 0.15))
        from numpy import random
        rinx = permutation(range(li_len))
        li_r = [rinx[0:l_t], rinx[l_t:l_t + l_v], rinx[l_t + l_v:]]
    li = []
    li_code = []
    li_points_split = []
    for i in li_r:
        tmp = []
        tmp_1 = []
        tmp_points = []
        for j in i:
            tmp.append(li_g[j])
            tmp_1.append(li_c[j])
            tmp_points.append(li_points[j] if j < len(li_points) else None)
        li.append(tmp)
        li_code.append(tmp_1)
        li_points_split.append(tmp_points)
    saver.log_info(f'{len(li_g)} graphs in total: {len(li[0])} train {len(li[1])} val {len(li[2])} test')
    if FLAGS.inference_use_all_data:
        saver.log_info('[Inference] inference_use_all_data=True, using 100% unseen samples for evaluation')
    saver.log_info(f'{len(li_c)} codes in total: {len(li_code[0])} train {len(li_code[1])} val {len(li_code[2])} test')
    test_loader = DataLoader(li[2], batch_size=FLAGS.batch_size, pin_memory=True, collate_fn=custom_collate)
    test_codes = li_code[2]
    test_points = li_points_split[2] if len(li_points_split) > 2 else None
    if test_points is not None:
        non_none_pts = sum((1 for p in test_points if p is not None))
    all_non_none_pts = sum((1 for p in li_points if p is not None))
    edge_dim = test_loader.dataset[0].edge_attr.shape[1]
    max_degree = -1
    for data in li[2]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in li[2]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    if FLAGS.comparative_if:
        if FLAGS.comparative_model == 'pna':
            model = Net(deg, edge_dim).to(FLAGS.device)
        else:
            model = Net().to(FLAGS.device)
    else:
        model = Net().to(FLAGS.device)
    if FLAGS.task == 'regression':
        if FLAGS.model_path != None:
            old_state_dict = torch.load(FLAGS.model_path, map_location=torch.device(FLAGS.device))
            missing_keys, unexpected_keys = model.load_state_dict(old_state_dict, strict=False)
            saver.info(f'Loaded model from {FLAGS.model_path} (partial load, strict=False)')
            if missing_keys:
                saver.info(f'Missing keys (initialized randomly): {len(missing_keys)}')
            if unexpected_keys:
                if not FLAGS.use_causal:
                    causal_keys = [k for k in unexpected_keys if 'pragma_encoder' in k or 'causal_head' in k]
                    non_causal_keys = [k for k in unexpected_keys if k not in causal_keys]
                    if causal_keys:
                        saver.info(f'Ignored causal checkpoint keys (use_causal=False): {len(causal_keys)}')
                    if non_causal_keys:
                        saver.warning(f'Unexpected non-causal checkpoint keys: {len(non_causal_keys)}')
                else:
                    saver.warning(f'Unexpected checkpoint keys: {len(unexpected_keys)}')
        else:
            saver.error(f'model path should be set during inference')
            raise RuntimeError()
        print(model)
    else:
        if FLAGS.class_model_path != None:
            old_state_dict = torch.load(FLAGS.class_model_path, map_location=torch.device(FLAGS.device))
            missing_keys, unexpected_keys = model.load_state_dict(old_state_dict, strict=False)
            saver.info(f'Loaded class model from {FLAGS.class_model_path} (partial load, strict=False)')
            if missing_keys:
                saver.info(f'Missing keys (initialized randomly): {len(missing_keys)}')
            if unexpected_keys:
                saver.warning(f'Unexpected checkpoint keys: {len(unexpected_keys)}')
        else:
            saver.error(f'model path should be set during inference')
            raise RuntimeError()
        print(model)
    saver.log_model_architecture(model)
    testr, loss_dict, encode_loss = test(test_loader, test_codes, 'test', model, 0, plot_test=True, design_points=test_points)
    saver.log_info(f'{loss_dict}')
    saver.log_info('Test loss: {:.7f}, encode loss: {:.7f}'.format(testr, encode_loss))

def train_main(dataset, pragma_dim=None):
    saver.info(f'Reading dataset from')
    try:
        if hasattr(dataset, 'processed_file_names_dict'):
            processed_dict = dataset.processed_file_names_dict
        elif hasattr(dataset, 'processed_file_names'):
            processed_dict = dataset.processed_file_names
            if isinstance(processed_dict, (tuple, list)) and len(processed_dict) == 2:
                gp, cp = processed_dict
                saver.log_info(f"[Debug] processed_file_names returns tuple, gp keys (first 5): {(list(gp.keys())[:5] if isinstance(gp, dict) else 'not dict')}")
                processed_dict = None
            else:
                processed_dict = processed_dict
        else:
            raise AttributeError('Dataset has neither processed_file_names_dict nor processed_file_names')
        if processed_dict is None:
            saver.log_info(f'[Debug] Already unpacked from processed_file_names, gp keys: {list(gp.keys())[:5]}, total kernels: {len(gp)}')
        elif isinstance(processed_dict, dict):
            saver.log_info(f'[Debug] processed_file_names_dict keys (first 5): {list(processed_dict.keys())[:5]}')
            first_key = list(processed_dict.keys())[0] if processed_dict else None
            first_value = processed_dict[first_key] if first_key else None
            if first_value and isinstance(first_value, (list, tuple)) and (len(first_value) > 0):
                first_path = first_value[0] if isinstance(first_value[0], str) else None
                saver.log_info(f"[Debug] First path sample: {(first_path[:100] if first_path else 'None')}")
                if first_path and ('two_tower_dataset' in first_path or '.pt' in first_path):
                    saver.log_info(f'[Debug] Detected file paths list, extracting kernel names from paths')
                    gp = defaultdict(list)
                    cp = defaultdict(list)
                    for key, file_paths in processed_dict.items():
                        file_list = file_paths if isinstance(file_paths, (list, tuple)) else [file_paths]
                        saver.log_info(f'[Debug] Processing key: {key}, file_list length: {len(file_list)}')
                        for file_path in file_list:
                            if isinstance(file_path, str):
                                path_parts = file_path.split('/')
                                kernel = None
                                for i, part in enumerate(path_parts):
                                    if part in ['poly', 'machsuite'] and i + 1 < len(path_parts):
                                        kernel = path_parts[i + 1]
                                        saver.log_info(f'[Debug] Found kernel: {kernel} from path: {file_path}')
                                        break
                                if kernel:
                                    if kernel.endswith('_processed_result'):
                                        kernel = kernel[:-len('_processed_result')]
                                    gp[kernel].append(file_path)
                                    cp[kernel].append(file_path)
                                else:
                                    saver.warning(f'[Debug] Failed to extract kernel from path: {file_path}')
                    saver.log_info(f'[Debug] Extracted kernels: {list(gp.keys())}')
                    gp, cp = (dict(gp), dict(cp))
                else:
                    gp = {}
                    cp = {}
                    for kernel, file_paths in processed_dict.items():
                        kernel_normalized = kernel
                        if kernel.endswith('_processed_result'):
                            kernel_normalized = kernel[:-len('_processed_result')]
                        if isinstance(file_paths, (list, tuple)):
                            if len(file_paths) > 0:
                                gp[kernel_normalized] = list(file_paths)
                                cp[kernel_normalized] = list(file_paths)
                        elif isinstance(file_paths, str):
                            gp[kernel_normalized] = [file_paths]
                            cp[kernel_normalized] = [file_paths]
                        else:
                            saver.warning(f'[Debug] Unexpected file_paths type for kernel {kernel}: {type(file_paths)}')
            else:
                gp = {}
                cp = {}
                for kernel, file_paths in processed_dict.items():
                    kernel_normalized = kernel
                    if kernel.endswith('_processed_result'):
                        kernel_normalized = kernel[:-len('_processed_result')]
                    if isinstance(file_paths, (list, tuple)):
                        if len(file_paths) > 0:
                            gp[kernel_normalized] = list(file_paths)
                            cp[kernel_normalized] = list(file_paths)
                    elif isinstance(file_paths, str):
                        gp[kernel_normalized] = [file_paths]
                        cp[kernel_normalized] = [file_paths]
                    else:
                        saver.warning(f'[Debug] Unexpected file_paths type for kernel {kernel}: {type(file_paths)}')
            saver.log_info(f'[Debug] After processing: gp keys (first 5): {list(gp.keys())[:5]}, total kernels: {len(gp)}')
        elif isinstance(processed_dict, (tuple, list)) and len(processed_dict) == 2:
            gp, cp = processed_dict
            saver.log_info(f"[Debug] processed_file_names_dict is tuple, gp keys (first 5): {(list(gp.keys())[:5] if isinstance(gp, dict) else 'not dict')}")
        elif processed_dict is None:
            pass
        else:
            raise ValueError(f'Unexpected processed_file_names_dict type: {type(processed_dict)}')
    except (AttributeError, ValueError, TypeError) as e:
        saver.warning(f'processed_file_names_dict not available or invalid ({e}), trying to build from processed_file_names')
        gp = defaultdict(list)
        cp = defaultdict(list)
        try:
            all_files = dataset.processed_file_names
        except:
            all_files = []
        for file_path in all_files:
            try:
                if isinstance(file_path, dict):
                    saver.warning(f'Skipping dict entry in processed_file_names: {file_path}')
                    continue
                path_parts = file_path.split('/')
                kernel = None
                for i, part in enumerate(path_parts):
                    if part in ['poly', 'machsuite'] and i + 1 < len(path_parts):
                        kernel = path_parts[i + 1]
                        break
                if kernel:
                    if kernel.endswith('_processed_result'):
                        kernel = kernel[:-len('_processed_result')]
                    gp[kernel].append(file_path)
                    cp[kernel].append(file_path)
                else:
                    data = torch.load(file_path)
                    kernel = getattr(data, 'kernel', None)
                    if kernel is not None:
                        if kernel.endswith('_processed_result'):
                            kernel = kernel[:-len('_processed_result')]
                        gp[kernel].append(file_path)
                        cp[kernel].append(file_path)
            except Exception as e:
                saver.warning(f'Failed to process {file_path}: {e}')
        gp, cp = (dict(gp), dict(cp))
    if len(gp) == 0:
        saver.warning('[Debug] gp is empty after processing, trying alternative extraction method')
        try:
            processed_dict = dataset.processed_file_names_dict
            if isinstance(processed_dict, dict):
                gp = defaultdict(list)
                cp = defaultdict(list)
                all_paths = []
                for key, value in processed_dict.items():
                    if isinstance(value, (list, tuple)):
                        all_paths.extend(value)
                    elif isinstance(value, str):
                        all_paths.append(value)
                saver.log_info(f'[Debug] Collected {len(all_paths)} file paths from processed_file_names_dict')
                for file_path in all_paths:
                    if isinstance(file_path, str):
                        path_parts = file_path.split('/')
                        kernel = None
                        for i, part in enumerate(path_parts):
                            if part in ['poly', 'machsuite'] and i + 1 < len(path_parts):
                                kernel = path_parts[i + 1]
                                break
                        if kernel:
                            if kernel.endswith('_processed_result'):
                                kernel = kernel[:-len('_processed_result')]
                            gp[kernel].append(file_path)
                            cp[kernel].append(file_path)
                gp, cp = (dict(gp), dict(cp))
                saver.log_info(f'[Debug] After alternative extraction: gp keys: {list(gp.keys())}, total kernels: {len(gp)}')
        except Exception as e:
            saver.warning(f'[Debug] Alternative extraction failed: {e}')
    gpr = [f'{ke} graph nums: {len(p)}' for ke, p in gp.items()]
    cpr = [f'{ke} code nums: {len(p)}' for ke, p in cp.items()]
    gpr_sum = sum([len(p) for p in gp.values()])
    cpr_sum = sum([len(p) for p in cp.values()])
    for i in gpr:
        print(i)
    for j in cpr:
        print(j)
    print(f'graph total number: {gpr_sum}, code total number: {cpr_sum}')
    print(f'machsuite len: {len(MACHSUITE_KERNEL)}, poly len: {len(poly_KERNEL)}')
    minx, pinx = (5, 10)
    li_g = []
    li_c = []
    li_points = []
    has_get_point = hasattr(dataset, 'get_point') or (hasattr(dataset, '_points_list') and dataset._points_list is not None)
    global_points_list = None
    if not has_get_point:
        try:
            from utils import get_save_path
            from config import FLAGS
            TARGET_LOCAL = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']
            SAVE_DIR = join(get_save_path(), FLAGS.dataset, f"new-train-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{''.join(TARGET_LOCAL)}")
            global_points_file = join(SAVE_DIR, 'points_list.pkl')
            if os.path.exists(global_points_file):
                import pickle
                with open(global_points_file, 'rb') as f:
                    global_points_list = pickle.load(f)
    global_idx = 0
    kernel_idx_map = {}
    for m in tqdm(MACHSUITE_KERNEL[:minx], position=0, total=len(MACHSUITE_KERNEL[:minx]), file=sys.stdout):
        if m in gp:
            gpt = gp[m]
        elif f'{m}_processed_result' in gp:
            gpt = gp[f'{m}_processed_result']
        else:
            saver.warning(f'Kernel {m} not found in gp, available keys: {list(gp.keys())[:10]}')
            continue
        for i in range(1, len(gpt)):
            g, c = dataset.get_data(i, m)
            li_g.append(g)
            li_c.append(c)
            kernel_idx_map[m, i] = global_idx
            point = None
            if has_get_point:
                try:
                    point = dataset.get_point(global_idx)
                except (AttributeError, IndexError, TypeError) as e:
                    pass
            if point is None:
                try:
                    if i < len(gpt):
                        file_path = gpt[i]
                        if global_idx < 5:
                            saver.log_info(f'[Causal Debug] Processing file_path: {file_path}')
                        path_parts = file_path.split(os.sep)
                        benchmark = None
                        kernel_name = None
                        for j, part in enumerate(path_parts):
                            if part in ['machsuite', 'poly']:
                                benchmark = part
                                if j + 1 < len(path_parts):
                                    kernel_name = path_parts[j + 1]
                                break
                        if benchmark and kernel_name:
                            points_base_dir = os.path.join(get_root_path(), 'two_tower_dataset', 'points')
                            points_file = os.path.join(points_base_dir, benchmark, kernel_name, 'points_list.pkl')
                            if global_idx < 5:
                                saver.log_info(f'[Causal Debug] Checking new location: {points_file}, exists={os.path.exists(points_file)}')
                            if os.path.exists(points_file):
                                import pickle
                                with open(points_file, 'rb') as f:
                                    kernel_points_list = pickle.load(f)
                                file_name = os.path.basename(file_path)
                                file_idx = None
                                if file_name.startswith('data_') and file_name.endswith('.pt'):
                                    file_idx = int(file_name.replace('data_', '').replace('.pt', ''))
                                elif file_name.endswith('.pt'):
                                    file_idx = int(file_name.replace('.pt', ''))
                                if file_idx is not None and 0 <= file_idx < len(kernel_points_list):
                                    point = kernel_points_list[file_idx]
                            else:
                                file_dir = os.path.dirname(file_path)
                                old_points_file = os.path.join(file_dir, 'points_list.pkl')
                                if os.path.exists(old_points_file):
                                    import pickle
                                    with open(old_points_file, 'rb') as f:
                                        kernel_points_list = pickle.load(f)
                                    file_name = os.path.basename(file_path)
                                    file_idx = None
                                    if file_name.startswith('data_') and file_name.endswith('.pt'):
                                        file_idx = int(file_name.replace('data_', '').replace('.pt', ''))
                                    elif file_name.endswith('.pt'):
                                        file_idx = int(file_name.replace('.pt', ''))
                                    if file_idx is not None and 0 <= file_idx < len(kernel_points_list):
                                        point = kernel_points_list[file_idx]
                                    if global_points_list is not None and global_idx < len(global_points_list):
                                        point = global_points_list[global_idx]
                        else:
                            if point is None:
                                if global_points_list is not None and global_idx < len(global_points_list):
                                    point = global_points_list[global_idx]
                except Exception as e:
            if point is None and global_points_list is not None and (global_idx < len(global_points_list)):
                point = global_points_list[global_idx]
            li_points.append(point)
            global_idx += 1
    for p in tqdm(poly_KERNEL[:pinx], position=0, total=len(poly_KERNEL[:pinx]), file=sys.stdout):
        if p in gp:
            gpt = gp[p]
        elif f'{p}_processed_result' in gp:
            gpt = gp[f'{p}_processed_result']
        else:
            saver.warning(f'Kernel {p} not found in gp, available keys: {list(gp.keys())[:10]}')
            continue
        for i in range(1, len(gpt)):
            g, c = dataset.get_data(i, p)
            li_g.append(g)
            li_c.append(c)
            kernel_idx_map[p, i] = global_idx
            point = None
            if has_get_point:
                try:
                    point = dataset.get_point(global_idx)
                except (AttributeError, IndexError, TypeError) as e:
                    pass
            if point is None:
                try:
                    if i < len(gpt):
                        file_path = gpt[i]
                        if global_idx < 5:
                            saver.log_info(f'[Causal Debug] Processing file_path: {file_path}')
                        path_parts = file_path.split(os.sep)
                        benchmark = None
                        kernel_name = None
                        for j, part in enumerate(path_parts):
                            if part in ['machsuite', 'poly']:
                                benchmark = part
                                if j + 1 < len(path_parts):
                                    kernel_name = path_parts[j + 1]
                                break
                        if benchmark and kernel_name:
                            points_base_dir = os.path.join(get_root_path(), 'two_tower_dataset', 'points')
                            points_file = os.path.join(points_base_dir, benchmark, kernel_name, 'points_list.pkl')
                            if global_idx < 5:
                                saver.log_info(f'[Causal Debug] Checking new location: {points_file}, exists={os.path.exists(points_file)}')
                            if os.path.exists(points_file):
                                import pickle
                                with open(points_file, 'rb') as f:
                                    kernel_points_list = pickle.load(f)
                                file_name = os.path.basename(file_path)
                                file_idx = None
                                if file_name.startswith('data_') and file_name.endswith('.pt'):
                                    file_idx = int(file_name.replace('data_', '').replace('.pt', ''))
                                elif file_name.endswith('.pt'):
                                    file_idx = int(file_name.replace('.pt', ''))
                                if file_idx is not None and 0 <= file_idx < len(kernel_points_list):
                                    point = kernel_points_list[file_idx]
                            else:
                                file_dir = os.path.dirname(file_path)
                                old_points_file = os.path.join(file_dir, 'points_list.pkl')
                                if os.path.exists(old_points_file):
                                    import pickle
                                    with open(old_points_file, 'rb') as f:
                                        kernel_points_list = pickle.load(f)
                                    file_name = os.path.basename(file_path)
                                    file_idx = None
                                    if file_name.startswith('data_') and file_name.endswith('.pt'):
                                        file_idx = int(file_name.replace('data_', '').replace('.pt', ''))
                                    elif file_name.endswith('.pt'):
                                        file_idx = int(file_name.replace('.pt', ''))
                                    if file_idx is not None and 0 <= file_idx < len(kernel_points_list):
                                        point = kernel_points_list[file_idx]
                                if point is None:
                                    if global_points_list is not None and global_idx < len(global_points_list):
                                        point = global_points_list[global_idx]
                        else:
                            if point is None:
                                if global_points_list is not None and global_idx < len(global_points_list):
                                    point = global_points_list[global_idx]
            if point is None and global_points_list is not None and (global_idx < len(global_points_list)):
                point = global_points_list[global_idx]
            li_points.append(point)
            global_idx += 1
    points_loaded = sum((1 for p in li_points if p is not None))
    kernel_points_count = defaultdict(int)
    kernel_points_loaded = defaultdict(int)
    current_kernel = None
    point_idx = 0
    for m in MACHSUITE_KERNEL[:minx]:
        if m in gp:
            gpt = gp[m]
        elif f'{m}_processed_result' in gp:
            gpt = gp[f'{m}_processed_result']
        else:
            continue
        for i in range(1, len(gpt)):
            if point_idx < len(li_points):
                kernel_points_count[m] += 1
                if li_points[point_idx] is not None:
                    kernel_points_loaded[m] += 1
                point_idx += 1
    for p in poly_KERNEL[:pinx]:
        if p in gp:
            gpt = gp[p]
        elif f'{p}_processed_result' in gp:
            gpt = gp[f'{p}_processed_result']
        else:
            continue
        for i in range(1, len(gpt)):
            if point_idx < len(li_points):
                kernel_points_count[p] += 1
                if li_points[point_idx] is not None:
                    kernel_points_loaded[p] += 1
                point_idx += 1
    for kernel, total in sorted(kernel_points_count.items()):
        loaded = kernel_points_loaded[kernel]
    if points_loaded == 0:
        if len(li_g) > 0:
            try:
                first_kernel = MACHSUITE_KERNEL[0] if len(MACHSUITE_KERNEL) > 0 else poly_KERNEL[0]
                if first_kernel in gp:
                    first_file = gp[first_kernel][1] if len(gp[first_kernel]) > 1 else None
                    if first_file:
                        path_parts = first_file.split(os.sep)
                        benchmark = None
                        for part in path_parts:
                            if part in ['machsuite', 'poly']:
                                benchmark = part
                                break
                        if benchmark:
                            first_points_file = os.path.join(get_root_path(), 'two_tower_dataset', 'points', benchmark, first_kernel, 'points_list.pkl')
                        else:
                            first_dir = os.path.dirname(first_file)
                            first_points_file = os.path.join(first_dir, 'points_list.pkl')
            except Exception as e:
                pass
    li_len = len(li_g)
    l_t, l_v = (int(li_len * 0.7), int(li_len * 0.15))
    from numpy import random
    rinx = permutation(range(li_len))
    li_r = [rinx[0:l_t], rinx[l_t:l_t + l_v], rinx[l_t + l_v:]]
    li = []
    li_code = []
    li_points_split = []
    for i in li_r:
        tmp = []
        tmp_1 = []
        tmp_points = []
        for j in i:
            tmp.append(li_g[j])
            tmp_1.append(li_c[j])
            tmp_points.append(li_points[j])
        li.append(tmp)
        li_code.append(tmp_1)
        li_points_split.append(tmp_points)
    saver.log_info(f'{len(li_g)} graphs in total: {len(li[0])} train {len(li[1])} val {len(li[2])} test')
    saver.log_info(f'{len(li_c)} codes in total: {len(li_code[0])} train {len(li_code[1])} val {len(li_code[2])} test')
    train_loader = DataLoader(li[0], batch_size=FLAGS.batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(li[1], batch_size=FLAGS.batch_size, pin_memory=True)
    test_loader = DataLoader(li[2], batch_size=FLAGS.batch_size, pin_memory=True)
    train_codes = li_code[0]
    val_codes = li_code[1]
    test_codes = li_code[2]
    train_points = li_points_split[0]
    val_points = li_points_split[1]
    test_points = li_points_split[2]
    if len(li[0]) == 0:
        saver.error(f'No training data found! gp dictionary is empty or contains no valid kernels.')
        saver.error(f'This usually happens when force_regen=False and processed_file_names_dict structure is unexpected.')
        saver.error(f'Please try setting force_regen=True to regenerate the dataset, or check the dataset structure.')
        raise ValueError('No training data available. Please set force_regen=True or check dataset configuration.')
    edge_dim = train_loader.dataset[0].edge_attr.shape[1]
    max_degree = -1
    for data in li[2]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in li[2]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    if FLAGS.comparative_if:
        if FLAGS.comparative_model == 'pna':
            model = Net(deg, edge_dim).to(FLAGS.device)
        else:
            model = Net().to(FLAGS.device)
    else:
        model = Net().to(FLAGS.device)
    if FLAGS.model_path != None:
        old_state_dict = torch.load(FLAGS.model_path, map_location=torch.device(FLAGS.device))
        missing_keys, unexpected_keys = model.load_state_dict(old_state_dict, strict=False)
        saver.info(f'Loaded model from {FLAGS.model_path} (partial load, strict=False)')
        if missing_keys:
            saver.info(f'Missing keys (will be initialized randomly): {len(missing_keys)} keys')
            if len(missing_keys) <= 10:
                saver.info(f'  {missing_keys}')
        if unexpected_keys:
            if not FLAGS.use_causal:
                causal_keys = [k for k in unexpected_keys if 'pragma_encoder' in k or 'causal_head' in k]
                if causal_keys:
                    saver.info(f'Ignored causal module keys (use_causal=False): {len(causal_keys)} keys')
                    if len(causal_keys) <= 10:
                        saver.info(f'  {causal_keys}')
            else:
                saver.info(f'Unexpected keys (ignored): {len(unexpected_keys)} keys')
                if len(unexpected_keys) <= 10:
                    saver.info(f'  {unexpected_keys}')
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    if len(val_loader) > 0:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=30, verbose=True, min_lr=1e-06, cooldown=10)
    else:
        scheduler = None
    train_losses = []
    val_losses = []
    test_losses = []
    epochs = range(FLAGS.epoch_num)
    best_test_loss = float('inf')
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 50
    patience_counter = 0
    val_patience_counter = 0
    best_model_state = None
    val_loss_window = []
    window_size = 5
    for epoch in epochs:
        plot_test = False
        timer = OurTimer()
        saver.log_info(f'Epoch {epoch + 1} train')
        loss, loss_dict_train = train(epoch, model, train_loader, train_codes, optimizer, train_points=train_points)
        if len(val_loader) > 0:
            saver.log_info(f'\nEpoch {epoch + 1} val')
            val, loss_dict_val = test(val_loader, val_codes, 'val', model, epoch, design_points=val_points)
            scheduler.step(val)
        saver.log_info(f'\nEpoch {epoch + 1} test')
        testr, loss_dict_test = test(test_loader, test_codes, 'test', model, epoch, plot_test, test_losses, design_points=test_points)
        saver.log_info(f'\nTrain loss breakdown {loss_dict_train}')
        saver.log_info(f'\nTest loss breakdown {loss_dict_test}')
        if len(val_loader) > 0:
            saver.log_info(f'\nVal loss breakdown {loss_dict_val}')
            saver.log_info('Epoch: {:03d}, Train Loss: {:.4f}, Val loss: {:.4f}, Test: {:.4f}) Time: {}'.format(epoch + 1, loss, val, testr, timer.time_and_clear()))
            val_losses.append(val)
        else:
            saver.log_info('Epoch: {:03d}, Loss: {:.4f}, Train loss: {:.3f}, Test: {:.3f}) Time: {}'.format(epoch + 1, loss, loss, testr, timer.time_and_clear()))
        train_losses.append(loss)
        test_losses.append(testr)
        if testr < best_test_loss:
            best_test_loss = testr
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, join(get_root_path(), 'save_models_and_data/regression_model_state_dict.pth'))
            saver.log_info(f'[Best Model] Saved at epoch {best_epoch} with test loss {best_test_loss:.4f}')
        else:
            patience_counter += 1
        if len(val_loader) > 0:
            val_loss_window.append(val)
            if len(val_loss_window) > window_size:
                val_loss_window.pop(0)
            val_avg = sum(val_loss_window) / len(val_loss_window) if val_loss_window else val
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                val_patience_counter = 0
                if epoch % 10 == 0:
                    saver.log_info(f'[Early Stop] Val loss improved: {val_avg:.4f} (raw: {val:.4f}) < previous best, reset patience')
            else:
                val_patience_counter += 1
                if val_patience_counter % 10 == 0:
                    saver.log_info(f'[Early Stop] Val loss no improvement for {val_patience_counter}/{patience} epochs (val_avg: {val_avg:.4f}, raw: {val:.4f}, best: {best_val_loss:.4f})')
                if val_patience_counter >= patience:
                    saver.log_info(f'[Early Stop] Validation loss no improvement for {patience} epochs.')
                    saver.log_info(f'[Early Stop] Best test loss: {best_test_loss:.4f} at epoch {best_epoch}')
                    saver.log_info(f'[Early Stop] Restoring best model from epoch {best_epoch}')
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        if len(train_losses) > 50:
            if len(set(train_losses[-50:])) == 1 and len(set(test_losses[-50:])) == 1:
                saver.log_info('[Early Stop] Loss unchanged for 50 epochs')
                break
    if best_model_state is not None and epoch + 1 != best_epoch:
        saver.log_info(f'[Final] Loading best model from epoch {best_epoch} (test loss: {best_test_loss:.4f})')
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, join(get_root_path(), 'save_models_and_data/regression_model_state_dict.pth'))
    epochs = range(epoch + 1)
    import matplotlib
    if os.environ.get('DISPLAY', '') == '':
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    if len(val_loader) > 0:
        plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.plot(epochs, test_losses, 'r', label='Testing loss')
    plt.title('Training, Validation, and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(saver.get_log_dir(), 'losses.png'), bbox_inches='tight')
    plt.show()
    best_test_epoch = test_losses.index(min(test_losses)) + 1
    best_train_epoch = train_losses.index(min(train_losses)) + 1
    saver.log_info(f'min test loss at epoch: {best_test_epoch} (loss: {min(test_losses):.4f})')
    saver.log_info(f'min train loss at epoch: {best_train_epoch} (loss: {min(train_losses):.4f})')
    if len(val_loader) > 0:
        best_val_epoch = val_losses.index(min(val_losses)) + 1
        saver.log_info(f'min val loss at epoch: {best_val_epoch} (loss: {min(val_losses):.4f})')
    saver.log_info(f'[Final] Best model saved: epoch {best_epoch}, test loss: {best_test_loss:.4f}')

def train(epoch, model, train_loader, train_codes, optimizer, train_points=None):
    model.train()
    total_loss = 0
    correct = 0
    i = 0
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if FLAGS.task == 'regression':
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    else:
        target_list = [_target_list[0]]
    loss_dict = {}
    for t in target_list:
        loss_dict[t] = 0.0
    causal_loss_total = 0.0
    num_causal_pairs = 0
    intervention_delta_train: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    inx = 0
    for data in tqdm(train_loader, position=0, total=len(train_loader), file=sys.stdout):
        code = train_codes[inx:inx + FLAGS.batch_size]
        data = data.to(FLAGS.device)
        optimizer.zero_grad()
        design_points = None
        if FLAGS.use_causal:
            if hasattr(data, 'batch') and data.batch is not None:
                batch_size = int(data.batch.max().item() + 1)
            else:
                batch_size = len(code)
            design_points = []
            if train_points is not None:
                for i in range(batch_size):
                    idx = inx + i
                    if idx < len(train_points):
                        point = train_points[idx]
                        design_points.append(point)
                    else:
                        design_points.append(None)
                if inx == 0:
                    valid_count = sum((1 for dp in design_points if dp is not None))
                    saver.log_info(f'[Causal Debug] Batch 0: extracted {valid_count}/{len(design_points)} valid design_points from train_points')
                    if valid_count > 0:
                        saver.log_info(f"[Causal Debug] First design_point keys: {(list(design_points[0].keys())[:5] if design_points[0] else 'None')}")
                if all((dp is None for dp in design_points)):
                    design_points = None
            else:
                if inx == 0:
                    saver.log_info(f'[Causal Debug] train_points is None, design_points will be None')
                design_points = None
        fusion_w = None
        if getattr(FLAGS, 'causal_main_pred_mode', 'replace') == 'fusion':
            import math
            ramp = max(1, int(getattr(FLAGS, 'causal_fusion_ramp_epochs', 10)))
            progress = min(1.0, float(epoch) / float(ramp))
            schedule = getattr(FLAGS, 'causal_fusion_schedule', 'linear')
            if schedule == 'exp':
                w_prog = 1.0 - math.exp(-5.0 * progress)
            else:
                w_prog = progress
            fusion_w = float(getattr(FLAGS, 'causal_fusion_w_max', 1.0)) * w_prog
        out, loss, loss_dict_ = model.to(FLAGS.device)(data, code, design_point=design_points, fusion_w=fusion_w)
        if FLAGS.use_causal and epoch == 0 and (inx == 0):
            try:
                first_t = target_list[0] if isinstance(target_list, list) and len(target_list) > 0 else None
                if first_t is not None and first_t in out:
                    saver.log_info(f"[GradCheck] out_dict[{first_t}].requires_grad={getattr(out[first_t], 'requires_grad', None)}")
                else:
                    saver.log_info('[GradCheck] out_dict first target not found (unexpected)')
            except Exception as _e:
                saver.log_info(f'[GradCheck] failed to log out_dict requires_grad: {_e}')
        causal_loss = None
        if FLAGS.use_causal and FLAGS.causal_lambda > 0:
            try:
                causal_loss = compute_causal_loss(model, data, code, design_points, target_list, FLAGS, out_dict=out, intervention_delta_accumulator=intervention_delta_train)
                if FLAGS.use_causal and epoch == 0 and (inx == 0):
                    if causal_loss is None:
                        saver.log_info('[GradCheck] causal_loss=None')
                    else:
                        saver.log_info(f'[GradCheck] causal_loss.requires_grad={causal_loss.requires_grad}, grad_fn={(type(causal_loss.grad_fn).__name__ if causal_loss.grad_fn is not None else None)}')
                if causal_loss is not None and causal_loss.item() > 0:
                    loss = loss + FLAGS.causal_lambda * causal_loss
                    causal_loss_total += causal_loss.item()
                    num_causal_pairs += 1
            except Exception as e:
                if epoch == 0 and inx == 0:
                    import traceback
                    traceback.print_exc()
        elif FLAGS.use_causal and FLAGS.causal_lambda == 0:
            try:
                with torch.no_grad():
                    compute_causal_loss(model, data, code, design_points, target_list, FLAGS, out_dict=out, intervention_delta_accumulator=intervention_delta_train)
            except Exception:
                pass
        if FLAGS.use_causal and getattr(FLAGS, 'causal_entropy_beta', 0.0) > 0:
            beta_ent = float(getattr(FLAGS, 'causal_entropy_beta', 0.0))
            if beta_ent != 0.0 and hasattr(model, '_last_alpha_matrix'):
                alpha_matrix = getattr(model, '_last_alpha_matrix', None)
                if alpha_matrix is not None:
                    eps = 1e-12
                    pragma_mask_batch = getattr(model, '_last_pragma_mask_batch', None)
                    if pragma_mask_batch is not None and isinstance(pragma_mask_batch, (list, tuple)) and (len(pragma_mask_batch) == alpha_matrix.shape[0]):
                        mask = torch.stack([m.to(alpha_matrix.device).bool() for m in pragma_mask_batch], dim=0)
                    else:
                        mask = None
                    if mask is not None:
                        valid_counts = mask.sum(dim=1).clamp(min=1).to(alpha_matrix.device)
                        alpha_v = alpha_matrix.masked_fill(~mask.unsqueeze(-1), 0.0)
                        ent = -(alpha_v * torch.log(alpha_v + eps)).sum(dim=1)
                        ent_norm = ent / torch.log(valid_counts.unsqueeze(-1).float() + eps)
                        ent_mean = ent_norm.mean()
                    else:
                        ent = -(alpha_matrix * torch.log(alpha_matrix + eps)).sum(dim=1)
                        ent_mean = ent.mean()
                    loss = loss - beta_ent * ent_mean
        if FLAGS.use_causal and FLAGS.causal_reg_beta > 0:
            if hasattr(model, '_last_alpha_matrix'):
                alpha_matrix = model._last_alpha_matrix
                if alpha_matrix is not None:
                    alpha_reg = FLAGS.causal_reg_beta * torch.abs(alpha_matrix).mean()
                    loss = loss + alpha_reg
        loss.backward()
        if FLAGS.use_causal and epoch == 0 and (inx == 0):
            try:
                causal_grad_norms = []
                mlp_grad_norms = []
                for name, p in model.named_parameters():
                    if 'pragma_encoder' in name or 'causal_head' in name:
                        if p.grad is not None:
                            causal_grad_norms.append((name, float(p.grad.detach().norm().item())))
                    if 'MLPs' in name:
                        if p.grad is not None:
                            mlp_grad_norms.append((name, float(p.grad.detach().norm().item())))
                if len(causal_grad_norms) == 0:
                    saver.log_info('[GradCheck] No grads found for pragma_encoder/causal_head (unexpected)')
                else:
                    causal_grad_norms.sort(key=lambda x: x[1], reverse=True)
                    topk = causal_grad_norms[:10]
                    saver.log_info('[GradCheck] Top grad norms for pragma_encoder/causal_head:')
                    for k, v in topk:
                        saver.log_info(f'  {k}: {v:.6e}')
                    if mlp_grad_norms:
                        mlp_grad_norms.sort(key=lambda x: x[1], reverse=True)
                        topk_mlp = mlp_grad_norms[:10]
                        saver.log_info('[GradCheck] Top grad norms for MLPs (should be ~0 if hard-replaced):')
                        for k, v in topk_mlp:
                            saver.log_info(f'  {k}: {v:.6e}')
                    else:
                        saver.log_info('[GradCheck] MLPs grad is None (good sign: main loss not training MLP heads)')
            except Exception as _e:
                saver.log_info(f'[GradCheck] failed to log grad norms: {_e}')
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        total_loss += loss.item() * data.num_graphs
        for t in target_list:
            loss_dict[t] += loss_dict_[t].item()
        optimizer.step()
        inx += FLAGS.batch_size
    if FLAGS.task == 'regression':
        avg_loss = total_loss / len(train_loader.dataset)
        avg_loss_dict = {key: v / len(train_loader) for key, v in loss_dict.items()}
        if FLAGS.use_causal and FLAGS.causal_lambda > 0 and (num_causal_pairs > 0):
            avg_causal_loss = causal_loss_total / num_causal_pairs
        if FLAGS.use_causal and FLAGS.task == 'regression':
            total_iv = sum((len(intervention_delta_train[t]) for t in target_list))
        return (avg_loss, avg_loss_dict)
    else:
        return (1 - correct / total_loss, {key: v / len(train_loader) for key, v in loss_dict.items()})

def inference_loss_function(pred, true):
    return (pred - true) ** 2

def test(loader, codes, tvt, model, epoch, plot_test=False, test_losses=[-1], design_points=None):
    model.eval()
    inference_loss = 0
    correct, total = (0, 0)
    loss_dict = {}
    i = 0
    points_dict = OrderedDict()
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if FLAGS.task == 'regression':
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    else:
        target_list = [_target_list[0]]
    for t in target_list:
        loss_dict[t] = 0.0
    for target_name in target_list:
        points_dict[target_name] = {'true': [], 'pred': []}
    inx = 0
    causal_eval_total = 0.0
    causal_eval_count = 0
    intervention_delta_eval: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    intervention_batches_with_pairs = 0
    _raw_eval_pairs = int(getattr(FLAGS, 'causal_max_pairs_eval', 0) or 0)
    eval_pair_cap = _raw_eval_pairs if _raw_eval_pairs > 0 else None
    effective_iv_per_batch_cap = eval_pair_cap if eval_pair_cap is not None else int(getattr(FLAGS, 'causal_max_pairs_per_batch', 10) or 10)
    print(f'[Causal Debug] ===== Starting {tvt} loop =====')
    print(f'[Causal Debug] FLAGS.use_causal = {FLAGS.use_causal}')
    print(f'[Causal Debug] type(FLAGS.use_causal) = {type(FLAGS.use_causal)}')
    print(f'[Causal Debug] loader length = {len(loader)}')
    if FLAGS.use_causal and len(loader) > 0:
        print(f'[Causal Debug] Starting {tvt} loop: use_causal={FLAGS.use_causal}, dataset_size={len(loader.dataset)}')
        saver.log_info(f'[Causal Debug] Starting {tvt} loop: use_causal={FLAGS.use_causal}, dataset_size={len(loader.dataset)}')
        try:
            first_sample = loader.dataset[0]
            has_point = hasattr(first_sample, 'point')
            point_value = getattr(first_sample, 'point', None)
            print(f'[Causal Debug] First sample: has_point={has_point}, point={point_value is not None}')
            saver.log_info(f'[Causal Debug] First sample: has_point={has_point}, point={point_value is not None}')
            if point_value is not None:
                print(f'[Causal Debug] First sample point keys: {list(point_value.keys())[:5]}')
                saver.log_info(f'[Causal Debug] First sample point keys: {list(point_value.keys())[:5]}')
        except Exception as e:
            print(f'[Causal Debug] Error checking first sample: {e}')
            saver.log_info(f'[Causal Debug] Error checking first sample: {e}')
    else:
        print(f'[Causal Debug] Skipping debug (use_causal={FLAGS.use_causal}, len(loader)={len(loader)})')
    for data in tqdm(loader, position=0, total=len(loader), file=sys.stdout):
        _point_cache = getattr(data, 'point', None)
        data = data.to(FLAGS.device)
        if _point_cache is not None:
            try:
                setattr(data, 'point', _point_cache)
            except Exception:
                pass
        code = codes[inx:inx + FLAGS.batch_size]
        design_points_batch = None
        if FLAGS.use_causal:
            if hasattr(data, 'batch') and data.batch is not None:
                batch_size = int(data.batch.max().item() + 1)
            else:
                batch_size = len(code)
            design_points_batch = []
            if design_points is not None:
                for i in range(batch_size):
                    idx = inx + i
                    if idx < len(design_points):
                        point = design_points[idx]
                        design_points_batch.append(point)
                    else:
                        design_points_batch.append(None)
                if inx == 0:
                    valid_count = sum((1 for dp in design_points_batch if dp is not None))
                    saver.log_info(f'[Causal Debug] Batch 0: extracted {valid_count}/{len(design_points_batch)} valid design_points from design_points list')
                    if valid_count > 0:
                        saver.log_info(f"[Causal Debug] First design_point keys: {(list(design_points_batch[0].keys())[:5] if design_points_batch[0] else 'None')}")
                if all((dp is None for dp in design_points_batch)):
                    design_points_batch = None
            else:
                if inx == 0:
                    saver.log_info(f'[Causal Debug] design_points parameter is None, design_points_batch will be None')
                design_points_batch = None
        design_points = design_points_batch
        if FLAGS.use_causal and inx == 0:
            saver.log_info(f'[Causal Debug] Batch 0: use_causal={FLAGS.use_causal}, design_points={design_points is not None}, design_points_len={(len(design_points) if design_points else 0)}')
            if design_points:
                valid_count = sum((1 for dp in design_points if dp is not None))
                saver.log_info(f'[Causal Debug] Valid design_points: {valid_count}/{len(design_points)}')
                if valid_count > 0:
                    saver.log_info(f"[Causal Debug] First design_point sample: {(list(design_points[0].keys())[:3] if design_points[0] else 'None')}")
        fusion_w = None
        if getattr(FLAGS, 'causal_main_pred_mode', 'replace') == 'fusion':
            import math
            ramp = max(1, int(getattr(FLAGS, 'causal_fusion_ramp_epochs', 10)))
            progress = min(1.0, float(epoch) / float(ramp))
            schedule = getattr(FLAGS, 'causal_fusion_schedule', 'linear')
            if schedule == 'exp':
                w_prog = 1.0 - math.exp(-5.0 * progress)
            else:
                w_prog = progress
            fusion_w = float(getattr(FLAGS, 'causal_fusion_w_max', 1.0)) * w_prog
        out_dict, loss, loss_dict_ = model.to(FLAGS.device)(data, code, design_point=design_points, fusion_w=fusion_w)
        if FLAGS.use_causal and design_points is not None:
            try:
                _iv_prev = sum((len(intervention_delta_eval[t]) for t in target_list))
                with torch.no_grad():
                    causal_eval = compute_causal_loss(model, data, code, design_points, target_list, FLAGS, out_dict=out_dict, intervention_delta_accumulator=intervention_delta_eval, max_pairs_override=eval_pair_cap)
                _iv_new = sum((len(intervention_delta_eval[t]) for t in target_list))
                if _iv_new > _iv_prev:
                    intervention_batches_with_pairs += 1
                if causal_eval is not None and getattr(FLAGS, 'causal_lambda', 0.0) > 0:
                    causal_eval_total += float(causal_eval.detach().item())
                    causal_eval_count += 1
            except Exception:
                pass
        if FLAGS.use_causal and inx == 0:
            has_alpha = hasattr(model, '_last_alpha_matrix')
            alpha = getattr(model, '_last_alpha_matrix', None)
            saver.log_info(f"[Causal Debug] After forward: has_alpha={has_alpha}, alpha={alpha is not None}, alpha_shape={(alpha.shape if alpha is not None else 'None')}")
        if FLAGS.task == 'regression':
            total += loss.item()
            for t in target_list:
                loss_dict[t] += loss_dict_[t].item()
        else:
            loss, pred = torch.max(out_dict[FLAGS.target[0]], 1)
            labels = _get_y_with_target(data, FLAGS.target[0])
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        for target_name in target_list:
            if FLAGS.subtask == 'inference':
                saver.info(f'{target_name}')
            if FLAGS.task == 'class':
                out = pred
            elif FLAGS.encode_log and 'perf' in target_name:
                out = out_dict['perf']
            else:
                out = out_dict[target_name]
            for i in range(len(out)):
                out_value = out[i].item()
                if FLAGS.encode_log and target_name == 'actual_perf':
                    out_value = 2 ** out_value * (1 / FLAGS.normalizer)
                if FLAGS.subtask == 'inference':
                    inference_loss += inference_loss_function(out_value, _get_y_with_target(data, target_name)[i].item())
                    if out_value != _get_y_with_target(data, target_name)[i].item():
                        saver.info(f'data {i} actual value: {_get_y_with_target(data, target_name)[i].item():.2f}, predicted value: {out_value:.2f}')
                points_dict[target_name]['pred'].append((_get_y_with_target(data, target_name)[i].item(), out_value))
                points_dict[target_name]['true'].append((_get_y_with_target(data, target_name)[i].item(), _get_y_with_target(data, target_name)[i].item()))
        inx += FLAGS.batch_size
    try:
        if FLAGS.use_causal:
            if hasattr(model, '_last_alpha_matrix'):
                alpha = getattr(model, '_last_alpha_matrix', None)
                if alpha is not None:
                    import torch as _torch
                    from os.path import join as _join
                    import os
                    obj_dir = saver.get_obj_dir()
                    os.makedirs(obj_dir, exist_ok=True)
                    fn = f'alpha_matrix_epoch_{epoch + 1}_{tvt}.pt'
                    filepath = _join(obj_dir, fn)
                    _torch.save({'alpha_matrix': alpha.detach().cpu(), 'target_list': target_list, 'pragma_ids_batch': getattr(model, '_last_pragma_ids_batch', None), 'pragma_mask_batch': [m.detach().cpu() if hasattr(m, 'detach') else m for m in getattr(model, '_last_pragma_mask_batch', None) or []]}, filepath)
    if FLAGS.plot_pred_points and tvt == 'test' and (plot_test or (test_losses and total / len(loader) < min(test_losses))):
        from utils import plot_points, plot_points_with_subplot
        saver.log_info(f'@@@ plot_pred_points')
        if not FLAGS.multi_target:
            plot_points({f'{FLAGS.target[0]}-pred_points': points_dict[f'{FLAGS.target[0]}']['pred'], f'{FLAGS.target[0]}-true_points': points_dict[f'{FLAGS.target[0]}']['true']}, f'epoch_{epoch + 1}_{tvt}', saver.get_log_dir())
            print(f'done plotting with {correct} corrects out of {total}')
        else:
            assert isinstance(FLAGS.target, list)
            plot_points_with_subplot(points_dict, f'epoch_{epoch + 1}_{tvt}', saver.get_log_dir(), target_list)
    if FLAGS.subtask == 'inference':
        if FLAGS.task == 'regression':
            result_df = _report_rmse_etc(points_dict, f'epoch {epoch}:', True)
        elif FLAGS.task == 'class':
            report_class_loss(points_dict)
    if FLAGS.use_causal and FLAGS.task == 'regression':
        n_iv_per_target = max((len(intervention_delta_eval[t]) for t in target_list), default=0)
        _cap_note = f'pair_count<{effective_iv_per_batch_cap} means this split had at most that many Hamming 1–2 pairs in the contributing batch(es), not a cap artifact.' if n_iv_per_target < effective_iv_per_batch_cap else f'pair_count reached per-batch cap ({effective_iv_per_batch_cap}).'
        total_iv = sum((len(intervention_delta_eval[t]) for t in target_list))')
    if FLAGS.task == 'regression':
        if FLAGS.subtask == 'inference':
            return (total / len(loader), {key: v / len(loader) for key, v in loss_dict.items()}, inference_loss / len(loader) / FLAGS.batch_size)
        else:
            return (total / len(loader), {key: v / len(loader) for key, v in loss_dict.items()})
    else:
        return (1 - correct / total, {key: v / len(loader) for key, v in loss_dict.items()})
