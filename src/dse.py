from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
try:
    from src.config import FLAGS
except ImportError:
    from config import FLAGS
from src.saver import saver
from datetime import datetime
from src.utils import MLP, load, save, get_save_path, argsort, get_root_path, get_src_path, _get_y_with_target, _get_y
from src.programl_data import print_data_stats, _check_any_in_str, NON_OPT_PRAGMAS, WITH_VAR_PRAGMAS, _in_between, _encode_edge_dict, _encode_edge_torch, _encode_X_torch, create_edge_index
from src.model import Net
from src.parameter import DesignSpace, DesignPoint, DesignParameter, get_default_point, topo_sort_param_ids, compile_design_space, gen_key_from_design_point
from src.config_ds import build_config
from src.result import Result
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
import json
import ast
import re
import os
from math import ceil, inf, exp, log10
from os.path import join, dirname
import time
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Deque, Dict, List, Optional, Set, Union, Generator, Any
import sys
import copy
import itertools
import networkx as nx
from collections import OrderedDict
from glob import glob
import pickle
from torch.nn import Sequential, Linear, ReLU
from typing import NamedTuple, Any, Callable
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from random import uniform, randint, shuffle
from sklearn.preprocessing import OneHotEncoder
_DSE_RUN_ID = os.environ.get('DSE_RUN_ID')
if not _DSE_RUN_ID:
    _DSE_RUN_ID = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def _get_best_result_run_dir() -> str:
    root = join(get_root_path(), 'best_result_runs')
    variant = 'causal_on' if getattr(FLAGS, 'use_causal', False) else 'causal_off'
    return join(root, variant)
BEST_RESULT_RUN_DIR = _get_best_result_run_dir()
os.makedirs(BEST_RESULT_RUN_DIR, exist_ok=True)

def _dump_run_meta_once():
    meta_path = join(BEST_RESULT_RUN_DIR, 'run_meta.json')
    if os.path.exists(meta_path):
        return
    try:
        meta = {'run_id': _DSE_RUN_ID, 'time': datetime.now().isoformat(), 'cwd': os.getcwd(), 'llm_model': getattr(FLAGS, 'llm_model', None), 'explorer': getattr(FLAGS, 'explorer', None), 'benchmarks': getattr(FLAGS, 'benchmarks', None), 'dataset': getattr(FLAGS, 'dataset', None), 'norm_method': getattr(FLAGS, 'norm_method', None), 'use_causal': getattr(FLAGS, 'use_causal', None), 'device': getattr(FLAGS, 'device', None)}
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
SAVE_DIR = join(get_root_path(), f'save_models_and_data')
SAVE_DIR_CLASS = join(get_root_path(), f'save_models_and_data')
_REF_PERF_MAP_CACHE: Optional[Dict[str, float]] = None
_REF_NORM_BOUNDS_CACHE: Optional[Dict[str, Dict[str, float]]] = None

def _load_ref_perf_map() -> Optional[Dict[str, float]]:
    global _REF_PERF_MAP_CACHE
    if _REF_PERF_MAP_CACHE is not None:
        return _REF_PERF_MAP_CACHE
    ref_path = join(SAVE_DIR, 'ref_perf_map.json')
    if not os.path.exists(ref_path):
        _REF_PERF_MAP_CACHE = None
        return None
    try:
        with open(ref_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'ref_perf_map' in data and isinstance(data.get('ref_perf_map'), dict):
            raw_map = data['ref_perf_map']
        elif isinstance(data, dict):
            raw_map = data
        else:
            _REF_PERF_MAP_CACHE = None
            return None
        parsed: Dict[str, float] = {}
        for k, v in raw_map.items():
            try:
                parsed[str(k)] = float(v)
            except Exception:
                continue
        _REF_PERF_MAP_CACHE = parsed if parsed else None
        return _REF_PERF_MAP_CACHE
    except Exception:
        _REF_PERF_MAP_CACHE = None
        return None

def _lookup_ref_perf(ref_perf_map: Optional[Dict[str, float]], kernel_name: Optional[str]) -> Optional[float]:
    if not ref_perf_map or not kernel_name:
        return None
    candidates: List[str] = []
    k = str(kernel_name).strip()
    if k:
        candidates.append(k)
        if k.endswith('_processed_result'):
            candidates.append(k[:-len('_processed_result')])
        else:
            candidates.append(f'{k}_processed_result')
        candidates.append(os.path.basename(k))
    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    for c in ordered:
        if c in ref_perf_map:
            try:
                return float(ref_perf_map[c])
            except Exception:
                continue
    return None

def _load_ref_norm_bounds() -> Optional[Dict[str, Dict[str, float]]]:
    global _REF_NORM_BOUNDS_CACHE
    if _REF_NORM_BOUNDS_CACHE is not None:
        return _REF_NORM_BOUNDS_CACHE
    path = join(SAVE_DIR, 'ref_norm_bounds.json')
    if not os.path.exists(path):
        _REF_NORM_BOUNDS_CACHE = None
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get('norm_bounds'), dict):
            raw_map = data.get('norm_bounds')
        elif isinstance(data, dict):
            raw_map = data
        else:
            _REF_NORM_BOUNDS_CACHE = None
            return None
        parsed: Dict[str, Dict[str, float]] = {}
        for k, v in raw_map.items():
            if not isinstance(v, dict):
                continue
            entry: Dict[str, float] = {}
            aliases = {'perf_min': ['perf_min'], 'perf_max': ['perf_max'], 'util_min': ['util_min', 'util_sum_min'], 'util_max': ['util_max', 'util_sum_max']}
            ok = True
            for std_key, cand_keys in aliases.items():
                vv = None
                for ck in cand_keys:
                    if ck in v:
                        vv = v.get(ck)
                        break
                try:
                    entry[std_key] = float(vv)
                except Exception:
                    ok = False
                    break
            if ok:
                parsed[str(k)] = entry
        _REF_NORM_BOUNDS_CACHE = parsed if parsed else None
        return _REF_NORM_BOUNDS_CACHE
    except Exception:
        _REF_NORM_BOUNDS_CACHE = None
        return None

def _lookup_ref_norm_bounds(bounds_map: Optional[Dict[str, Dict[str, float]]], kernel_name: Optional[str]) -> Optional[Dict[str, float]]:
    if not bounds_map or not kernel_name:
        return None
    candidates: List[str] = []
    k = str(kernel_name).strip()
    if k:
        candidates.append(k)
        if k.endswith('_processed_result'):
            candidates.append(k[:-len('_processed_result')])
        else:
            candidates.append(f'{k}_processed_result')
        candidates.append(os.path.basename(k))
    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    for c in ordered:
        if c in bounds_map and isinstance(bounds_map[c], dict):
            b = bounds_map[c]
            try:
                return {'perf_min': float(b['perf_min']), 'perf_max': float(b['perf_max']), 'util_min': float(b['util_min']), 'util_max': float(b['util_max'])}
            except Exception:
                continue
    return None

def _find_local_codebert_path() -> str:
    env_path = os.environ.get('CODEBERT_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    base = join(get_root_path(), 'codebert', 'models--microsoft--codebert-base', 'snapshots')
    if os.path.isdir(base):
        for snap in sorted(os.listdir(base), reverse=True):
            snap_dir = join(base, snap)
            if not os.path.isdir(snap_dir):
                continue
            cfg = join(snap_dir, 'config.json')
            has_bin = os.path.exists(join(snap_dir, 'pytorch_model.bin'))
            has_st = os.path.exists(join(snap_dir, 'model.safetensors'))
            if os.path.exists(cfg) and (has_bin or has_st):
                return snap_dir
    return 'microsoft/codebert-base'

def _rebuild_encoders_from_graph(g) -> dict:
    from src.programl_data import _encode_X_dict, _encode_edge_dict
    enc_ntype = OneHotEncoder(handle_unknown='ignore')
    enc_ptype = OneHotEncoder(handle_unknown='ignore')
    enc_itype = OneHotEncoder(handle_unknown='ignore')
    enc_ftype = OneHotEncoder(handle_unknown='ignore')
    enc_btype = OneHotEncoder(handle_unknown='ignore')
    enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
    enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')
    x_dict = _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None, itypes=None, ftypes=None, btypes=None, obj=None)
    e_dict = _encode_edge_dict(g, ftypes=None, ptypes=None)
    enc_ntype.fit(x_dict['X_ntype'])
    enc_ptype.fit(x_dict['X_ptype'])
    enc_itype.fit(x_dict['X_itype'])
    enc_ftype.fit(x_dict['X_ftype'])
    enc_btype.fit(x_dict['X_btype'])
    enc_ftype_edge.fit(e_dict['X_ftype'])
    enc_ptype_edge.fit(e_dict['X_ptype'])
    return {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype, 'enc_itype': enc_itype, 'enc_ftype': enc_ftype, 'enc_btype': enc_btype, 'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge}

def gin_mlp_func() -> Callable:

    def mlp_func(in_channels: int, out_channels: int, bias: bool):
        return Sequential(Linear(in_channels, out_channels, bias=bias), ReLU(), Linear(out_channels, out_channels, bias=bias))
    return mlp_func
out_dim = FLAGS.out_dim
gin_mlp_func = gin_mlp_func()
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen', 'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']

def _convert_model_type_if_needed(model_type_value):
    if isinstance(model_type_value, str):
        from CoGNN.layers import ModelType
        return ModelType.from_string(model_type_value)
    return model_type_value
temp_model_type = _convert_model_type_if_needed(FLAGS.temp_model_type)
env_model_type = _convert_model_type_if_needed(FLAGS.env_model_type)
act_model_type = _convert_model_type_if_needed(FLAGS.act_model_type)
gumbel_args = GumbelArgs(learn_temp=FLAGS.learn_temp, temp_model_type=temp_model_type, tau0=FLAGS.tau0, temp=FLAGS.temp, gin_mlp_func=gin_mlp_func)
env_args = EnvArgs(model_type=env_model_type, num_layers=FLAGS.env_num_layers, env_dim=FLAGS.env_dim, layer_norm=FLAGS.layer_norm, skip=FLAGS.skip, batch_norm=FLAGS.batch_norm, dropout=FLAGS.dropout, in_dim=FLAGS.num_features, out_dim=FLAGS.D, dec_num_layers=FLAGS.dec_num_layers, gin_mlp_func=gin_mlp_func, act_type=ActivationType.RELU)
action_args = ActionNetArgs(model_type=act_model_type, num_layers=FLAGS.act_num_layers, hidden_dim=FLAGS.act_dim, dropout=FLAGS.dropout, act_type=ActivationType.RELU, gin_mlp_func=gin_mlp_func, env_dim=FLAGS.env_dim)

class GNNModel:

    def __init__(self, path, saver, multi_target=True, task='regression', num_layers=FLAGS.num_layers, D=FLAGS.D, target=FLAGS.target, model_name=f'{FLAGS.model_tag}_model_state_dict.pth', encoder_name='encoders', deg=None, edge_dim=0, kernel_graph=None):
        model_name = f'{task}_model_state_dict.pth'
        self.log = saver
        self.path = path
        if task == 'regression':
            if FLAGS.model_path == None:
                self.model_path = join(self.path, model_name)
            else:
                self.model_path = FLAGS.model_path
        elif FLAGS.class_model_path == None:
            self.model_path = join(self.path, model_name)
        else:
            self.model_path = FLAGS.class_model_path
        if FLAGS.encoder_path == None:
            self.encoder_path = join(self.path, encoder_name)
        else:
            self.encoder_path = FLAGS.encoder_path
        self.num_features = FLAGS.num_features
        self.model = Net(deg=deg, edge_dim=edge_dim).to(FLAGS.device)
        state_dict = torch.load(join(self.model_path), map_location=torch.device('cuda:0'))
        load_result = self.model.load_state_dict(state_dict, strict=False)
        missing, unexpected = (load_result.missing_keys, load_result.unexpected_keys)
        if missing:
            saver.warning(f'missing keys when loading {self.model_path}: {missing}')
        if unexpected:
            saver.warning(f'unexpected keys when loading {self.model_path}: {unexpected}')
        saver.info(f'loaded {self.model_path}')
        encoder_loaded = False
        try:
            self.encoder = load(self.encoder_path)
            from sklearn.utils.validation import check_is_fitted
            try:
                check_is_fitted(self.encoder['enc_ntype'])
                encoder_loaded = True
                saver.log_info(f'Loaded encoders from {self.encoder_path} (fitted)')
            except Exception as fit_check_error:
                saver.warning(f'Encoders loaded but not fitted: {fit_check_error}')
                encoder_loaded = False
        except Exception as e:
            saver.warning(f'Failed to load encoders from {self.encoder_path}: {e}')
            encoder_loaded = False
        if not encoder_loaded:
            if kernel_graph is None:
                raise RuntimeError(f'Cannot rebuild encoders: kernel_graph is None. Please ensure kernel graph is loaded before initializing GNNModel.')
            saver.warning('Rebuilding encoders from current kernel graph (not fitted or load failed).')
            self.encoder = _rebuild_encoders_from_graph(kernel_graph)
            try:
                save(self.encoder, self.encoder_path)
                saver.log_info(f'Rebuilt encoders saved to {self.encoder_path}.klepto')
            except Exception as e2:
                saver.warning(f'Failed to save rebuilt encoders: {e2}')

    def encode_node(self, g, point: DesignPoint):
        X_ntype = []
        X_ptype = []
        X_numeric = []
        X_itype = []
        X_ftype = []
        X_btype = []
        for node, ndata in g.nodes(data=True):
            numeric = 0
            if 'full_text' in ndata and 'pragma' in ndata['full_text']:
                p_text = ndata['full_text'].rstrip()
                assert p_text[0:8] == '#pragma '
                p_text_type = p_text[8:].upper()
                if _check_any_in_str(NON_OPT_PRAGMAS, p_text_type):
                    p_text_type = 'None'
                else:
                    if _check_any_in_str(WITH_VAR_PRAGMAS, p_text_type):
                        t_li = p_text_type.split(' ')
                        for i in range(len(t_li)):
                            if 'VARIABLE=' in t_li[i]:
                                t_li[i] = 'VARIABLE=<>'
                            elif 'DEPTH=' in t_li[i]:
                                t_li[i] = 'DEPTH=<>'
                            elif 'DIM=' in t_li[i]:
                                numeric = int(t_li[i][4:])
                                t_li[i] = 'DIM=<>'
                            elif 'LATENCY=' in t_li[i]:
                                numeric = int(t_li[i][8:])
                                t_li[i] = 'LATENCY=<>'
                        p_text_type = ' '.join(t_li)
                    if point is not None:
                        t_li = p_text_type.split(' ')
                        for i in range(len(t_li)):
                            if 'AUTO{' in t_li[i]:
                                auto_what = _in_between(t_li[i], '{', '}')
                                numeric = point[auto_what]
                                if type(numeric) is not int:
                                    t_li[i] = numeric
                                    numeric = 0
                                else:
                                    t_li[i] = 'AUTO{<>}'
                                break
                        p_text_type = ' '.join(t_li)
                    else:
                        assert 'AUTO' not in p_text_type
                ptype = p_text_type
            else:
                ptype = 'None'
            X_ntype.append([ndata['type']])
            X_ptype.append([ptype])
            X_numeric.append([numeric])
            X_itype.append([ndata['text']])
            X_ftype.append([ndata['function']])
            X_btype.append([ndata['block']])
        node_dict = {'X_ntype': X_ntype, 'X_ptype': X_ptype, 'X_numeric': X_numeric, 'X_itype': X_itype, 'X_ftype': X_ftype, 'X_btype': X_btype}
        enc_ntype = self.encoder['enc_ntype']
        enc_ptype = self.encoder['enc_ptype']
        enc_itype = self.encoder['enc_itype']
        enc_ftype = self.encoder['enc_ftype']
        enc_btype = self.encoder['enc_btype']
        return _encode_X_torch(node_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)

    def encode_edge(self, g):
        edge_dict = _encode_edge_dict(g)
        enc_ptype_edge = self.encoder['enc_ptype_edge']
        enc_ftype_edge = self.encoder['enc_ftype_edge']
        return _encode_edge_torch(edge_dict, enc_ftype_edge, enc_ptype_edge)

    def perf_as_quality(self, new_result: Result) -> float:
        return 1.0 / new_result.perf

    def quantify_util(self, result: Result) -> float:
        utils = [5 * ceil(max(0.0, u) * 100 / 5) / 100 for k, u in result.res_util.items() if k.startswith('util')]
        res = sum([2.0 ** u for u in utils])
        return res

    def eff_as_quality(self, new_result: Result) -> float:
        util_vals = [max(0.0, float(u)) for k, u in new_result.res_util.items() if isinstance(k, str) and k.startswith('util')]
        util_sum = float(sum(util_vals)) if util_vals else 0.0
        eps = getattr(FLAGS, 'epsilon', 1e-09)
        area_scalar = log10(1.0 + util_sum)
        return 1.0 / (max(float(new_result.perf), eps) * max(float(area_scalar), eps))

    def test(self, loader, code_loader, config, mode: str='regression', kernel_name: Optional[str]=None, ref_perf_map: Optional[Dict[str, float]]=None):
        self.model.eval()
        i = 0
        results: List[Result] = []
        target_list = FLAGS.target
        if not isinstance(FLAGS.target, list):
            target_list = [FLAGS.target]
        inx = 0
        for data in loader:
            torch.cuda.empty_cache()
            data = data.to(FLAGS.device)
            code_emb = code_loader[inx]
            inx += 1
            fusion_w = None
            if getattr(FLAGS, 'use_causal', False) and getattr(FLAGS, 'causal_main_pred_mode', 'mlp') == 'fusion':
                fusion_w = float(getattr(FLAGS, 'causal_fusion_w_max', 0.2))
            out_dict, loss, loss_dict = self.model(data, code_emb, fusion_w=fusion_w)
            if mode == 'regression':
                for i in range(len(out_dict['perf'])):
                    curr_result = Result()
                    curr_result.point = data[i].point
                    for target_name in target_list:
                        out = out_dict[target_name]
                        out_value = out[i].item()
                        if target_name == 'perf':
                            eps = getattr(FLAGS, 'epsilon', 1e-09)
                            normalizer = float(getattr(FLAGS, 'normalizer', 1.0))
                            denom = pow(100.0, float(out_value)) - 1.0
                            denom = max(denom, eps)
                            curr_result.perf = max(normalizer / denom, eps)
                            curr_result.actual_perf = float(curr_result.perf)
                        elif target_name in curr_result.res_util.keys():
                            v = float(out_value)
                            if v != v:
                                v = 0.0
                            if v <= 0.0:
                                curr_result.res_util[target_name] = 0.0
                            else:
                                curr_result.res_util[target_name] = pow(100.0, v) - 1.0
                        else:
                            raise NotImplementedError()
                    quality = self.perf_as_quality(curr_result)
                    curr_result.area = self.eff_as_quality(curr_result)
                    curr_result.quality = (quality + curr_result.area) / 2
                    max_utils = config['max-util']
                    results.append(curr_result)
            elif mode == 'class':
                _, pred = torch.max(out_dict['perf'], 1)
                labels = _get_y_with_target(data, 'perf')
                return pred == labels
            else:
                raise NotImplementedError()
        try:
            if getattr(FLAGS, 'use_causal', False) and hasattr(self.model, '_last_alpha_matrix'):
                alpha = getattr(self.model, '_last_alpha_matrix', None)
                pragma_ids_batch = getattr(self.model, '_last_pragma_ids_batch', None)
                pragma_mask_batch = getattr(self.model, '_last_pragma_mask_batch', None)
                if alpha is not None and alpha.numel() > 0:
                    score = alpha.abs().mean(dim=(0, 2))
                    valid_mask = None
                    if isinstance(pragma_mask_batch, list) and len(pragma_mask_batch) > 0:
                        try:
                            mask_stack = []
                            for m in pragma_mask_batch:
                                if torch.is_tensor(m):
                                    mask_stack.append(m.to(score.device).bool())
                            if len(mask_stack) > 0:
                                valid_mask = torch.stack(mask_stack, dim=0).any(dim=0)
                                score = score.masked_fill(~valid_mask, float('-inf'))
                        except Exception:
                            pass
                    k = min(5, score.shape[0])
                    topv, topi = torch.topk(score, k=k)
                    self.log.info('[Causal] Top pragma impacts (mean|alpha|):')
                    pragma_id_map = {}
                    if isinstance(pragma_ids_batch, list):
                        for sample_ids in pragma_ids_batch:
                            if isinstance(sample_ids, list):
                                for idx, pid in enumerate(sample_ids):
                                    if isinstance(pid, str) and pid:
                                        pragma_id_map[idx] = pid
                                if len(pragma_id_map) > 0:
                                    break
                    for vv, ii in zip(topv.tolist(), topi.tolist()):
                        pid = pragma_id_map.get(ii, str(ii))
                        self.log.info(f'  - {pid}: {vv:.4f}')
                    if not getattr(self, '_printed_causal_debug_flag_once', False):
                        self.log.info(f"[Causal Debug] FLAGS.debug_causal_alpha={getattr(FLAGS, 'debug_causal_alpha', False)}")
                        setattr(self, '_printed_causal_debug_flag_once', True)
                    if getattr(FLAGS, 'debug_causal_alpha', False):
                        try:
                            a = alpha.detach()
                            if valid_mask is not None and torch.is_tensor(valid_mask):
                                vm = valid_mask.to(a.device).bool()
                                a = a[:, vm, :]
                            B, P, T = a.shape
                            eps = 1e-12
                            ent = -(a * torch.log(a + eps)).sum(dim=1)
                            ent_mean = float(ent.mean().item())
                            ent_norm = float((ent / max(float(torch.log(torch.tensor(float(P), device=a.device)).item()), eps)).mean().item()) if P > 1 else 0.0
                            top2 = torch.topk(a, k=min(2, P), dim=1).values
                            top1_mean = float(top2[:, 0, :].mean().item())
                            top2_mean = float(top2[:, 1, :].mean().item()) if P >= 2 else 0.0
                            gap_mean = float((top2[:, 0, :] - (top2[:, 1, :] if P >= 2 else 0.0)).mean().item())
                            self.log.info(f'[Causal Debug] alpha shape={tuple(alpha.shape)}, valid_pragmas={P}, entropy_mean={ent_mean:.4f}, entropy_norm={ent_norm:.4f}, top1_mean={top1_mean:.4f}, top2_mean={top2_mean:.4f}, top1-top2={gap_mean:.4f}')
                        except Exception as _e:
                            self.log.info(f'[Causal Debug] failed to compute alpha diagnostics: {_e}')
        except Exception:
            pass
        return results

class Explorer:

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=False):
        self.run_dse = run_dse
        self.log = saver
        self.kernel_name = kernel_name
        self.config_path = join(path_kernel, f'{kernel_name}_ds_config.json')
        self.config = self.load_config()
        self.timeout = 60 * 60
        self.ds, self.ds_size = compile_design_space(self.config['design-space']['definition'], None, self.log)
        self.batch_size = 1
        self.num_top_designs = 3
        self.key_perf_dict = OrderedDict()
        self.best_results_dict = {}
        self.best_result: Result = Result()
        self.explored_point = 0
        self.ordered_pids = self.topo_sort_param_ids(self.ds)
        self._ref_perf_map = _load_ref_perf_map()
        gexf_file = sorted([f for f in glob(path_graph + '/*') if f.endswith('.gexf') and kernel_name in f])
        assert len(gexf_file) >= 1
        self.graph_path = join(path_graph, gexf_file[0])
        self.graph = nx.read_gexf(self.graph_path)
        deg_hist = None
        try:
            deg_list = [int(d) for _, d in self.graph.degree()]
            max_deg = max(deg_list) if len(deg_list) > 0 else 0
            hist = [0] * (max_deg + 1)
            for d in deg_list:
                if d >= 0:
                    hist[d] += 1
            deg_hist = torch.tensor(hist, dtype=torch.long)
        except Exception:
            deg_hist = torch.ones(1, dtype=torch.long)
        inferred_edge_dim = 7
        self.GNNmodel = GNNModel(SAVE_DIR, self.log, multi_target=True, task='regression', num_layers=FLAGS.num_layers, D=FLAGS.D, deg=deg_hist, edge_dim=inferred_edge_dim, kernel_graph=self.graph)
        self.best_save_results = {}
        self._norm_perf_min = float('inf')
        self._norm_perf_max = float('-inf')
        self._norm_util_min = float('inf')
        self._norm_util_max = float('-inf')
        self._fixed_norm_bounds = _lookup_ref_norm_bounds(_load_ref_norm_bounds(), self.kernel_name)
        if isinstance(self._fixed_norm_bounds, dict):
            try:
                b = self._fixed_norm_bounds
                self.log.info('[Norm] Using fixed bounds for %s: perf=[%.6g, %.6g], util_sum=[%.6g, %.6g]' % (self.kernel_name, float(b['perf_min']), float(b['perf_max']), float(b['util_min']), float(b['util_max'])))
            except Exception:
                pass
        else:
            self.log.info('[Norm] Fixed bounds not found for %s; fallback to running min-max.' % self.kernel_name)
        if FLAGS.separate_perf:
            perf_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP']
            self.GNNmodel_perf = GNNModel(SAVE_DIR, self.log, multi_target=True, task='regression_perf', num_layers=8, D=64, target=perf_target, deg=deg_hist, edge_dim=inferred_edge_dim, kernel_graph=self.graph)
        self.prune_invalid = prune_invalid
        codebert_path = _find_local_codebert_path()
        local_only = os.path.isdir(codebert_path)
        self.codebert = AutoModelForMaskedLM.from_pretrained(codebert_path, local_files_only=local_only).to(FLAGS.device)
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_path, local_files_only=local_only)
        if self.prune_invalid:
            self.GNNmodel_valid = GNNModel(SAVE_DIR_CLASS, self.log, multi_target=False, task='class', num_layers=FLAGS.num_layers, D=FLAGS.D, deg=deg_hist, edge_dim=inferred_edge_dim, kernel_graph=self.graph)
        if self.ds_size <= 500:
            self.result_number = 10
            self.stop_cond = ceil(0.5 * self.ds_size)
        elif 500 < self.ds_size <= 10000:
            self.result_number = 10
            self.stop_cond = ceil(0.3 * self.ds_size)
        elif 10000 < self.ds_size <= 100000:
            self.result_number = 10
            self.stop_cond = ceil(0.05 * self.ds_size)
        elif 100000 < self.ds_size <= 1000000.0:
            self.result_number = 10
            self.stop_cond = ceil(0.005 * self.ds_size)
        elif 1000000.0 < self.ds_size <= 10000000.0:
            self.result_number = 10
            self.stop_cond = ceil(0.0005 * self.ds_size)
        elif 10000000.0 < self.ds_size <= 100000000.0:
            self.result_number = 10
            self.stop_cond = ceil(5e-05 * self.ds_size)
        elif 100000000.0 < self.ds_size <= 1000000000.0:
            self.result_number = 10
            self.stop_cond = ceil(5e-06 * self.ds_size)
        elif 1000000000.0 < self.ds_size <= 10000000000.0:
            self.result_number = 10
            self.stop_cond = ceil(5e-07 * self.ds_size)
        elif 10000000000.0 < self.ds_size <= 100000000000.0:
            self.result_number = 10
            self.stop_cond = ceil(5e-07 * self.ds_size)
        elif 100000000000.0 < self.ds_size <= 1000000000000.0:
            self.result_number = 10
            self.stop_cond = ceil(5e-08 * self.ds_size)
        elif 1000000000000.0 < self.ds_size <= 10000000000000.0:
            self.result_number = 10
            self.stop_cond = ceil(5e-09 * self.ds_size)
        else:
            self.result_number = 10
            self.stop_cond = ceil(5e-10 * self.ds_size)

    def topo_sort_param_ids(self, space: DesignSpace) -> List[str]:
        return topo_sort_param_ids(space)

    def load_config(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.config_path):
                self.log.error(('Config JSON file not found: %s', self.config_path))
                raise RuntimeError()
            self.log.info('Loading configurations')
            with open(self.config_path, 'r', errors='replace') as filep:
                try:
                    user_config = json.load(filep)
                except ValueError as err:
                    self.log.error(('Failed to load config: %s', str(err)))
                    raise RuntimeError()
            config = build_config(user_config, self.log)
            if config is None:
                self.log.error(('Config %s is invalid', self.config_path))
                raise RuntimeError()
        except RuntimeError:
            sys.exit(1)
        return config

    def apply_design_point(self, g, point: DesignPoint, mode='regression') -> Data:
        X = self.GNNmodel.encode_node(g, point)
        edge_attr = self.GNNmodel.encode_edge(g)
        edge_index = create_edge_index(g)
        d_node = dict()
        resources = ['BRAM', 'DSP', 'LUT', 'FF']
        keys = ['perf', 'actual_perf', 'quality']
        for r in resources:
            keys.append('util-' + r)
            keys.append('total-' + r)
        for key in keys:
            d_node[key] = 0
        if mode == 'class':
            d_node['perf'] = 1
        if 'regression' in mode:
            data = Data(x=X, edge_index=edge_index, perf=d_node['perf'], actual_perf=d_node['actual_perf'], quality=d_node['quality'], util_BRAM=d_node['util-BRAM'], util_DSP=d_node['util-DSP'], util_LUT=d_node['util-LUT'], util_FF=d_node['util-FF'], total_BRAM=d_node['total-BRAM'], total_DSP=d_node['total-DSP'], total_LUT=d_node['total-LUT'], total_FF=d_node['total-FF'], point=point, edge_attr=edge_attr)
        elif mode == 'class':
            data = Data(x=X, edge_index=edge_index, perf=d_node['perf'], edge_attr=edge_attr, kernel=self.kernel_name)
        else:
            raise NotImplementedError()
        return data

    def update_best(self, result: Result):
        if not isinstance(result, Result) or not result.valid:
            return False
        update_flag = False
        point_key = gen_key_from_design_point(result.point)
        if point_key in self.key_perf_dict:
            return False
        if not self.key_perf_dict:
            is_better = True
        else:
            current_best_q = max(self.key_perf_dict.values())
            is_better = result.quality > current_best_q
        self.key_perf_dict[point_key] = result.quality
        self.best_results_dict[result.quality, point_key] = result
        if len(self.key_perf_dict) > self.num_top_designs:
            worst_key = min(self.key_perf_dict, key=lambda k: self.key_perf_dict[k])
            worst_q = self.key_perf_dict[worst_key]
            self.key_perf_dict.pop(worst_key, None)
            self.best_results_dict.pop((worst_q, worst_key), None)
        if is_better:
            self.best_result = result
            self.log.info('Found a better result at {}: Quality {:.6f}, Perf {:.6f}'.format(self.explored_point, result.quality, result.perf))
            try:
                attrs = vars(result)
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
            except Exception:
                pass
            update_flag = True
        return update_flag

    def gen_options(self, point: DesignPoint, pid: str, default=False) -> List[Union[int, str]]:
        if default:
            dep_values = {dep: point[dep].default for dep in self.ds[pid].deps}
        else:
            dep_values = {dep: point[dep] for dep in self.ds[pid].deps}
        dep_values = {dep: point[dep] for dep in self.ds[pid].deps}
        options = eval(self.ds[pid].option_expr, dep_values)
        if options is None:
            self.log.error(f'Failed to evaluate {self.ds[pid].option_expr} with dep {str(dep_values)}')
            print('Error: failed to manipulate design points')
            sys.exit(1)
        return options

    def get_order(self, point: DesignPoint, pid: str) -> int:
        if not self.ds[pid].order:
            return 0
        order = eval(self.ds[pid].order['expr'], {self.ds[pid].order['var']: point[pid]})
        if order is None or not isinstance(order, int):
            self.log.warning(f'Failed to evaluate the order of {pid} with value {str(point[pid])}: {str(order)}')
            return 0
        return order

    def update_child(self, point: DesignPoint, pid: str) -> None:
        pendings = [child for child in self.ds[pid].child if self.validate_value(point, child)]
        for child in pendings:
            self.update_child(point, child)

    def validate_point(self, point: DesignPoint) -> bool:
        changed_any = False
        max_passes = max(1, len(getattr(self, 'ordered_pids', [])) + 1)
        for _ in range(max_passes):
            changed_this_pass = False
            for pid in getattr(self, 'ordered_pids', []):
                if pid not in point:
                    continue
                try:
                    if self.validate_value(point, pid):
                        changed_any = True
                        changed_this_pass = True
                        self.update_child(point, pid)
                except Exception:
                    continue
            for pid in list(point.keys()):
                if pid in getattr(self, 'ordered_pids', []):
                    continue
                try:
                    if self.validate_value(point, pid):
                        changed_any = True
                        changed_this_pass = True
                except Exception:
                    continue
            if not changed_this_pass:
                break
        return changed_any

    def validate_value(self, point: DesignPoint, pid: str) -> bool:
        options = self.gen_options(point, pid)
        value = point[pid]
        if not options:
            self.log.warning(f'No valid options for {pid} with point {str(point)}')
            point[pid] = self.ds[pid].default
            return False
        if isinstance(value, int):
            cand = min(options, key=lambda x: abs(int(x) - int(value)))
            if cand != value:
                point[pid] = cand
                return True
        if value not in options:
            point[pid] = self.ds[pid].default
            return True
        return False

    def move_by(self, point: DesignPoint, pid: str, step: int=1) -> int:
        try:
            options = self.gen_options(point, pid)
            idx = options.index(point[pid])
        except (AttributeError, ValueError) as err:
            self.log.error(f'Fail to identify the index of value {point[pid]} of parameter {pid} at design point {str(point)}: {str(err)}')
            print('Error: failed to manipulate design points')
            sys.exit(1)
        target = idx + step
        if target >= len(options):
            target = len(options) - 1
        elif target < 0:
            target = 0
        if target != idx:
            point[pid] = options[target]
            self.update_child(point, pid)
        return target - idx

    def traverse(self, point: DesignPoint, idx: int) -> Generator[DesignPoint, None, None]:
        if idx == len(self.ordered_pids):
            yield point
        else:
            yield from self.traverse(point, idx + 1)
            new_point = self.clone_point(point)
            while self.move_by(new_point, self.ordered_pids[idx]) == 1:
                yield from self.traverse(new_point, idx + 1)
                new_point = self.clone_point(new_point)

    @staticmethod
    def clone_point(point: DesignPoint) -> DesignPoint:
        return dict(point)

    def get_results(self, population: List[DesignPoint]) -> List[Result]:

        def _canonicalize_point(p: DesignPoint) -> DesignPoint:
            if p is None:
                return p
            out: Dict[str, Any] = {}
            for k, v in dict(p).items():
                try:
                    if torch.is_tensor(v):
                        if v.numel() == 1:
                            v = v.item()
                        else:
                            v = v.detach().cpu().tolist()
                    try:
                        import numpy as _np
                        if isinstance(v, _np.generic):
                            v = v.item()
                    except Exception:
                        pass
                except Exception:
                    pass
                out[k] = v
            return out
        data_list = []
        code_list = []
        for point in population:
            point_for_eval = point
            if getattr(FLAGS, 'project_invalid_to_valid', False):
                try:
                    full_point = get_default_point(self.ds)
                    if isinstance(point, dict):
                        full_point.update(point)
                    else:
                        full_point.update(dict(point))
                    if isinstance(full_point, dict):
                        self.validate_point(full_point)
                    point_for_eval = full_point
                except Exception:
                    point_for_eval = point
            point_for_eval = _canonicalize_point(point_for_eval)
            data_list.append(self.apply_design_point(self.graph, point_for_eval))
            code_list.append(self.gen_code_embedding(point_for_eval, self.kernel_name))
        test_loader = DataLoader(data_list, batch_size=self.batch_size)
        code_loader = code_list
        results = self.GNNmodel.test(test_loader, code_loader, self.config['evaluate'], mode='regression', kernel_name=self.kernel_name, ref_perf_map=self._ref_perf_map)
        eps = getattr(FLAGS, 'epsilon', 1e-09)
        for r in results:
            if not isinstance(r, Result):
                continue
            try:
                perf = float(r.perf)
                if perf != perf:
                    continue
                util_sum = 0.0
                for k, u in getattr(r, 'res_util', {}).items():
                    if isinstance(k, str) and k.startswith('util'):
                        uu = float(u)
                        if uu == uu:
                            util_sum += max(0.0, uu)
                if isinstance(self._fixed_norm_bounds, dict):
                    perf_min = float(self._fixed_norm_bounds.get('perf_min', 0.0))
                    perf_max = float(self._fixed_norm_bounds.get('perf_max', 0.0))
                    util_min = float(self._fixed_norm_bounds.get('util_min', 0.0))
                    util_max = float(self._fixed_norm_bounds.get('util_max', 0.0))
                else:
                    self._norm_perf_min = min(self._norm_perf_min, perf)
                    self._norm_perf_max = max(self._norm_perf_max, perf)
                    self._norm_util_min = min(self._norm_util_min, util_sum)
                    self._norm_util_max = max(self._norm_util_max, util_sum)
                    perf_min = self._norm_perf_min
                    perf_max = self._norm_perf_max
                    util_min = self._norm_util_min
                    util_max = self._norm_util_max
                if perf_max <= perf_min + eps:
                    perf_n = 0.0
                else:
                    perf_n = (perf_max - perf) / max(perf_max - perf_min, eps)
                if util_max <= util_min + eps:
                    util_n = 0.0
                else:
                    util_n = (util_sum - util_min) / max(util_max - util_min, eps)
                perf_n = float(max(0.0, min(1.0, perf_n)))
                util_n = float(max(0.0, min(1.0, util_n)))
                r.perf_n = perf_n
                r.util_n = util_n
                r.util_sum = float(util_sum)
                r.quality = 0.5 * perf_n + 0.5 * (1.0 - util_n)
            except Exception:
                continue
        return results

    def get_config_dafault_options(self):
        defaults_dict = {key: self.ds[key].default for key in self.ordered_pids}
        config_options = {}
        config_cond = {}
        for key in self.ordered_pids:
            if self.ds[key].deps:
                value = self.ds[key].option_expr.split('if')
                config_options[key] = eval(value[0] + ']')
                config_cond[key] = value[1][:-1]
            else:
                config_options[key] = eval(self.ds[key].option_expr)
                config_cond[key] = ''
        for i, j in config_options.items():
            if '' in j:
                j.remove('')
        return (defaults_dict, config_options, config_cond)

    def extract_causal_guidance(self, top_k: int=5) -> Optional[str]:
        try:
            if not getattr(FLAGS, 'use_causal', False):
                return None
            model = self.GNNmodel.model
            if not hasattr(model, '_last_alpha_matrix'):
                return None
            alpha = getattr(model, '_last_alpha_matrix', None)
            pragma_ids_batch = getattr(model, '_last_pragma_ids_batch', None)
            if alpha is None or alpha.numel() == 0:
                return None
            target_names = getattr(model, 'target_list', None)
            if not isinstance(target_names, list) or len(target_names) == 0:
                target_names = [f'target_{i}' for i in range(alpha.shape[2])]
            perf_like = {'perf', 'actual_perf', 'latency'}
            perf_idx = None
            util_indices: List[int] = []
            for ti, tn in enumerate(target_names):
                if not isinstance(tn, str):
                    continue
                if tn in perf_like and perf_idx is None:
                    perf_idx = ti
                if tn.startswith('util') or 'util-' in tn:
                    util_indices.append(ti)
            abs_alpha = alpha.abs()
            if perf_idx is not None and perf_idx < abs_alpha.shape[2]:
                perf_score = abs_alpha[:, :, perf_idx].mean(dim=0)
            else:
                perf_score = abs_alpha.mean(dim=(0, 2))
            if util_indices:
                util_idx_t = torch.tensor(util_indices, device=abs_alpha.device, dtype=torch.long)
                util_score = abs_alpha.index_select(dim=2, index=util_idx_t).mean(dim=(0, 2))
            else:
                util_score = abs_alpha.mean(dim=(0, 2))
            score = 0.5 * perf_score + 0.5 * util_score
            ref_row: Optional[List[Optional[str]]] = None
            if isinstance(pragma_ids_batch, list):
                for batch_pragma_ids in pragma_ids_batch:
                    if batch_pragma_ids is not None and isinstance(batch_pragma_ids, list):
                        ref_row = list(batch_pragma_ids)
                        break
            pragma_mask_batch = getattr(model, '_last_pragma_mask_batch', None)
            valid_mask = None
            if isinstance(pragma_mask_batch, list) and len(pragma_mask_batch) > 0:
                try:
                    mask_stack = []
                    for m in pragma_mask_batch:
                        if torch.is_tensor(m):
                            mask_stack.append(m.to(score.device).bool())
                    if len(mask_stack) > 0:
                        valid_mask = torch.stack(mask_stack, dim=0).any(dim=0)
                except Exception:
                    valid_mask = None

            def _masked_for_topk(t: torch.Tensor) -> torch.Tensor:
                if valid_mask is None or valid_mask.numel() != t.numel():
                    return t
                out = t.clone()
                return out.masked_fill(~valid_mask, float('-inf'))

            def _topk(v: torch.Tensor) -> Optional[tuple]:
                if v is None or v.numel() == 0:
                    return None
                kk = min(top_k, int(v.shape[0]))
                if kk <= 0:
                    return None
                tv, ti = torch.topk(v, k=kk)
                return (tv, ti)
            score_rank = _masked_for_topk(score)
            if not torch.isfinite(score_rank).any():
                return None
            top_quality = _topk(score_rank)
            top_perf = _topk(_masked_for_topk(perf_score))
            top_util = _topk(_masked_for_topk(util_score))
            if top_quality is None:
                return None
            topv, topi = top_quality

            def _pragma_label(ii: int) -> str:
                if ref_row is not None and 0 <= ii < len(ref_row):
                    pid = ref_row[ii]
                    if isinstance(pid, str) and len(pid) > 0:
                        return pid
                return f'pragma_{ii}'

            def _fmt_list(header: str, tv_ti: Optional[tuple]) -> str:
                if tv_ti is None:
                    return ''
                tv, ti = tv_ti
                lines = [header]
                for vv, ii in zip(tv.tolist(), ti.tolist()):
                    lines.append(f'  - {_pragma_label(int(ii))}: importance={vv:.4f}')
                return '\n'.join(lines)
            try:
                s = score
                if valid_mask is not None and valid_mask.numel() == s.numel():
                    sv = s[valid_mask]
                else:
                    sv = s
                if sv.numel() == 0:
                    self._last_causal_importance = {}
                else:
                    s_min = float(sv.min().item())
                    s_max = float(sv.max().item())
                    denom = max(s_max - s_min, 1e-09)
                    s_norm = torch.zeros_like(s)
                    if valid_mask is not None and valid_mask.numel() == s.numel():
                        s_norm[valid_mask] = (s[valid_mask] - s_min) / denom
                    else:
                        s_norm = (s - s_min) / denom
                    importance_map: Dict[str, float] = {}
                    for idx in range(int(s_norm.shape[0])):
                        if valid_mask is not None and (idx >= int(valid_mask.shape[0]) or not bool(valid_mask[idx].item())):
                            continue
                        pid = ref_row[idx] if ref_row is not None and idx < len(ref_row) else None
                        if pid is None or not isinstance(pid, str) or len(pid) <= 1:
                            continue
                        importance_map[pid] = float(s_norm[idx].item())
                    self._last_causal_importance = importance_map
            except Exception:
                self._last_causal_importance = {}
            parts = []
            parts.append(_fmt_list('Quality-aligned top pragmas (quality = 0.5*perf_n + 0.5*(1-util_n)):', top_quality))
            if top_perf is not None:
                parts.append(_fmt_list('Top pragmas for perf-related target:', top_perf))
            if top_util is not None:
                parts.append(_fmt_list('Top pragmas for util-* targets (lower util is better):', top_util))
            guidance_text = 'Causal guidance (from alpha_matrix):\n' + '\n\n'.join([p for p in parts if p])
            guidance_text += "\n\nPlease prioritize modifying the 'Quality-aligned top pragmas' first."
            guidance_text += '\nWhen adjusting pragmas, prefer changes that improve perf while not increasing util; or reduce util with minimal perf regression.'
            return guidance_text
        except Exception as e:
            self.log.debug(f'Failed to extract causal guidance: {e}')
            return None

    def load_llm_process_ec(self, fitness, current_population, pragmas_possible_value, result_number, temperature, causal_guidance=None):
        tokens = 0
        llm = ChatOpenAI(model=FLAGS.llm_model, temperature=temperature, openai_api_key=FLAGS.api_key, openai_api_base=FLAGS.api_base, request_timeout=2000, streaming=False, callbacks=[StreamingStdOutCallbackHandler()])
        start_time = datetime.now()
        secs = (datetime.now() - start_time).total_seconds()
        if secs >= 60:
            secs = 60
            tokens = 0
        res, tokens = llm_process_ec(llm=llm, tokens=tokens, secs=secs, fitness=fitness, current_population=current_population, pragmas_possible_value=pragmas_possible_value, result_number=result_number, causal_guidance=causal_guidance)
        return (res, tokens)

    def transfer_res_to_config(self, res, logger, co):
        logger.info('starting to transfer res to config')

        def _extract_list_candidates(text: str) -> List[str]:
            if not text:
                return []
            candidates: List[str] = []
            for m in re.finditer('<start>\\s*(\\[[\\s\\S]*?\\])\\s*<end>', text, flags=re.IGNORECASE):
                candidates.append(m.group(1))
            if candidates:
                return candidates
            for line in text.splitlines():
                if '[' not in line or ']' not in line:
                    continue
                start = line.find('[')
                end = line.find(']', start)
                if end > start:
                    candidates.append(line[start:end + 1])
            return candidates

        def _safe_parse_flat_list(list_str: str) -> Optional[List[Any]]:
            if not list_str:
                return None
            s = list_str.strip()
            s = s.replace('“', "'").replace('”', "'").replace('’', "'").replace('`', "'")
            s = re.sub("(\\d)'\\s*(?=[,\\]])", '\\1', s)
            s = re.sub("(?<=[,\\[])\\s*'(\\d+)'\\s*(?=[,\\]])", ' \\1', s)
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return val
            except Exception:
                pass
            tokens: List[Any] = []
            for m in re.finditer('\'([^\']*)\'|"([^"]*)"|(-?\\d+(?:\\.\\d+)?)', s):
                if m.group(1) is not None:
                    tokens.append(m.group(1))
                elif m.group(2) is not None:
                    tokens.append(m.group(2))
                else:
                    num = m.group(3)
                    try:
                        tokens.append(int(num))
                    except Exception:
                        try:
                            tokens.append(float(num))
                        except Exception:
                            tokens.append(num)
            return tokens or None

        def _coerce_to_option_type(x: Any, options: List[Any]) -> Any:
            if not options:
                return x
            want_int = any((isinstance(o, int) for o in options))
            want_float = any((isinstance(o, float) for o in options))
            if want_int:
                try:
                    return int(x)
                except Exception:
                    try:
                        return int(float(x))
                    except Exception:
                        return x
            if want_float:
                try:
                    return float(x)
                except Exception:
                    return x
            return x if isinstance(x, str) else str(x)
        keys = list(co.keys())
        options_by_key = list(co.values())
        out: List[Dict[str, Any]] = []
        candidates = _extract_list_candidates(res if isinstance(res, str) else str(res))
        if not candidates:
            logger.warning('No list candidates found in LLM output; skip this iteration.')
            return []
        for cand in candidates:
            parsed = _safe_parse_flat_list(cand)
            if not parsed or len(parsed) < len(keys):
                continue
            d: Dict[str, Any] = {}
            for idx, key in enumerate(keys):
                d[key] = _coerce_to_option_type(parsed[idx], options_by_key[idx])
            out.append(d)
        if not out:
            logger.warning('Failed to parse any valid configs from LLM output; skip this iteration.')
        return out

    def generate_all_solutions(self, default_dict, pragmas_possible_values, config_cond):
        pragma_order = list(default_dict.keys())
        value_lists = [pragmas_possible_values[pragma] for pragma in pragma_order]
        all_combinations = itertools.product(*value_lists)
        solutions = []
        for combination in all_combinations:
            config = dict(zip(pragma_order, combination))
            is_valid = True
            temp_dict = config
            for key in temp_dict.keys():
                if config_cond[key] != '':
                    cond = config_cond[key]
                    dep_list = self.ds[key].deps
                    x = temp_dict[key]
                    temp = cond
                    for dep in dep_list:
                        if type(temp_dict[dep]) == int:
                            temp = temp.replace(dep, str(temp_dict[dep]))
                        else:
                            temp = temp.replace(dep, f"'{temp_dict[dep]}'")
                    if eval(temp) == False:
                        is_valid = False
                        break
            if is_valid:
                solutions.append(config)
        return solutions

    def gen_code_embedding(self, point, k):
        if k in MACHSUITE_KERNEL:
            CODE_FILES = join(get_root_path(), 'dse_database', 'programl', 'machsuite', k, k + '.c')
        else:
            CODE_FILES = join(get_root_path(), 'dse_database', 'programl', 'poly', k, k + '.c')
        with open(CODE_FILES, 'r') as file:
            fc = file.read()
        fcc = copy.deepcopy(fc)
        kd_d = point
        for k, v in kd_d.items():
            fcc = fcc.replace('auto' + '{' + k + '}', str(v))
        code_inputs = self.tokenizer(fcc, padding=True, truncation=True, max_length=512, return_tensors='pt').to(FLAGS.device)
        try:
            self.codebert.eval()
        except Exception:
            pass
        with torch.no_grad():
            code_outputs = self.codebert(**code_inputs, output_hidden_states=True)
            features = []
            for i, hidden_states in enumerate(code_outputs.hidden_states):
                features.append(hidden_states[:, 0, :])
            code_features = torch.stack(features).sum(dim=0) / len(features)
        return code_features

    def run(self) -> None:
        raise NotImplementedError()

class _ParetoArchiveExplorerMixin:

    @staticmethod
    def _to_plain_value(v):
        try:
            if torch.is_tensor(v):
                return v.item() if v.numel() == 1 else v.detach().cpu().tolist()
        except Exception:
            pass
        return v

    def _area_from_result(self, r: Result) -> float:
        if not isinstance(getattr(r, 'res_util', None), dict):
            return float('inf')
        vals = []
        for k, u in r.res_util.items():
            if isinstance(k, str) and k.startswith('util'):
                try:
                    vals.append(float(self._to_plain_value(u)))
                except Exception:
                    continue
        if not vals:
            return float('inf')
        return float(sum(vals) / len(vals))

    def _build_perf_area_snapshot(self):
        snap = []
        for r in self.best_results_dict.values():
            try:
                perf = float(self._to_plain_value(getattr(r, 'perf', float('inf'))))
            except Exception:
                perf = float('inf')
            area = self._area_from_result(r)
            snap.append([perf, area])
        pareto_points = []
        for i, p in enumerate(snap):
            pi_perf, pi_area = (p[0], p[1])
            dominated = False
            for j, q in enumerate(snap):
                if i == j:
                    continue
                q_perf, q_area = (q[0], q[1])
                if (q_perf <= pi_perf and q_area <= pi_area) and (q_perf < pi_perf or q_area < pi_area):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append(p)
        return pareto_points

    def _result_perf_area(self, r: Result):
        try:
            perf = float(self._to_plain_value(getattr(r, 'perf', float('inf'))))
        except Exception:
            perf = float('inf')
        area = self._area_from_result(r)
        return (perf, area)

    @staticmethod
    def _dominates_2d(a_perf: float, a_area: float, b_perf: float, b_area: float) -> bool:
        return (a_perf <= b_perf and a_area <= b_area) and (a_perf < b_perf or a_area < b_area)

    def update_best(self, result: Result):
        if not isinstance(result, Result) or not result.valid:
            return False
        point_key = gen_key_from_design_point(result.point)
        cand_perf, cand_area = self._result_perf_area(result)
        if point_key in self.best_results_dict:
            old_r = self.best_results_dict[point_key]
            old_perf, old_area = self._result_perf_area(old_r)
            if self._dominates_2d(old_perf, old_area, cand_perf, cand_area):
                return False
            self.best_results_dict[point_key] = result
        else:
            self.best_results_dict[point_key] = result
        items = list(self.best_results_dict.items())
        keep = {}
        for i, (ki, ri) in enumerate(items):
            pi, ai = self._result_perf_area(ri)
            dominated = False
            for j, (kj, rj) in enumerate(items):
                if i == j:
                    continue
                pj, aj = self._result_perf_area(rj)
                if self._dominates_2d(pj, aj, pi, ai):
                    dominated = True
                    break
            if not dominated:
                keep[ki] = ri
        self.best_results_dict = keep
        if self.best_results_dict:
            self.best_result = min(self.best_results_dict.values(), key=lambda r: self._result_perf_area(r)[0])
        return True

class CausalHybridExplorer(Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=FLAGS.prune_class, point: DesignPoint=None):
        super(CausalHybridExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.defaults_dict, self.config_options, self.config_cond = self.get_config_dafault_options()
        self.log.info('Done init')
        if self.run_dse:
            _dump_run_meta_once()
            self.run()
            self.log.info('Best Results Found:')
            i = 1
            out_dir = join(BEST_RESULT_RUN_DIR, 'CausalHybrid')
            os.makedirs(out_dir, exist_ok=True)
            with open(join(out_dir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))

    def _sample_random_point(self) -> Dict[str, Any]:
        d = {}
        for k, opts in self.config_options.items():
            if not opts:
                continue
            d[k] = opts[randint(0, len(opts) - 1)]
        return d

    def _canonical_simple(self, v: Any) -> Any:
        try:
            if torch.is_tensor(v):
                return v.item() if v.numel() == 1 else tuple(v.detach().cpu().tolist())
        except Exception:
            pass
        if isinstance(v, list):
            return tuple(v)
        return v

    def _point_key(self, p: Dict[str, Any]) -> tuple:
        return tuple(((k, self._canonical_simple(p.get(k))) for k in self.config_options.keys()))

    @staticmethod
    def _to_plain_value(v):
        try:
            if torch.is_tensor(v):
                return v.item() if v.numel() == 1 else v.detach().cpu().tolist()
        except Exception:
            pass
        return v

    def _area_from_result(self, r: Result) -> float:
        if not isinstance(getattr(r, 'res_util', None), dict):
            return float('inf')
        vals = []
        for k, u in r.res_util.items():
            if isinstance(k, str) and k.startswith('util'):
                try:
                    vals.append(float(self._to_plain_value(u)))
                except Exception:
                    continue
        if not vals:
            return float('inf')
        return float(sum(vals) / len(vals))

    def _result_perf_area(self, r: Result):
        try:
            perf = float(self._to_plain_value(getattr(r, 'perf', float('inf'))))
        except Exception:
            perf = float('inf')
        area = self._area_from_result(r)
        return (perf, area)

    @staticmethod
    def _dominates_2d(a_perf: float, a_area: float, b_perf: float, b_area: float) -> bool:
        return (a_perf <= b_perf and a_area <= b_area) and (a_perf < b_perf or a_area < b_area)

    def _build_perf_area_snapshot(self):
        snap = []
        for r in self.best_results_dict.values():
            perf, area = self._result_perf_area(r)
            snap.append([perf, area])
        pareto_points = []
        for i, p in enumerate(snap):
            pi_perf, pi_area = (p[0], p[1])
            dominated = False
            for j, q in enumerate(snap):
                if i == j:
                    continue
                q_perf, q_area = (q[0], q[1])
                if self._dominates_2d(q_perf, q_area, pi_perf, pi_area):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append(p)
        return pareto_points

    def update_best(self, result: Result):
        if not isinstance(result, Result) or not result.valid:
            return False
        point_key = gen_key_from_design_point(result.point)
        cand_perf, cand_area = self._result_perf_area(result)
        if point_key in self.best_results_dict:
            old_r = self.best_results_dict[point_key]
            old_perf, old_area = self._result_perf_area(old_r)
            if self._dominates_2d(old_perf, old_area, cand_perf, cand_area):
                return False
            self.best_results_dict[point_key] = result
        else:
            self.best_results_dict[point_key] = result
        items = list(self.best_results_dict.items())
        keep = {}
        for i, (ki, ri) in enumerate(items):
            pi, ai = self._result_perf_area(ri)
            dominated = False
            for j, (kj, rj) in enumerate(items):
                if i == j:
                    continue
                pj, aj = self._result_perf_area(rj)
                if self._dominates_2d(pj, aj, pi, ai):
                    dominated = True
                    break
            if not dominated:
                keep[ki] = ri
        self.best_results_dict = keep
        if self.best_results_dict:
            self.best_result = min(self.best_results_dict.values(), key=lambda r: self._result_perf_area(r)[0])
        return True

    def _pick_elite_base(self, elite_rows: List[tuple]) -> Dict[str, Any]:
        if len(elite_rows) == 1:
            return dict(elite_rows[0][0])
        if uniform(0, 1) < 0.62:
            return dict(elite_rows[randint(0, len(elite_rows) - 1)][0])
        scores = []
        for _, perf, area in elite_rows:
            p = float(perf) if np.isfinite(perf) else np.inf
            a = float(area) if np.isfinite(area) else np.inf
            scores.append(p + a)
        s_arr = np.asarray(scores, dtype=np.float64)
        finite = np.isfinite(s_arr)
        if not np.any(finite):
            return dict(elite_rows[randint(0, len(elite_rows) - 1)][0])
        smin = float(np.min(s_arr[finite]))
        smax = float(np.max(s_arr[finite]))
        span = smax - smin if smax > smin else 1.0
        w = []
        for i, sc in enumerate(s_arr):
            if not finite[i]:
                w.append(1.0)
                continue
            n = (float(sc) - smin) / span
            w.append(1.0 / (1e-06 + n + 0.08))
        w_arr = np.asarray(w, dtype=np.float64)
        w_arr /= w_arr.sum()
        j = int(np.random.choice(len(elite_rows), p=w_arr))
        return dict(elite_rows[j][0])

    def _build_next_population(self, importance_map: Dict[str, float]) -> List[Dict[str, Any]]:
        elite_rows: List[tuple] = []
        for r in self.best_results_dict.values():
            if isinstance(getattr(r, 'point', None), dict):
                perf, area = self._result_perf_area(r)
                elite_rows.append((dict(r.point), perf, area))
        if not elite_rows:
            elite_rows = [(dict(self.defaults_dict), float('inf'), float('inf'))]
        keys = list(self.config_options.keys())
        ranked = sorted(keys, key=lambda k: float(importance_map.get(k, -1.0)), reverse=True)
        topk = max(1, min(len(keys), int(getattr(FLAGS, 'causal_subspace_topk', 0) or 0)))
        causal_dims = ranked[:topk] if topk > 0 else ranked
        if not causal_dims:
            causal_dims = keys
        imp_vals = [float(importance_map.get(k, -1.0)) for k in causal_dims]
        if imp_vals and max(imp_vals) - min(imp_vals) < 1e-09:
            causal_dims = list(causal_dims)
            shuffle(causal_dims)
        prog = float(self.explored_point) / max(float(self.stop_cond), 1.0)
        prog = min(1.0, max(0.0, prog))
        exploit_ratio = 0.52 + 0.38 * prog
        pop: List[Dict[str, Any]] = []
        seen: Set[tuple] = set()
        max_tries = max(50, self.result_number * 20)
        tries = 0
        while len(pop) < self.result_number and tries < max_tries:
            tries += 1
            if np.random.rand() < exploit_ratio:
                base = self._pick_elite_base(elite_rows)
                cand = dict(base)
                n_mut = 1 if len(causal_dims) < 2 else randint(1, min(3, len(causal_dims)))
                mut_keys = list(np.random.choice(causal_dims, size=n_mut, replace=False))
                for mk in mut_keys:
                    opts = self.config_options.get(mk, [])
                    if not opts:
                        continue
                    cur = cand.get(mk, self.defaults_dict.get(mk))
                    alt = [o for o in opts if o != cur]
                    pool = alt if alt else opts
                    cand[mk] = pool[randint(0, len(pool) - 1)]
            else:
                cand = self._sample_random_point()
            for k in keys:
                if k not in cand:
                    cand[k] = self.defaults_dict.get(k)
            ksig = self._point_key(cand)
            if ksig in seen:
                continue
            seen.add(ksig)
            pop.append(cand)
        while len(pop) < self.result_number:
            pop.append(self._sample_random_point())
        return pop

    def run(self) -> None:
        timer = time.time()
        if getattr(FLAGS, 'use_causal', False):
            old_norm = (self._norm_perf_min, self._norm_perf_max, self._norm_util_min, self._norm_util_max)
            try:
                _ = self.get_results([dict(self.defaults_dict)])
            except Exception as e:
                self.log.warning(f'[CausalHybrid][Warmup] failed: {e}')
            finally:
                self._norm_perf_min, self._norm_perf_max, self._norm_util_min, self._norm_util_max = old_norm
        current_population = [self._sample_random_point() for _ in range(self.result_number)]
        while time.time() - timer < self.timeout and self.explored_point <= self.stop_cond:
            results = self.get_results(current_population)
            for r in results:
                self.explored_point += 1
                self.update_best(r)
                self.best_save_results[self.explored_point] = self._build_perf_area_snapshot()
                if self.explored_point > self.stop_cond:
                    break
            _ = self.extract_causal_guidance(top_k=5)
            importance_map = getattr(self, '_last_causal_importance', {}) if getattr(FLAGS, 'use_causal', False) else {}
            current_population = self._build_next_population(importance_map if isinstance(importance_map, dict) else {})
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('explorer CausalHybrid')
            print('------------------------------------------------------')
        self.log.info(f'Explored {self.explored_point} points')


class ACOExplorer(_ParetoArchiveExplorerMixin, Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=FLAGS.prune_class, point: DesignPoint=None):
        super(ACOExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.defaults_dict, self.config_options, self.config_cond = self.get_config_dafault_options()
        self.batch_size = 1
        self.params = self.config_options
        self.param_names = list(self.config_options.keys())
        self.n_params = len(self.param_names)
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.01
        self.pheromone = {param: np.ones(len(values)) for param, values in self.params.items()}
        self.pareto_front = []
        self.visited_points = set()
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            self.log.info('Best Results Found:')
            i = 1
            out_dir = join(BEST_RESULT_RUN_DIR, 'ACO')
            os.makedirs(out_dir, exist_ok=True)
            with open(join(out_dir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    class Ant:

        def __init__(self, aco):
            self.aco = aco
            self.solution = {}
            self.fitness = None
            self.violation = 0

        def construct_solution(self):
            if self.aco.explored_point % 200 == 0:
                init_solution = {}
                for key, value in self.aco.config_options.items():
                    init_solution[key] = value[randint(0, len(value) - 1)]
                self.solution = init_solution
                return
            for key in self.aco.params.keys():
                prob = self.aco.calculate_probability(key)
                selected_idx = np.random.choice(len(prob), p=prob)
                self.solution[key] = self.aco.params[key][selected_idx]

    def run(self) -> None:
        while self.explored_point <= self.stop_cond:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            ants = [self.Ant(self) for _ in range(self.result_number)]
            for ant in ants:
                max_retry = 20
                valid_solution = False
                for _ in range(max_retry):
                    ant.construct_solution()
                    point_key = gen_key_from_design_point(ant.solution)
                    if point_key not in self.visited_points:
                        self.visited_points.add(point_key)
                        valid_solution = True
                        break
                if not valid_solution:
                    continue
                result = self.get_results([ant.solution])
                if not result:
                    continue
                r = result[0]
                self.explored_point += 1
                self.update_best(r)
                self.best_save_results[self.explored_point] = self._build_perf_area_snapshot()
                ant.fitness = self._result_perf_area(r)
            self.update_pareto_front(ants)
            self.update_pheromone(ants)
        self.log.info(f'Explored {self.explored_point} points')

    def calculate_probability(self, param):
        pheromone = self.pheromone[param]
        heuristic = np.ones(len(pheromone))
        probabilities = pheromone ** self.alpha * heuristic ** self.beta
        probabilities /= np.sum(probabilities)
        return probabilities

    def update_pheromone(self, ants: List['ACOExplorer.Ant']):
        for param in self.param_names:
            self.pheromone[param] *= 1 - self.rho
        for ant in self.pareto_front:
            if ant.fitness is None:
                continue
            perf, area = ant.fitness
            reward = 1.0 / (perf + area + 1e-06)
            for param in self.param_names:
                idx = self.params[param].index(ant.solution[param])
                self.pheromone[param][idx] += reward

    def update_pareto_front(self, ants: List['ACOExplorer.Ant']):
        for ant in ants:
            if ant.fitness is None or ant.violation > 0:
                continue
            is_pareto = True
            to_remove = []
            for front_sol in self.pareto_front:
                if self.is_dominated(front_sol.fitness, ant.fitness):
                    to_remove.append(front_sol)
                elif self.is_dominated(ant.fitness, front_sol.fitness):
                    is_pareto = False
                    break
            for sol in to_remove:
                self.pareto_front.remove(sol)
            if is_pareto:
                self.pareto_front.append(ant)

    def is_dominated(self, sol_a, sol_b) -> bool:
        if sol_a is None or sol_b is None:
            return False
        return sol_b[0] <= sol_a[0] and sol_b[1] <= sol_a[1] and (sol_b[0] < sol_a[0] or sol_b[1] < sol_a[1])

class EAExplorer(_ParetoArchiveExplorerMixin, Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=FLAGS.prune_class, point: DesignPoint=None):
        super(EAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.visited_points = set()
        self.log.info('Done init')
        if self.run_dse:
            _dump_run_meta_once()
            self.run()
            self.log.info('Best Results Found:')
            i = 1
            out_dir = join(BEST_RESULT_RUN_DIR, 'EA')
            os.makedirs(out_dir, exist_ok=True)
            with open(join(out_dir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))

    def _fitness_from_result(self, r: Result):
        perf, area = self._result_perf_area(r)
        return 1.0 / (perf + area + 1e-06)

    def _random_solution(self, config_options):
        sol = {}
        for key, value in config_options.items():
            sol[key] = value[randint(0, len(value) - 1)]
        return sol

    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        population = []
        while len(population) < self.result_number:
            sol = self._random_solution(config_options)
            point_key = gen_key_from_design_point(sol)
            if point_key not in self.visited_points:
                self.visited_points.add(point_key)
                population.append(sol)
        p_cs = 0.1
        p_mt = 0.1
        while self.explored_point <= self.stop_cond:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            results = self.get_results(population)
            valid_pairs = []
            for idx, r in enumerate(results):
                self.explored_point += 1
                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug('Evaluating Design')
                    self.log.debug(', '.join(('%s: %s' % item for item in attrs.items())))
                    self.update_best(r)
                    if r.valid:
                        fit = self._fitness_from_result(r)
                        valid_pairs.append((population[idx], fit))
                self.best_save_results[self.explored_point] = self._build_perf_area_snapshot()
            if not valid_pairs:
                population = [self._random_solution(config_options) for _ in range(self.result_number)]
                continue
            selected = []
            while len(selected) < self.result_number:
                contestants = random.sample(valid_pairs, k=min(3, len(valid_pairs)))
                winner = max(contestants, key=lambda x: x[1])[0]
                selected.append(winner.copy())
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 >= len(selected):
                    offspring.append(selected[i].copy())
                    break
                p1, p2 = (selected[i], selected[i + 1])
                c1, c2 = ({}, {})
                for para in config_options.keys():
                    if random.random() < p_cs:
                        c1[para] = p2[para]
                        c2[para] = p1[para]
                    else:
                        c1[para] = p1[para]
                        c2[para] = p2[para]
                offspring.append(c1)
                offspring.append(c2)
            next_population = []
            max_retry = 20
            for child in offspring:
                for _ in range(max_retry):
                    temp = child.copy()
                    for j in config_options.keys():
                        if random.random() < p_mt:
                            temp[j] = config_options[j][randint(0, len(config_options[j]) - 1)]
                    point_key = gen_key_from_design_point(temp)
                    if point_key not in self.visited_points:
                        self.visited_points.add(point_key)
                        next_population.append(temp)
                        break
                if len(next_population) >= self.result_number:
                    break
            while len(next_population) < self.result_number:
                temp = self._random_solution(config_options)
                point_key = gen_key_from_design_point(temp)
                if point_key not in self.visited_points:
                    self.visited_points.add(point_key)
                    next_population.append(temp)
            population = next_population
        self.log.info(f'Explored {self.explored_point} points')

class SAExplorer(_ParetoArchiveExplorerMixin, Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=FLAGS.prune_class, point: DesignPoint=None):
        super(SAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.visited_points = set()
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            self.log.info('Best Results Found:')
            i = 1
            out_dir = join(BEST_RESULT_RUN_DIR, 'SA')
            os.makedirs(out_dir, exist_ok=True)
            with open(join(out_dir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    def _random_solution(self, config_options):
        return {k: v[randint(0, len(v) - 1)] for k, v in config_options.items()}

    def _energy(self, r: Result):
        perf, area = self._result_perf_area(r)
        return perf + area

    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        initial_temperature = FLAGS.initial_temperature
        temperature_1 = initial_temperature
        cand_solutions = []
        while len(cand_solutions) < self.result_number:
            sol = self._random_solution(config_options)
            key = gen_key_from_design_point(sol)
            if key not in self.visited_points:
                self.visited_points.add(key)
                cand_solutions.append(sol)
        config_len = {key: len(value) for key, value in config_options.items()}
        base_neighbor_dis = [max(1, ceil(value * FLAGS.neighbor_distance_rate)) for value in config_len.values()]
        while self.explored_point <= self.stop_cond and temperature_1 >= FLAGS.stop_temperature:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            results = self.get_results(cand_solutions)
            current_pairs = []
            for idx, r in enumerate(results):
                self.explored_point += 1
                if isinstance(r, Result):
                    self.update_best(r)
                    if r.valid:
                        current_pairs.append((cand_solutions[idx], r))
                self.best_save_results[self.explored_point] = self._build_perf_area_snapshot()
            next_solutions = []
            for solution, cur_result in current_pairs:
                if len(next_solutions) >= self.result_number:
                    break
                cur_neighbor_dis = [max(1, int(d * temperature_1 / (initial_temperature + 1e-09))) for d in base_neighbor_dis]
                new_solution = {}
                for idx, (key, value) in enumerate(solution.items()):
                    options = config_options[key]
                    cur_idx = options.index(value)
                    dis = randint(-cur_neighbor_dis[idx], cur_neighbor_dis[idx])
                    new_idx = cur_idx + dis
                    new_solution[key] = options[new_idx] if 0 <= new_idx < len(options) else value
                point_key = gen_key_from_design_point(new_solution)
                if point_key in self.visited_points:
                    restart_solution = self._random_solution(config_options)
                    restart_key = gen_key_from_design_point(restart_solution)
                    if restart_key not in self.visited_points:
                        self.visited_points.add(restart_key)
                        next_solutions.append(restart_solution)
                    else:
                        next_solutions.append(solution.copy())
                    continue
                new_result_list = self.get_results([new_solution])
                if not new_result_list:
                    next_solutions.append(solution.copy())
                    continue
                new_result = new_result_list[0]
                self.explored_point += 1
                if isinstance(new_result, Result):
                    self.update_best(new_result)
                self.best_save_results[self.explored_point] = self._build_perf_area_snapshot()
                if isinstance(new_result, Result) and new_result.valid:
                    cur_perf, cur_area = self._result_perf_area(cur_result)
                    new_perf, new_area = self._result_perf_area(new_result)
                    delta_val = new_perf / (cur_perf + 1e-09) + new_area / (cur_area + 1e-09) - 2.0
                    if delta_val < 0 or random.random() < exp(-delta_val / (temperature_1 + 1e-09)):
                        self.visited_points.add(point_key)
                        next_solutions.append(new_solution)
                    else:
                        next_solutions.append(solution.copy())
                else:
                    next_solutions.append(solution.copy())
            while len(next_solutions) < self.result_number:
                sol = self._random_solution(config_options)
                key = gen_key_from_design_point(sol)
                if key not in self.visited_points:
                    self.visited_points.add(key)
                    next_solutions.append(sol)
            cand_solutions = next_solutions
            temperature_1 *= max(0.95, 1 - FLAGS.cooling_rate * 0.2)
            if self.explored_point > 0 and self.explored_point % 200 == 0:
                restart_solution = self._random_solution(config_options)
                restart_key = gen_key_from_design_point(restart_solution)
                if restart_key not in self.visited_points and next_solutions:
                    self.visited_points.add(restart_key)
                    next_solutions[-1] = restart_solution
        self.log.info(f'Explored {self.explored_point} points')

class ACOExplorer(_ParetoArchiveExplorerMixin, Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=FLAGS.prune_class, point: DesignPoint=None):
        super(ACOExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.defaults_dict, self.config_options, self.config_cond = self.get_config_dafault_options()
        self.batch_size = 1
        self.params = self.config_options
        self.param_names = list(self.config_options.keys())
        self.n_params = len(self.param_names)
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.01
        self.pheromone = {param: np.ones(len(values)) for param, values in self.params.items()}
        self.pareto_front = []
        self.visited_points = set()
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            self.log.info('Best Results Found:')
            i = 1
            out_dir = join(BEST_RESULT_RUN_DIR, 'ACO')
            os.makedirs(out_dir, exist_ok=True)
            with open(join(out_dir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    class Ant:

        def __init__(self, aco):
            self.aco = aco
            self.solution = {}
            self.fitness = None
            self.violation = 0

        def construct_solution(self):
            if self.aco.explored_point % 200 == 0:
                init_solution = {}
                for key, value in self.aco.config_options.items():
                    init_solution[key] = value[randint(0, len(value) - 1)]
                self.solution = init_solution
                return
            for key in self.aco.params.keys():
                prob = self.aco.calculate_probability(key)
                selected_idx = np.random.choice(len(prob), p=prob)
                self.solution[key] = self.aco.params[key][selected_idx]

    def run(self) -> None:
        while self.explored_point <= self.stop_cond:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            ants = [self.Ant(self) for _ in range(self.result_number)]
            for ant in ants:
                max_retry = 20
                valid_solution = False
                for _ in range(max_retry):
                    ant.construct_solution()
                    point_key = gen_key_from_design_point(ant.solution)
                    if point_key not in self.visited_points:
                        self.visited_points.add(point_key)
                        valid_solution = True
                        break
                if not valid_solution:
                    continue
                result = self.get_results([ant.solution])
                if not result:
                    continue
                r = result[0]
                self.explored_point += 1
                self.update_best(r)
                self.best_save_results[self.explored_point] = self._build_perf_area_snapshot()
                ant.fitness = self._result_perf_area(r)
            self.update_pareto_front(ants)
            self.update_pheromone(ants)
        self.log.info(f'Explored {self.explored_point} points')

    def calculate_probability(self, param):
        pheromone = self.pheromone[param]
        heuristic = np.ones(len(pheromone))
        probabilities = pheromone ** self.alpha * heuristic ** self.beta
        probabilities /= np.sum(probabilities)
        return probabilities

    def update_pheromone(self, ants: List['ACOExplorer.Ant']):
        for param in self.param_names:
            self.pheromone[param] *= 1 - self.rho
        for ant in self.pareto_front:
            if ant.fitness is None:
                continue
            perf, area = ant.fitness
            reward = 1.0 / (perf + area + 1e-06)
            for param in self.param_names:
                idx = self.params[param].index(ant.solution[param])
                self.pheromone[param][idx] += reward

    def update_pareto_front(self, ants: List['ACOExplorer.Ant']):
        for ant in ants:
            if ant.fitness is None or ant.violation > 0:
                continue
            is_pareto = True
            to_remove = []
            for front_sol in self.pareto_front:
                if self.is_dominated(front_sol.fitness, ant.fitness):
                    to_remove.append(front_sol)
                elif self.is_dominated(ant.fitness, front_sol.fitness):
                    is_pareto = False
                    break
            for sol in to_remove:
                self.pareto_front.remove(sol)
            if is_pareto:
                self.pareto_front.append(ant)

    def is_dominated(self, sol_a, sol_b) -> bool:
        if sol_a is None or sol_b is None:
            return False
        return sol_b[0] <= sol_a[0] and sol_b[1] <= sol_a[1] and (sol_b[0] < sol_a[0] or sol_b[1] < sol_a[1])

class NSGAIIExplorer(_ParetoArchiveExplorerMixin, Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=FLAGS.prune_class, point: DesignPoint=None):
        super(NSGAIIExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.visited_points = set()
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            self.log.info('Best Results Found:')
            i = 1
            out_dir = join(BEST_RESULT_RUN_DIR, 'NSGAII')
            os.makedirs(out_dir, exist_ok=True)
            with open(join(out_dir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    def _sample_random_solution(self, config_options):
        sol = {}
        for key, value in config_options.items():
            sol[key] = value[randint(0, len(value) - 1)]
        return sol

    def _fitness_from_result(self, r: Result):
        perf, area = self._result_perf_area(r)
        return 1.0 / (perf + area + 1e-06)

    def _non_dominated_sort(self, results):
        fronts = []
        domination_count = {}
        dominated_set = {}
        rank = {}
        front = []
        for i, r1 in enumerate(results):
            domination_count[i] = 0
            dominated_set[i] = []
            p1, a1 = self._result_perf_area(r1)
            for j, r2 in enumerate(results):
                if i == j:
                    continue
                p2, a2 = self._result_perf_area(r2)
                if self._dominates_2d(p1, a1, p2, a2):
                    dominated_set[i].append(j)
                elif self._dominates_2d(p2, a2, p1, a1):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                rank[i] = 0
                front.append(i)
        fronts.append(front)
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        return fronts[:-1]

    def _crowding_distance(self, front, results):
        if not front:
            return {}
        distance = {i: 0.0 for i in front}
        objs = []
        for idx in front:
            perf, area = self._result_perf_area(results[idx])
            objs.append((idx, perf, area))
        for obj_id in [1, 2]:
            sorted_front = sorted(objs, key=lambda x: x[obj_id])
            distance[sorted_front[0][0]] = float('inf')
            distance[sorted_front[-1][0]] = float('inf')
            min_v = sorted_front[0][obj_id]
            max_v = sorted_front[-1][obj_id]
            if max_v == min_v:
                continue
            for i in range(1, len(sorted_front) - 1):
                prev_v = sorted_front[i - 1][obj_id]
                next_v = sorted_front[i + 1][obj_id]
                distance[sorted_front[i][0]] += (next_v - prev_v) / (max_v - min_v)
        return distance

    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        population = [self._sample_random_solution(config_options) for _ in range(self.result_number)]
        p_cs = 0.1
        p_mt = 0.1
        while self.explored_point <= self.stop_cond:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            results = self.get_results(population)
            valid_results = []
            valid_population = []
            for idx, r in enumerate(results):
                self.explored_point += 1
                if isinstance(r, Result) and r.valid:
                    self.update_best(r)
                    valid_results.append(r)
                    valid_population.append(population[idx])
                self.best_save_results[self.explored_point] = self._build_perf_area_snapshot()
                if self.explored_point > self.stop_cond:
                    break
            if len(valid_results) < 2:
                population = [self._sample_random_solution(config_options) for _ in range(self.result_number)]
                continue
            fronts = self._non_dominated_sort(valid_results)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= self.result_number:
                    selected.extend(front)
                else:
                    crowd = self._crowding_distance(front, valid_results)
                    remain = self.result_number - len(selected)
                    front_sorted = sorted(front, key=lambda x: crowd[x], reverse=True)
                    selected.extend(front_sorted[:remain])
                    break
            mating_pool = [valid_population[i] for i in selected]
            offspring = []
            while len(offspring) < self.result_number:
                p1 = random.choice(mating_pool)
                p2 = random.choice(mating_pool)
                c1, c2 = ({}, {})
                for para in config_options.keys():
                    if random.random() < p_cs:
                        c1[para] = p2[para]
                        c2[para] = p1[para]
                    else:
                        c1[para] = p1[para]
                        c2[para] = p2[para]
                for child in [c1, c2]:
                    for key in config_options.keys():
                        if random.random() < p_mt:
                            child[key] = config_options[key][randint(0, len(config_options[key]) - 1)]
                    offspring.append(child)
                    if len(offspring) >= self.result_number:
                        break
            population = offspring[:self.result_number]
        self.log.info(f'Explored {self.explored_point} points')
from collections import Counter
import os
import time
import random
import pickle
from os.path import join

class LatticeExplorer(_ParetoArchiveExplorerMixin, Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=FLAGS.prune_class, point: DesignPoint=None):
        super(LatticeExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.lattice_radius = 2
        self.max_neighbors = 8
        self.restart_prob = 0.1
        self.defaults_dict, self.config_options, self.config_cond = self.get_config_dafault_options()
        self.param_names = list(self.config_options.keys())
        self.visited = set()
        self.current_point = None
        self.log.info('Lattice Explorer initialization completed')
        start_time = time.time()
        if self.run_dse:
            _dump_run_meta_once()
            self.run()
            out_dir = join(BEST_RESULT_RUN_DIR, 'Lattice')
            os.makedirs(out_dir, exist_ok=True)
            with open(join(out_dir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.log.info('Pareto front designs:')
            i = 1
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
        end_time = time.time()
        print(f'Lattice execution time: {end_time - start_time}')

    def _random_solution(self):
        sol = {}
        for k, vals in self.config_options.items():
            sol[k] = random.choice(vals)
        return sol

    def _get_neighbors(self, point):
        neighbors = []
        for key in self.param_names:
            values = self.config_options[key]
            cur_idx = values.index(point[key])
            for delta in range(-self.lattice_radius, self.lattice_radius + 1):
                if delta == 0:
                    continue
                new_idx = cur_idx + delta
                if 0 <= new_idx < len(values):
                    new_point = dict(point)
                    new_point[key] = values[new_idx]
                    neighbors.append(new_point)
        random.shuffle(neighbors)
        return neighbors[:self.max_neighbors]

    def _evaluate(self, candidates):
        results = self.get_results(candidates)
        valid = []
        for r in results:
            if self.explored_point >= self.stop_cond:
                break
            self.explored_point += 1
            self.update_best(r)
            self.best_save_results[self.explored_point] = self._build_perf_area_snapshot()
            valid.append(r)
        return valid

    def run(self):
        self.current_point = self._random_solution()
        self.visited.add(gen_key_from_design_point(self.current_point))
        self._evaluate([self.current_point])
        while self.explored_point < self.stop_cond:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('explorer Lattice')
            print('------------------------------------------------------')
            if random.random() < self.restart_prob:
                self.current_point = self._random_solution()
                continue
            neighbors = self._get_neighbors(self.current_point)
            candidates = []
            for n in neighbors:
                key = gen_key_from_design_point(n)
                if key not in self.visited:
                    self.visited.add(key)
                    candidates.append(n)
            if not candidates:
                self.current_point = self._random_solution()
                continue
            results = self._evaluate(candidates)
            best = None
            best_perf = float('inf')
            for r in results:
                try:
                    perf = float(r.perf)
                except:
                    perf = float('inf')
                if perf < best_perf:
                    best_perf = perf
                    best = r
            if best is not None:
                self.current_point = dict(best.point)
            else:
                self.current_point = self._random_solution()
        self.log.info(f'Lattice exploration completed. Total design points explored: {self.explored_point}')

class MOEDAExplorer(_ParetoArchiveExplorerMixin, Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=FLAGS.prune_class, point: DesignPoint=None):
        super(MOEDAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.T = 10
        self.crossover_rate = 1.0
        self.mutation_rate = 0.2
        self.eda_update_rate = 0.2
        self.weight_vectors = []
        self.neighborhoods = []
        self.probability_vectors = []
        self.reference_point = None
        self.population = []
        self.defaults_dict, self.config_options, self.config_cond = self.get_config_dafault_options()
        self.param_names = list(self.config_options.keys())
        self.log.info('MOEDA initialization completed')
        start_time = time.time()
        if self.run_dse:
            _dump_run_meta_once()
            self.run()
            out_dir = join(BEST_RESULT_RUN_DIR, 'MOEDA')
            os.makedirs(out_dir, exist_ok=True)
            with open(join(out_dir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            self.log.info('Pareto front designs:')
            i = 1
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
        end_time = time.time()
        print(f'MOEDA execution time: {end_time - start_time}')

    def initialize_weight_vectors(self):
        self.weight_vectors = []
        for i in range(self.result_number):
            w1 = i / max(1, self.result_number - 1)
            self.weight_vectors.append([w1, 1.0 - w1])

    def initialize_neighborhoods(self):
        self.neighborhoods = []
        for i in range(self.result_number):
            dists = []
            for j in range(self.result_number):
                if i == j:
                    continue
                d = np.linalg.norm(np.array(self.weight_vectors[i]) - np.array(self.weight_vectors[j]))
                dists.append((j, d))
            dists.sort(key=lambda x: x[1])
            self.neighborhoods.append([idx for idx, _ in dists[:self.T]])

    def initialize_probability_vectors(self):
        self.probability_vectors = []
        for _ in range(self.result_number):
            prob_vector = {}
            for key, values in self.config_options.items():
                n = len(values)
                prob_vector[key] = [1.0 / n] * n
            self.probability_vectors.append(prob_vector)

    def _area_from_point_result(self, result):
        return self._area_from_result(result)

    def update_reference_point(self, result):
        area = self._area_from_point_result(result)
        perf = float(getattr(result, 'perf', float('inf')))
        if self.reference_point is None:
            self.reference_point = [area, perf]
            self.min_area = self.max_area = area
            self.min_latency = self.max_latency = perf
        else:
            self.reference_point[0] = min(self.reference_point[0], area)
            self.reference_point[1] = min(self.reference_point[1], perf)
            self.min_area = min(self.min_area, area)
            self.max_area = max(self.max_area, area)
            self.min_latency = min(self.min_latency, perf)
            self.max_latency = max(self.max_latency, perf)

    def normalize_objective(self, value, obj_type):
        if obj_type == 'area':
            lo = self.min_area
            hi = self.max_area
        else:
            lo = self.min_latency
            hi = self.max_latency
        if hi <= lo:
            return 0.0
        return (value - lo) / (hi - lo)

    def tchebycheff_scalarizing_function(self, result, weight_vector):
        area = self._area_from_point_result(result)
        perf = float(getattr(result, 'perf', float('inf')))
        na = self.normalize_objective(area, 'area')
        npf = self.normalize_objective(perf, 'perf')
        ref_a = self.normalize_objective(self.reference_point[0], 'area')
        ref_p = self.normalize_objective(self.reference_point[1], 'perf')
        return max(weight_vector[0] * abs(na - ref_a), weight_vector[1] * abs(npf - ref_p))

    def update_probability_vector(self, subproblem_idx):
        if subproblem_idx >= len(self.population):
            return
        neighbors = self.neighborhoods[subproblem_idx]
        neighborhood_solutions = [self.population[idx] for idx in neighbors if idx < len(self.population)]
        if not neighborhood_solutions:
            return
        prob_vector = {}
        for key, values in self.config_options.items():
            counts = {v: 1 for v in values}
            for sol in neighborhood_solutions:
                v = sol.get(key)
                if v in counts:
                    counts[v] += 1
            total = sum(counts.values())
            prob_vector[key] = [counts[v] / total for v in values]
        self.probability_vectors[subproblem_idx] = prob_vector

    def eda_update_operator(self, solution, subproblem_idx):
        child = dict(solution)
        prob_vector = self.probability_vectors[subproblem_idx]
        for key in self.param_names:
            if random.random() < self.eda_update_rate:
                values = self.config_options[key]
                probs = prob_vector[key]
                child[key] = random.choices(values, weights=probs)[0]
        return child

    def crossover_and_mutation(self, parent1, parent2):
        child1, child2 = ({}, {})
        keys = self.param_names
        if len(keys) > 1:
            cp = random.randint(1, len(keys) - 1)
        else:
            cp = 1
        for i, key in enumerate(keys):
            if random.random() < self.crossover_rate and i >= cp:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
            else:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
        for child in [child1, child2]:
            for key in keys:
                if random.random() < self.mutation_rate:
                    child[key] = random.choice(self.config_options[key])
        return (child1, child2)

    def _random_solution(self):
        sol = {}
        for key, values in self.config_options.items():
            sol[key] = random.choice(values)
        return sol

    def _evaluate_candidates(self, candidates):
        results = self.get_results(candidates)
        valid_results = []
        for result in results:
            if self.explored_point >= self.stop_cond:
                break
            self.explored_point += 1
            self.update_reference_point(result)
            self.update_best(result)
            self.best_save_results[self.explored_point] = self._build_perf_area_snapshot()
            valid_results.append(result)
        return valid_results

    def run(self) -> None:
        self.initialize_weight_vectors()
        self.initialize_neighborhoods()
        self.initialize_probability_vectors()
        self.population = [self._random_solution() for _ in range(self.result_number)]
        self._evaluate_candidates(self.population)
        while self.explored_point < self.stop_cond:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('explorer MOEDA')
            print('------------------------------------------------------')
            new_population = []
            for i in range(self.result_number):
                if self.explored_point >= self.stop_cond:
                    break
                neighbors = self.neighborhoods[i]
                if len(neighbors) < 2:
                    new_population.append(self.population[i])
                    continue
                p1_idx, p2_idx = random.sample(neighbors, 2)
                parent1 = self.population[p1_idx]
                parent2 = self.population[p2_idx]
                child1, child2 = self.crossover_and_mutation(parent1, parent2)
                child3 = self.eda_update_operator(self.population[i], i)
                candidates = [child1, child2, child3]
                results = self._evaluate_candidates(candidates)
                best_solution = self.population[i]
                best_score = float('inf')
                for result in results:
                    score = self.tchebycheff_scalarizing_function(result, self.weight_vectors[i])
                    if score < best_score:
                        best_score = score
                        best_solution = dict(result.point)
                new_population.append(best_solution)
                self.update_probability_vector(i)
            if new_population:
                self.population = new_population
            else:
                break
        self.log.info(f'MOEDA exploration completed. Total design points explored: {self.explored_point}')

class ExhaustiveExplorer(Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool=True, prune_invalid=FLAGS.prune_class, point: DesignPoint=None):
        super(ExhaustiveExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        cfg_timeout = float(getattr(FLAGS, 'exhaustive_timeout', 0))
        self.timeout = float('inf') if cfg_timeout <= 0 else cfg_timeout
        self.num_top_designs = 1000000
        self._raw_perf_util_records: List[Dict[str, Any]] = []
        self.log.info('Done init')
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            if cfg_timeout > 0:
                out_dir = join(get_root_path(), 'best_result_runs', 'GNNDSE')
            else:
                out_dir = join(get_root_path(), 'best_result_runs', 'ref_GNN')
            os.makedirs(out_dir, exist_ok=True)
            with open(join(out_dir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            raw_path = join(out_dir, f'{kernel_name}_raw_perf_util.json')
            try:
                with open(raw_path, 'w', encoding='utf-8') as f:
                    json.dump({'kernel': kernel_name, 'num_records': len(self._raw_perf_util_records), 'records': self._raw_perf_util_records}, f, indent=2, ensure_ascii=False)
                self.log.info(f'[REF] Raw perf/util saved: {raw_path}')
            except Exception as e:
                self.log.warning(f'[REF] Failed to save raw perf/util records: {e}')
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join(('%s: %s' % item for item in attrs.items())))

    def gen(self) -> Generator[List[DesignPoint], Optional[Dict[str, Result]], None]:
        self.log.info('Launch exhaustive search algorithm')
        traverser = self.traverse(get_default_point(self.ds), 0)
        iter_cnt = 0
        while True:
            next_points: List[DesignPoint] = []
            try:
                iter_cnt += 1
                self.log.debug(f'Iteration {iter_cnt}')
                while len(next_points) < self.batch_size:
                    next_points.append(next(traverser))
                    self.log.debug(f'Next point: {str(next_points[-1])}')
                yield next_points
            except StopIteration:
                if next_points:
                    yield next_points
                break
        self.log.info('No more points to be explored, stop.')

    def run(self) -> None:
        gen_next = self.gen()
        timer = time.time()
        duplicated_iters = 0
        completed = False
        while time.time() - timer < self.timeout:
            try:
                next_points = next(gen_next)
                self.log.debug(f'The algorithm generates {len(next_points)} design points')
            except StopIteration:
                completed = True
                break
            results = self.get_results(next_points)
            for r in results:
                if isinstance(r, Result):
                    self.explored_point += 1
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(', '.join(('%s: %s' % item for item in attrs.items())))
                    self.update_best(r)
                    try:
                        perf_v = float(r.perf)
                        util_sum_v = 0.0
                        util_dict: Dict[str, float] = {}
                        if isinstance(getattr(r, 'res_util', None), dict):
                            for kk, vv in r.res_util.items():
                                if isinstance(kk, str) and kk.startswith('util'):
                                    fvv = float(vv)
                                    util_dict[kk] = fvv
                                    util_sum_v += max(0.0, fvv)
                        self._raw_perf_util_records.append({'perf': perf_v, 'util_sum': float(util_sum_v), 'res_util': util_dict})
                    except Exception:
                        pass
        self.log.info(f'Explored {self.explored_point} points')
        try:
            final_points = []
            for _r in self.best_results_dict.values():
                try:
                    _perf = float(getattr(_r, 'perf', float('inf')))
                except Exception:
                    _perf = float('inf')
                _vals = []
                if isinstance(getattr(_r, 'res_util', None), dict):
                    for _k, _u in _r.res_util.items():
                        if isinstance(_k, str) and _k.startswith('util'):
                            try:
                                _vals.append(float(_u))
                            except Exception:
                                continue
                _area = float(sum(_vals) / len(_vals)) if _vals else float('inf')
                final_points.append([_perf, _area])
            final_pareto = []
            for i, p in enumerate(final_points):
                pi_perf, pi_area = (p[0], p[1])
                dominated = False
                for j, q in enumerate(final_points):
                    if i == j:
                        continue
                    q_perf, q_area = (q[0], q[1])
                    if (q_perf <= pi_perf and q_area <= pi_area) and (q_perf < pi_perf or q_area < pi_area):
                        dominated = True
                        break
                if not dominated:
                    final_pareto.append(p)
            self.best_save_results = {self.explored_point: final_pareto}
        except Exception:
            pass
        try:
            if completed and hasattr(self, 'ds_size') and isinstance(self.ds_size, int):
                if self.explored_point < self.ds_size:
                    self.log.warning(f'[REF] Exhaustive completed traversal but explored_point({self.explored_point}) < ds_size({self.ds_size}). Ref may be incomplete due to duplicates/collisions.')
                else:
                    self.log.info(f'[REF] Exhaustive traversal complete: explored_point={self.explored_point}, ds_size={self.ds_size}')
            elif not completed:
                self.log.warning(f"[REF] Exhaustive stopped early (timeout/interruption?): explored_point={self.explored_point}, ds_size={getattr(self, 'ds_size', None)}")
        except Exception:
            pass
        try:
            msg = f"[REF_PRINT] completed={completed}, explored_point={self.explored_point}, ds_size={getattr(self, 'ds_size', None)}, timeout={getattr(self, 'timeout', None)}"
            print(msg, flush=True)
            try:
                import atexit

                def _print_ref_at_exit(m=msg):
                    try:
                        print(m, flush=True)
                    except Exception:
                        pass
                atexit.register(_print_ref_at_exit)
            except Exception:
                pass
        except Exception:
            pass