from src.config import FLAGS
from src.saver import saver
from src.utils import get_root_path, MLP, print_stats, get_save_path, create_dir_if_not_exists, plot_dist, load
from src.result import Result
from os.path import join, basename
from glob import glob, iglob
from math import ceil
from sklearn.preprocessing import OneHotEncoder

def _get_data_class():
    from torch_geometric.data import Data
    return Data

def _get_batch_class():
    from torch_geometric.data import Batch
    return Batch

def _get_dataset_class():
    from torch_geometric.data import Dataset
    return Dataset
import networkx as nx
import redis, pickle, random
import numpy as np
from collections import Counter, defaultdict, OrderedDict
import sys
import os
_dse_db_path = join(get_root_path(), 'dse_database')
if _dse_db_path not in sys.path:
    sys.path.insert(0, _dse_db_path)
from scipy.sparse import hstack, vstack
from scipy.sparse import csr_matrix
from tqdm import tqdm
import os.path as osp
import torch
from shutil import rmtree
import pandas as pd
import math
NON_OPT_PRAGMAS = ['LOOP_TRIPCOUNT', 'INTERFACE', 'INTERFACE', 'KERNEL']
WITH_VAR_PRAGMAS = ['DEPENDENCE', 'RESOURCE', 'STREAM', 'ARRAY_PARTITION']
TARGET = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']
tag = 'new_speedup'
SAVE_DIR = join(get_save_path(), FLAGS.dataset, f"new-train-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{''.join(TARGET)}")
ENCODER_PATH = '/home/yutao/MPM/save_models_and_data'
create_dir_if_not_exists(SAVE_DIR)
DATASET = 'machsuite-poly'
if DATASET == 'machsuite-poly':
    KERNEL = FLAGS.tag
    db_path = []
    for benchmark in FLAGS.benchmarks:
        db_path.append(f'../dse_database/{benchmark}/databases/**/*')
else:
    raise NotImplementedError()
if FLAGS.dataset == 'programl':
    GEXF_FOLDER = join(get_root_path(), 'dse_database', 'programl', '**', 'processed', '**')
else:
    raise NotImplementedError()
import config
TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL
if FLAGS.all_kernels:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf')])
else:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and KERNEL in f])

def finte_diff_as_quality(new_result: Result, ref_result: Result) -> float:

    def quantify_util(result: Result) -> float:
        utils = [5 * ceil(u * 100 / 5) / 100 + FLAGS.epsilon for k, u in result.res_util.items() if k.startswith('util')]
        return sum([2.0 ** (1.0 / (1.0 - u)) for u in utils])
    ref_util = quantify_util(ref_result)
    new_util = quantify_util(new_result)
    if new_util == ref_util:
        if new_result.perf < ref_result.perf:
            return FLAGS.max_number
        return 0
    return -(new_result.perf - ref_result.perf) / (new_util - ref_util)
_MyOwnDatasetClass = None

def _get_my_own_dataset_class():
    global _MyOwnDatasetClass
    if _MyOwnDatasetClass is None:
        Dataset = _get_dataset_class()

        class _MyOwnDatasetImpl(Dataset):

            def __init__(self, transform=None, pre_transform=None):
                self._gp_cache = None
                self._cp_cache = None
                self._points_list = None
                points_file = osp.join(SAVE_DIR, 'points_list.pkl')
                if osp.exists(points_file):
                    try:
                        with open(points_file, 'rb') as f:
                            self._points_list = pickle.load(f)
                        saver.log_info(f'Loaded {len(self._points_list)} points from {points_file}')
                        non_none_count = sum((1 for p in self._points_list if p is not None))
                        saver.log_info(f'Points list: {non_none_count}/{len(self._points_list)} are non-None')
                        if non_none_count == 0:
                            saver.warning(f'⚠️  All points in points_list are None! This may indicate a data loading issue.')
                    except Exception as e:
                        saver.warning(f'Failed to load points from {points_file}: {e}')
                        import traceback
                        saver.warning(f'Traceback: {traceback.format_exc()}')
                        self._points_list = None
                else:
                    saver.warning(f'Points file not found: {points_file}')
                    saver.warning(f'⚠️  To enable causal training, you need to run with --force_regen=True to generate points_list.pkl')
                    saver.warning(f'⚠️  Without points_list.pkl, design_point will be None and causal model will not work properly')
                super().__init__(SAVE_DIR, transform, pre_transform)

            def get_point(self, idx):
                if self._points_list is not None and 0 <= idx < len(self._points_list):
                    return self._points_list[idx]
                try:
                    data_file = osp.join(SAVE_DIR, f'data_{idx}.pt')
                    if osp.exists(data_file):
                        data = torch.load(data_file)
                        if hasattr(data, 'point') and data.point is not None:
                            return data.point
                except Exception as e:
                    pass
                return None

            @property
            def raw_file_names(self):
                return []

            def _normalize_kernel_name(self, kernel_name):
                if kernel_name is None:
                    return None
                if kernel_name.endswith('_processed_result'):
                    return kernel_name[:-len('_processed_result')]
                return kernel_name

            def _get_processed_file_dicts(self):
                if self._gp_cache is not None and self._cp_cache is not None:
                    return (self._gp_cache, self._cp_cache)
                all_files = sorted(glob(join(SAVE_DIR, 'data_*.pt')), key=lambda x: int(basename(x).replace('data_', '').replace('.pt', '')))
                gp = defaultdict(list)
                cp = defaultdict(list)
                for file_path in all_files:
                    try:
                        data = torch.load(file_path)
                        kernel_raw = getattr(data, 'kernel', None)
                        if kernel_raw is not None:
                            kernel = self._normalize_kernel_name(kernel_raw)
                            gp[kernel].append(file_path)
                            cp[kernel].append(file_path)
                    except Exception as e:
                        saver.warning(f'Failed to load {file_path} to determine kernel: {e}')
                self._gp_cache = dict(gp)
                self._cp_cache = dict(cp)
                return (self._gp_cache, self._cp_cache)

            @property
            def processed_file_names(self):
                all_files = sorted(glob(join(SAVE_DIR, 'data_*.pt')), key=lambda x: int(basename(x).replace('data_', '').replace('.pt', '')))
                return all_files

            @property
            def processed_file_names_dict(self):
                return self._get_processed_file_dicts()

            def download(self):
                pass

            def process(self):
                pass

            def len(self):
                return len(glob(join(SAVE_DIR, '*.pt')))

            def __len__(self):
                return self.len()

            def get(self, idx):
                data = torch.load(osp.join(SAVE_DIR, 'data_{}.pt'.format(idx)))
                return data

            def get_data(self, idx, k):
                gp, cp = self.processed_file_names_dict
                if k not in gp or idx >= len(gp[k]):
                    raise IndexError(f'Index {idx} out of range for kernel {k} (total: {len(gp[k])})')
                graph_path = gp[k][idx]
                data = torch.load(graph_path)
                try:
                    file_idx = int(basename(graph_path).replace('data_', '').replace('.pt', ''))
                    p = self.get_point(file_idx)
                    if p is not None:
                        setattr(data, 'point', p)
                except Exception:
                    pass
                code_path = cp[k][idx] if k in cp and idx < len(cp[k]) else None
                if code_path and code_path != graph_path:
                    code_d = torch.load(code_path)
                else:
                    code_d = None
                return (data, code_d)
        _MyOwnDatasetClass = _MyOwnDatasetImpl
    return _MyOwnDatasetClass

class MyOwnDataset:

    def __new__(cls, transform=None, pre_transform=None):
        actual_class = _get_my_own_dataset_class()
        return actual_class(transform, pre_transform)

def get_data_list():
    from os.path import join as path_join
    saver.log_info('=' * 60)
    saver.log_info('[Data Loading] Loading data from two_tower_dataset directory')
    saver.log_info('=' * 60)
    graph_path = path_join(get_root_path(), 'two_tower_dataset', 'graph')
    code_path = path_join(get_root_path(), 'two_tower_dataset', 'code')
    if not osp.exists(graph_path):
        saver.error(f'two_tower_dataset/graph directory not found: {graph_path}')
        saver.error('Please run gen_dataset.py first to generate the dataset')
        raise FileNotFoundError(f'two_tower_dataset/graph directory not found: {graph_path}')
    data_list = []
    global_points_list = []
    all_kernels_ordered = MACHSUITE_KERNEL + poly_KERNEL
    for kernel in all_kernels_ordered:
        if kernel in MACHSUITE_KERNEL:
            benchmark = 'machsuite'
        else:
            benchmark = 'poly'
        kernel_graph_dir = path_join(graph_path, benchmark, kernel)
        kernel_code_dir = path_join(code_path, benchmark, kernel)
        if not osp.exists(kernel_graph_dir):
            saver.warning(f'Kernel directory not found: {kernel_graph_dir}, skipping...')
            continue
        points_base_dir = path_join(get_root_path(), 'two_tower_dataset', 'points')
        kernel_points_file = path_join(points_base_dir, benchmark, kernel, 'points_list.pkl')
        kernel_points_list = []
        if osp.exists(kernel_points_file):
            try:
                with open(kernel_points_file, 'rb') as f:
                    kernel_points_list = pickle.load(f)
                saver.log_info(f'Loaded {len(kernel_points_list)} points from {kernel_points_file}')
            except Exception as e:
                saver.warning(f'Failed to load points from {kernel_points_file}: {e}')
        else:
            saver.warning(f'points_list.pkl not found for kernel {kernel}: {kernel_points_file}')
            old_points_file = path_join(kernel_graph_dir, 'points_list.pkl')
            if osp.exists(old_points_file):
                try:
                    with open(old_points_file, 'rb') as f:
                        kernel_points_list = pickle.load(f)
                    saver.log_info(f'Loaded {len(kernel_points_list)} points from old location: {old_points_file}')
                except Exception as e:
                    saver.warning(f'Failed to load points from old location: {e}')
        kernel_data_files = sorted(glob(path_join(kernel_graph_dir, 'data_*.pt')), key=lambda x: int(osp.basename(x).replace('data_', '').replace('.pt', '')))
        saver.log_info(f'Loading {len(kernel_data_files)} data files for kernel {kernel}')
        for data_file in kernel_data_files:
            try:
                data = torch.load(data_file)
                data_list.append(data)
                file_idx = int(osp.basename(data_file).replace('data_', '').replace('.pt', ''))
                if file_idx < len(kernel_points_list):
                    global_points_list.append(kernel_points_list[file_idx])
                else:
                    global_points_list.append(None)
            except Exception as e:
                saver.warning(f'Failed to load {data_file}: {e}')
                continue
    saver.log_info(f'=' * 60)
    saver.log_info(f'[Data Loading Summary]')
    saver.log_info(f'  Total data files loaded: {len(data_list)}')
    saver.log_info(f'  Total points loaded: {len(global_points_list)}')
    non_none_points = sum((1 for p in global_points_list if p is not None))
    saver.log_info(f'  Non-None points: {non_none_points}/{len(global_points_list)}')
    saver.log_info(f'=' * 60)
    if FLAGS.force_regen:
        saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
        rmtree(SAVE_DIR)
        create_dir_if_not_exists(SAVE_DIR)
        for i in tqdm(range(len(data_list))):
            torch.save(data_list[i], osp.join(SAVE_DIR, f'data_{i}.pt'))
        points_file = osp.join(SAVE_DIR, 'points_list.pkl')
        non_none_count = sum((1 for p in global_points_list if p is not None))
        saver.log_info(f'Preparing to save global points_list: {len(global_points_list)} total, {non_none_count} non-None')
        with open(points_file, 'wb') as f:
            pickle.dump(global_points_list, f)
        saver.log_info(f'Saved {len(global_points_list)} points to {points_file} (non-None: {non_none_count})')
        if osp.exists(points_file):
            file_size = osp.getsize(points_file)
            saver.log_info(f'Points file size: {file_size} bytes')
    if FLAGS.force_regen:
        from utils import save
        encoder_file = path_join(get_root_path(), 'save_models_and_data', 'encoders')
        if osp.exists(encoder_file + '.klepto'):
            try:
                encoders = load(encoder_file)
                enc_ntype = encoders['enc_ntype']
                enc_ptype = encoders['enc_ptype']
                enc_itype = encoders['enc_itype']
                enc_ftype = encoders['enc_ftype']
                enc_btype = encoders['enc_btype']
                enc_ftype_edge = encoders['enc_ftype_edge']
                enc_ptype_edge = encoders['enc_ptype_edge']
                saver.log_info(f'Loaded encoders from {encoder_file}')
            except Exception as e:
                saver.warning(f'Failed to load encoders from {encoder_file}: {e}')
                saver.warning('Encoders will need to be created separately')
                enc_ntype = OneHotEncoder(handle_unknown='ignore')
                enc_ptype = OneHotEncoder(handle_unknown='ignore')
                enc_itype = OneHotEncoder(handle_unknown='ignore')
                enc_ftype = OneHotEncoder(handle_unknown='ignore')
                enc_btype = OneHotEncoder(handle_unknown='ignore')
                enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
                enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')
        else:
            saver.warning(f'Encoder file not found: {encoder_file}')
            saver.warning('Encoders will need to be created separately')
            enc_ntype = OneHotEncoder(handle_unknown='ignore')
            enc_ptype = OneHotEncoder(handle_unknown='ignore')
            enc_itype = OneHotEncoder(handle_unknown='ignore')
            enc_ftype = OneHotEncoder(handle_unknown='ignore')
            enc_btype = OneHotEncoder(handle_unknown='ignore')
            enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
            enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')
        obj = {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype, 'enc_itype': enc_itype, 'enc_ftype': enc_ftype, 'enc_btype': enc_btype, 'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge}
        save(obj, join(ENCODER_PATH, 'encoders'))
    Dataset = _get_my_own_dataset_class()
    dataset = Dataset()
    if FLAGS.force_regen and hasattr(dataset, '_points_list'):
        if dataset._points_list is None:
            dataset._points_list = global_points_list
            saver.log_info(f'Manually set _points_list in Dataset: {len(global_points_list)} points')
    pragma_dim = 0
    if global_points_list:
        for point in global_points_list:
            if point is not None and isinstance(point, dict):
                pragma_dim = len(point)
                break
    return (dataset, pragma_dim)

def get_data_list_old():
    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')
    database = redis.StrictRedis(host='localhost', port=6379)
    ntypes = Counter()
    ptypes = Counter()
    numerics = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()
    if FLAGS.encoder_path != None:
        encoders = load(FLAGS.encoder_path)
        enc_ntype = encoders['enc_ntype']
        enc_ptype = encoders['enc_ptype']
        enc_itype = encoders['enc_itype']
        enc_ftype = encoders['enc_ftype']
        enc_btype = encoders['enc_btype']
        enc_ftype_edge = encoders['enc_ftype_edge']
        enc_ptype_edge = encoders['enc_ptype_edge']
    else:
        enc_ntype = OneHotEncoder(handle_unknown='ignore')
        enc_ptype = OneHotEncoder(handle_unknown='ignore')
        enc_itype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype = OneHotEncoder(handle_unknown='ignore')
        enc_btype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
        enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')
    data_list = []
    all_gs = OrderedDict()
    X_ntype_all = []
    X_ptype_all = []
    X_itype_all = []
    X_ftype_all = []
    X_btype_all = []
    edge_ftype_all = []
    edge_ptype_all = []
    tot_configs = 0
    num_files = 0
    init_feat_dict = {}
    for gexf_file in tqdm(GEXF_FILES[0:]):
        if FLAGS.dataset == 'machsuite' or 'programl' in FLAGS.dataset:
            proceed = False
            for k in ALL_KERNEL:
                if k in gexf_file:
                    proceed = True
                    break
            if not proceed:
                continue
        else:
            raise NotImplementedError()
        g = nx.read_gexf(gexf_file)
        g.variants = OrderedDict()
        gname = basename(gexf_file).split('.')[0]
        saver.log_info(gname)
        all_gs[gname] = g
        n = basename(gexf_file).split('_')[0]
        if FLAGS.dataset == 'programl':
            db_paths = []
            for db_p in db_path:
                paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f]
                db_paths.extend(paths)
            if db_paths is None:
                saver.warning(f'No database found for {n}. Skipping.')
                continue
        else:
            raise NotImplementedError()
        database.flushdb()
        saver.log_info(f'db_paths for {n}:')
        for d in db_paths:
            saver.log_info(f'{d}')
        if len(db_paths) == 0:
            saver.log_info(f'{n} has no db_paths')
        assert len(db_paths) >= 1
        for idx, file in enumerate(db_paths):
            f_db = open(file, 'rb')
            data = pickle.load(f_db)
            database.hmset(0, data)
            max_idx = idx + 1
            f_db.close()
        keys = [k.decode('utf-8') for k in database.hkeys(0)]
        lv2_keys = [k for k in keys if 'lv2' in k]
        saver.log_info(f'num keys for {n}: {len(keys)} and lv2 keys: {len(lv2_keys)}')
        got_reference = False
        res_reference = 0
        max_perf = 0
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            try:
                pickle_obj_fixed = pickle_obj.replace(b'localdse', b'dse_database.autodse').replace(b'autodse', b'dse_database.autodse')
                obj = pickle.loads(pickle_obj_fixed)
            except (ModuleNotFoundError, AttributeError, ImportError) as e:
                try:
                    obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
                except Exception as e2:
                    saver.log_info(f'Warning: Failed to unpickle {key}: {e}, {e2}. Skipping.')
                    continue
            if type(obj) is int or type(obj) is dict:
                continue
            if key[0:3] == 'lv1' or obj.perf == 0:
                continue
            if obj.perf > max_perf:
                max_perf = obj.perf
                got_reference = True
                res_reference = obj
        if res_reference != 0:
            saver.log_info(f'reference point for {n} is {res_reference.perf}')
        else:
            saver.log_info(f'did not find reference point for {n} with {len(keys)} points')
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            try:
                pickle_obj_fixed = pickle_obj.replace(b'localdse', b'dse_database.autodse').replace(b'autodse', b'dse_database.autodse')
                obj = pickle.loads(pickle_obj_fixed)
            except (ModuleNotFoundError, AttributeError, ImportError) as e:
                try:
                    obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
                except Exception as e2:
                    saver.log_info(f'Warning: Failed to unpickle {key}: {e}, {e2}. Skipping.')
                    continue
            if type(obj) is int or type(obj) is dict:
                continue
            if FLAGS.task == 'regression' and key[0:3] == 'lv1':
                continue
            if FLAGS.task == 'regression' and (not FLAGS.invalid) and (obj.perf == 0):
                continue
            xy_dict = _encode_X_dict(g, ntypes=ntypes, ptypes=ptypes, itypes=itypes, ftypes=ftypes, btypes=btypes, numerics=numerics, obj=obj)
            edge_dict = _encode_edge_dict(g, ftypes=ftypes_edge, ptypes=ptypes_edge)
            if FLAGS.task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(obj.perf + FLAGS.epsilon)
                        elif 'const' in FLAGS.norm_method:
                            y = obj.perf * FLAGS.normalizer
                            if y == 0:
                                y = FLAGS.max_number * FLAGS.normalizer
                            if FLAGS.norm_method == 'const-log2':
                                y = math.log2(y)
                        elif 'speedup' in FLAGS.norm_method:
                            assert obj.perf != 0
                            if tag == 'new_speedup':
                                y = FLAGS.normalizer / obj.perf
                            elif obj.perf == 0:
                                y = 0
                            else:
                                y = res_reference.perf / obj.perf
                            if FLAGS.norm_method == 'speedup-log2':
                                y = math.log2(y)
                        elif FLAGS.norm_method == 'off':
                            y = obj.perf
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                        xy_dict['kernel_speedup'] = torch.FloatTensor(np.array([math.log2(res_reference.perf / obj.perf)]))
                    elif tname == 'quality':
                        y = finte_diff_as_quality(obj, res_reference)
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(y + FLAGS.epsilon)
                        elif FLAGS.norm_method == 'const':
                            y = y * FLAGS.normalizer
                        elif FLAGS.norm_method == 'off':
                            pass
                    elif 'util' in tname or 'total' in tname:
                        y = obj.res_util[tname]
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            elif FLAGS.task == 'class':
                if 'lv1' in key:
                    lv2_key = key.replace('lv1', 'lv2')
                    if lv2_key in keys:
                        continue
                    else:
                        y = 0
                else:
                    y = obj.perf if obj.perf == 0 else 1
                xy_dict['perf'] = torch.FloatTensor(np.array([y])).type(torch.LongTensor)
            else:
                raise NotImplementedError()
            vname = key
            if hasattr(obj, 'point') and obj.point is not None:
                g.variants[vname] = (xy_dict, edge_dict, obj.point)
            else:
                g.variants[vname] = (xy_dict, edge_dict, None)
            X_ntype_all += xy_dict['X_ntype']
            X_ptype_all += xy_dict['X_ptype']
            X_itype_all += xy_dict['X_itype']
            X_ftype_all += xy_dict['X_ftype']
            X_btype_all += xy_dict['X_btype']
            edge_ftype_all += edge_dict['X_ftype']
            edge_ptype_all += edge_dict['X_ptype']
        tot_configs += len(g.variants)
        num_files += 1
        saver.log_info(f'{n} g.variants {len(g.variants)} tot_configs {tot_configs}')
        saver.log_info(f'\tntypes {len(ntypes)}')
        saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
        saver.log_info(f'\tnumerics {len(numerics)} {numerics}')
    if FLAGS.encoder_path == None:
        enc_ptype.fit(X_ptype_all)
        enc_ntype.fit(X_ntype_all)
        enc_itype.fit(X_itype_all)
        enc_ftype.fit(X_ftype_all)
        enc_btype.fit(X_btype_all)
        enc_ftype_edge.fit(edge_ftype_all)
        enc_ptype_edge.fit(edge_ptype_all)
        saver.log_info(f'Done {num_files} files tot_configs {tot_configs}')
        saver.log_info(f'\tntypes {len(ntypes)}')
        saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
        saver.log_info(f'\tnumerics {len(numerics)} {numerics}')
    points_list = []
    for gname, g in all_gs.items():
        edge_index = create_edge_index(g)
        saver.log_info('edge_index created', gname)
        for vname, d in g.variants.items():
            if len(d) == 3:
                d_node, d_edge, point = d
            else:
                d_node, d_edge = d
                point = None
            points_list.append(point)
            X = _encode_X_torch(d_node, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
            edge_attr = _encode_edge_torch(d_edge, enc_ftype_edge, enc_ptype_edge)
            if FLAGS.task == 'regression':
                Data = _get_data_class()
                data_obj = Data(x=X, edge_index=edge_index, perf=d_node['perf'], actual_perf=d_node['actual_perf'], kernel_speedup=d_node['kernel_speedup'], quality=d_node['quality'], util_BRAM=d_node['util-BRAM'], util_DSP=d_node['util-DSP'], util_LUT=d_node['util-LUT'], util_FF=d_node['util-FF'], total_BRAM=d_node['total-BRAM'], total_DSP=d_node['total-DSP'], total_LUT=d_node['total-LUT'], total_FF=d_node['total-FF'], edge_attr=edge_attr, kernel=gname)
                data_list.append(data_obj)
            elif FLAGS.task == 'class':
                Data = _get_data_class()
                data_list.append(Data(x=X, edge_index=edge_index, perf=d_node['perf'], edge_attr=edge_attr, kernel=gname))
            else:
                raise NotImplementedError()
    nns = [d.x.shape[0] for d in data_list]
    print_stats(nns, 'number of nodes')
    ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
    print_stats(ads, 'avg degrees')
    saver.log_info('dataset[0].num_features', data_list[0].num_features)
    for target in TARGETS:
        if not hasattr(data_list[0], target.replace('-', '_')):
            saver.warning(f'Data does not have attribute {target}')
            continue
        ys = [_get_y(d, target).item() for d in data_list]
        plot_dist(ys, f'{target}_ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
        saver.log_info(f'{target}_ys', Counter(ys))
    if FLAGS.force_regen:
        saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
        rmtree(SAVE_DIR)
        create_dir_if_not_exists(SAVE_DIR)
        for i in tqdm(range(len(data_list))):
            torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i)))
        points_file = osp.join(SAVE_DIR, 'points_list.pkl')
        non_none_count = sum((1 for p in points_list if p is not None))
        saver.log_info(f'Preparing to save points_list: {len(points_list)} total, {non_none_count} non-None')
        with open(points_file, 'wb') as f:
            pickle.dump(points_list, f)
        saver.log_info(f'Saved {len(points_list)} points to {points_file} (non-None: {non_none_count})')
        if osp.exists(points_file):
            file_size = osp.getsize(points_file)
            saver.log_info(f'Points file size: {file_size} bytes')
    if FLAGS.force_regen:
        from utils import save
        obj = {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype, 'enc_itype': enc_itype, 'enc_ftype': enc_ftype, 'enc_btype': enc_btype, 'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge}
        save(obj, join(ENCODER_PATH, 'encoders'))
        save(init_feat_dict, join('/home/yutao/MPM/save_models_and_data', 'pragma_dim'))
        for gname, feat_dim in init_feat_dict.items():
            saver.log_info(f'{gname} has initial dim {feat_dim}')
    rtn = MyOwnDataset()
    return (rtn, init_feat_dict)

def _get_y(data, target):
    return getattr(data, target.replace('-', '_'))

def print_data_stats(data_loader, tvt):
    nns, ads, ys = ([], [], [])
    for d in tqdm(data_loader):
        nns.append(d.x.shape[0])
        ys.append(d.y.item())
    print_stats(nns, f'{tvt} number of nodes')
    plot_dist(ys, f'{tvt} ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
    saver.log_info(f'{tvt} ys', Counter(ys))

def load_all_gs(remove_all_pragma_nodes):
    rtn = []
    for gexf_file in tqdm(GEXF_FILES[0:]):
        g = nx.read_gexf(gexf_file)
        rtn.append(g)
        if remove_all_pragma_nodes:
            before = g.number_of_nodes()
            nodes_to_remove = []
            for node, ndata in g.nodes(data=True):
                if 'pragma' in ndata['full_text']:
                    nodes_to_remove.append(node)
            g.remove_nodes_from(nodes_to_remove)
            print(f'Removed {len(nodes_to_remove)} pragma nodes; before {before} now {g.number_of_nodes}')
    return rtn

def load_encoders():
    from utils import load
    rtn = load(ENCODER_PATH)
    return rtn

def encode_g_torch(g, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype):
    x_dict = _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None, itypes=None, eftypes=None, btypes=None, obj=None)
    X = _encode_X_torch(x_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
    edge_index = create_edge_index(g)
    return (X, edge_index)

def _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None, itypes=None, ftypes=None, btypes=None, obj=None):
    X_ntype = []
    X_ptype = []
    X_numeric = []
    X_itype = []
    X_ftype = []
    X_btype = []
    for node, ndata in g.nodes(data=True):
        if ntypes is not None:
            ntypes[ndata['type']] += 1
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
                if obj is not None:
                    t_li = p_text_type.split(' ')
                    for i in range(len(t_li)):
                        if 'AUTO{' in t_li[i]:
                            auto_what = _in_between(t_li[i], '{', '}')
                            numeric = obj.point[auto_what]
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
        if ptypes is not None:
            ptypes[ptype] += 1
        if numerics is not None:
            numerics[numeric] += 1
        X_ntype.append([ndata['type']])
        X_ptype.append([ptype])
        X_numeric.append([numeric])
        X_itype.append([ndata['text']])
        X_ftype.append([ndata['function']])
        X_btype.append([ndata['block']])
    return {'X_ntype': X_ntype, 'X_ptype': X_ptype, 'X_numeric': X_numeric, 'X_itype': X_itype, 'X_ftype': X_ftype, 'X_btype': X_btype}

def _encode_X_torch(x_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype):
    X_ntype = enc_ntype.transform(x_dict['X_ntype'])
    X_ptype = enc_ptype.transform(x_dict['X_ptype'])
    X_itype = enc_itype.transform(x_dict['X_itype'])
    X_ftype = enc_ftype.transform(x_dict['X_ftype'])
    X_btype = enc_btype.transform(x_dict['X_btype'])
    X_numeric = x_dict['X_numeric']
    if FLAGS.no_pragma:
        X = X_ntype
        X = X.toarray()
        X = torch.FloatTensor(X)
    else:
        X = hstack((X_ntype, X_ptype, X_numeric, X_itype, X_ftype, X_btype))
        X = _coo_to_sparse(X)
        X = X.to_dense()
    return X

def _encode_edge_dict(g, ftypes=None, ptypes=None):
    X_ftype = []
    X_ptype = []
    for nid1, nid2, edata in g.edges(data=True):
        X_ftype.append([edata['flow']])
        X_ptype.append([edata['position']])
    return {'X_ftype': X_ftype, 'X_ptype': X_ptype}

def _encode_edge_torch(edge_dict, enc_ftype, enc_ptype):
    X_ftype = enc_ftype.transform(edge_dict['X_ftype'])
    X_ptype = enc_ptype.transform(edge_dict['X_ptype'])
    X = hstack((X_ftype, X_ptype))
    X = _coo_to_sparse(X.tocoo())
    X = X.to_dense()
    return X

def _in_between(text, left, right):
    return text[text.index(left) + len(left):text.index(right)]

def _check_any_in_str(li, s):
    for li_item in li:
        if li_item in s:
            return True
    return False

def create_edge_index(g):
    g = nx.convert_node_labels_to_integers(g, ordering='sorted')
    edge_index = torch.LongTensor(list(g.edges)).t().contiguous()
    return edge_index

def _coo_to_sparse(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    rtn = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return rtn

def _check_prune_non_pragma_nodes(g):
    if FLAGS.only_pragma:
        to_remove = []
        for node, ndata in g.nodes(data=True):
            x = ndata.get('full_text')
            if x is None:
                x = ndata['type']
            if type(x) is not str or (not 'Pragma' in x and (not 'pragma' in x)):
                to_remove.append(node)
        before = g.number_of_nodes()
        g.remove_nodes_from(to_remove)
        saver.log_info(f'Removed {len(to_remove)} non-pragma nodes from G -- {before} to {g.number_of_nodes()}')
        assert g.number_of_nodes() + len(to_remove) == before
    return g