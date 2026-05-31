import os.path as osp
from os.path import join, basename
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dse import GNNModel
import src.config as config
from src.config import FLAGS
from glob import glob, iglob
from src.utils import get_root_path, MLP, print_stats, get_save_path, create_dir_if_not_exists, plot_dist, load, save
from collections import Counter, OrderedDict
import pickle
from sklearn.preprocessing import OneHotEncoder
import warnings
from tqdm import tqdm
from src.saver import saver
import networkx as nx
from src.programl_data import _encode_X_dict, _encode_edge_dict, _encode_X_torch, _encode_edge_torch, finte_diff_as_quality, create_edge_index
import math
import torch
import numpy as np
from shutil import rmtree
from torch_geometric.data import Data
from copy import deepcopy
from torch_geometric.data import Dataset
import pickle
from pathlib import Path
tag = 'new_speedup'
TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen', 'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d', 'correlation', 'covariance', 'syrk']
code_path = join(get_root_path(), 'two_tower_dataset/code')
graph_path = join(get_root_path(), 'two_tower_dataset/graph')
points_path = join(get_root_path(), 'two_tower_dataset/points')
ENCODER_PATH = join(get_root_path(), 'save_models_and_data')
db_path = []
for benchmark in FLAGS.benchmarks:
    db_path.append(f'../dse_database/{benchmark}/databases/**/*')
GEXF_FOLDER = join(get_root_path(), 'dse_database', 'programl', '**', 'processed', '**')
GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf')])
if __name__ == '__main__':
    database = {}
    print(graph_path)
    bench = ['machsuite', 'poly']
    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')
    ntypes = Counter()
    ptypes = Counter()
    numerics = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()
    encoders_loaded = False
    if FLAGS.encoder_path != None:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                encoders = load(FLAGS.encoder_path)

            def _fix_encoder_compat(encoder):
                if not hasattr(encoder, '_infrequent_enabled'):
                    try:
                        encoder._infrequent_enabled = False
                    except:
                        pass
                if not hasattr(encoder, '_infrequent_indices'):
                    try:
                        object.__setattr__(encoder, '_infrequent_indices', None)
                    except:
                        pass
                if not hasattr(encoder, 'infrequent_categories_'):
                    try:
                        object.__setattr__(encoder, 'infrequent_categories_', [])
                    except:
                        try:
                            encoder.__dict__['infrequent_categories_'] = []
                        except:
                            pass
                return encoder
            enc_ntype = _fix_encoder_compat(encoders['enc_ntype'])
            enc_ptype = _fix_encoder_compat(encoders['enc_ptype'])
            enc_itype = _fix_encoder_compat(encoders['enc_itype'])
            enc_ftype = _fix_encoder_compat(encoders['enc_ftype'])
            enc_btype = _fix_encoder_compat(encoders['enc_btype'])
            enc_ftype_edge = _fix_encoder_compat(encoders['enc_ftype_edge'])
            enc_ptype_edge = _fix_encoder_compat(encoders['enc_ptype_edge'])
            encoders_loaded = True
            saver.log_info(f'Successfully loaded encoders from {FLAGS.encoder_path}')
        except Exception as e:
            saver.warning(f'Failed to load/fix encoders from {FLAGS.encoder_path}: {e}')
            saver.warning('Falling back to creating new encoders...')
            encoders_loaded = False
    if not encoders_loaded:
        enc_ntype = OneHotEncoder(handle_unknown='ignore')
        enc_ptype = OneHotEncoder(handle_unknown='ignore')
        enc_itype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype = OneHotEncoder(handle_unknown='ignore')
        enc_btype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
        enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')
        saver.log_info('Created new OneHot encoders')
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
    skipped_kernels = []
    for gexf_file in tqdm(GEXF_FILES[0:]):
        if FLAGS.dataset == 'machsuite' or 'programl' in FLAGS.dataset:
            proceed = False
            matched_kernel = None
            for k in ALL_KERNEL:
                if k in gexf_file:
                    proceed = True
                    matched_kernel = k
                    break
            if not proceed:
                skipped_kernels.append(gexf_file)
                saver.log_info(f'[SKIP] Skipped gexf file (not in ALL_KERNEL): {basename(gexf_file)}')
                continue
        else:
            raise NotImplementedError()
        g = nx.read_gexf(gexf_file)
        g.variants = OrderedDict()
        gname = basename(gexf_file).split('.')[0]
        saver.log_info(gname)
        n = basename(gexf_file).split('_')[0]
        all_gs[n] = g
        if FLAGS.dataset == 'programl':
            db_paths = []
            saver.log_info(f'Searching for .db files for kernel: {n}')
            for db_p in db_path:
                saver.log_info(f'  Searching in pattern: {db_p}')
                paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f]
                saver.log_info(f'  Found {len(paths)} matching files in this pattern')
                db_paths.extend(paths)
            if len(db_paths) == 0:
                saver.warning(f'No database found for {n} (kernel name: "{n}"). Skipping this kernel.')
                saver.warning(f'  Searched in patterns: {db_path}')
                saver.warning(f'  All .gexf files found: {len(GEXF_FILES)}')
                saver.warning(f'  Current gexf file: {gexf_file}')
                continue
        else:
            raise NotImplementedError()
        database.clear()
        saver.log_info(f'db_paths for {n}: ({len(db_paths)} files found)')
        for d in db_paths:
            saver.log_info(f'  {d}')
        assert len(db_paths) >= 1
        for idx, file in enumerate(db_paths):
            with open(file, 'rb') as f_db:
                data = pickle.load(f_db)
            if not isinstance(data, dict):
                raise ValueError(f'Unexpected data format in {file}: {type(data)} (expected dict)')
            for k, v in data.items():
                if isinstance(k, bytes):
                    k_str = k.decode('utf-8')
                else:
                    k_str = str(k)
                database[k_str] = v
            max_idx = idx + 1
        keys = list(database.keys())
        lv2_keys = [k for k in keys if 'lv2' in k]
        saver.log_info(f'num keys for {n}: {len(keys)} and lv2 keys: {len(lv2_keys)}')
        got_reference = False
        res_reference = 0
        max_perf = 0
        for key in sorted(keys):
            pickle_obj = database[key]
            if isinstance(pickle_obj, bytes):
                obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
            else:
                obj = pickle_obj
            if type(obj) is int or type(obj) is dict:
                continue
            if key[0:3] == 'lv1':
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
            pickle_obj = database[key]
            if isinstance(pickle_obj, bytes):
                obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
            else:
                obj = pickle_obj
            if type(obj) is int or type(obj) is dict:
                continue
            if FLAGS.task == 'regression' and key[0:3] == 'lv1':
                continue
            print(obj.point)
            xy_dict = _encode_X_dict(g, ntypes=ntypes, ptypes=ptypes, itypes=itypes, ftypes=ftypes, btypes=btypes, numerics=numerics, obj=obj)
            edge_dict = _encode_edge_dict(g, ftypes=ftypes_edge, ptypes=ptypes_edge)
            if FLAGS.task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        if obj.perf > 0:
                            y = FLAGS.normalizer / obj.perf
                            y = math.log(y + 1, 100)
                        else:
                            y = obj.perf
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                    elif tname == 'quality':
                        y = finte_diff_as_quality(obj, res_reference)
                    elif 'util' in tname or 'total' in tname:
                        y = math.log(obj.res_util[tname] + 1, 100)
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            else:
                raise NotImplementedError()
            vname = key
            g.variants[vname] = (xy_dict, edge_dict, obj.point)
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
    saver.log_info(f"\n{'=' * 60}")
    saver.log_info(f'[SUMMARY] Processing Summary:')
    saver.log_info(f'  Total .gexf files found: {len(GEXF_FILES)}')
    saver.log_info(f'  Kernels successfully processed: {num_files}')
    saver.log_info(f'  Kernels skipped (not in ALL_KERNEL): {len(skipped_kernels)}')
    if skipped_kernels:
        saver.log_info(f'  Skipped files:')
        for sf in skipped_kernels:
            saver.log_info(f'    - {basename(sf)}')
    saver.log_info(f'  Expected kernels (MACHSUITE): {len(MACHSUITE_KERNEL)} - {MACHSUITE_KERNEL}')
    saver.log_info(f'  Expected kernels (POLY): {len(poly_KERNEL)} - {poly_KERNEL}')
    saver.log_info(f'  Total expected: {len(ALL_KERNEL)}')
    saver.log_info(f"{'=' * 60}\n")
    if not encoders_loaded:
        saver.log_info('Fitting OneHot encoders with collected features...')
        if len(X_ntype_all) > 0:
            enc_ntype.fit(np.array(X_ntype_all).reshape(-1, 1))
        if len(X_ptype_all) > 0:
            enc_ptype.fit(np.array(X_ptype_all).reshape(-1, 1))
        if len(X_itype_all) > 0:
            enc_itype.fit(np.array(X_itype_all).reshape(-1, 1))
        if len(X_ftype_all) > 0:
            enc_ftype.fit(np.array(X_ftype_all).reshape(-1, 1))
        if len(X_btype_all) > 0:
            enc_btype.fit(np.array(X_btype_all).reshape(-1, 1))
        if len(edge_ftype_all) > 0:
            enc_ftype_edge.fit(np.array(edge_ftype_all).reshape(-1, 1))
        if len(edge_ptype_all) > 0:
            enc_ptype_edge.fit(np.array(edge_ptype_all).reshape(-1, 1))
        saver.log_info('OneHot encoders fitted successfully')
        try:
            encoder_obj = {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype, 'enc_itype': enc_itype, 'enc_ftype': enc_ftype, 'enc_btype': enc_btype, 'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge}
            encoder_file = join(ENCODER_PATH, 'encoders')
            save(encoder_obj, encoder_file)
            saver.log_info(f'✓ Successfully saved fitted encoders to {encoder_file}.klepto')
        except Exception as e:
            saver.error(f'✗ Failed to save encoders: {e}')
            import traceback
            saver.error(traceback.format_exc())
            raise
        try:
            saver.log_info(f'PyTorch version: {torch.__version__}')
            test_tensor = torch.tensor([1.0])
            saver.log_info('PyTorch tensor creation test passed:', test_tensor)
            if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available'):
                cuda_available = torch.cuda.is_available()
                saver.log_info(f'CUDA available: {cuda_available}')
            else:
                saver.log_info('CUDA check not available')
        except Exception as e:
            saver.error(f'PyTorch test failed: {e}')
            if 'torch' in sys.modules:
                saver.error(f"PyTorch module: {sys.modules['torch']}")
            raise
        import importlib
        if 'torch' not in sys.modules:
            import torch
            saver.log_info('Re-imported torch before loading model')
        else:
            torch = importlib.reload(sys.modules['torch'])
            saver.log_info(f'Reloaded torch module: {torch.__version__}')
        try:
            _ = torch.tensor([1.0])
            saver.log_info('PyTorch verification before model loading: OK')
        except Exception as e:
            saver.error(f'PyTorch verification failed before model loading: {e}')
            raise
        try:
            if not hasattr(torch, 'compiler'):

                class _CompilerModule:

                    @staticmethod
                    def disable(recursive=False):

                        def decorator(func):
                            return func
                        return decorator
                torch.compiler = _CompilerModule()
                saver.log_info('✓ Patched torch.compiler for compatibility')
            else:
                saver.log_info('✓ torch.compiler already exists')
        except Exception as e:
            saver.warning(f'Could not patch torch.compiler: {e}')
        try:
            if not hasattr(torch, 'float8_e4m3fn'):
                torch.float8_e4m3fn = torch.float16
                saver.log_info('✓ Patched torch.float8_e4m3fn for compatibility')
            if not hasattr(torch, 'float8_e4m3fnuz'):
                torch.float8_e4m3fnuz = torch.float16
                saver.log_info('✓ Patched torch.float8_e4m3fnuz for compatibility')
            if not hasattr(torch, 'float8_e5m2'):
                torch.float8_e5m2 = torch.float16
                saver.log_info('✓ Patched torch.float8_e5m2 for compatibility')
            if not hasattr(torch, 'float8_e5m2fnuz'):
                torch.float8_e5m2fnuz = torch.float16
                saver.log_info('✓ Patched torch.float8_e5m2fnuz for compatibility')
        except Exception as e:
            saver.warning(f'Could not patch torch float8 dtypes: {e}')
        try:
            if not hasattr(torch, '_dynamo'):

                class _DynamoModule:

                    @staticmethod
                    def allow_in_graph(fn=None, **kwargs):
                        if fn is None:

                            def decorator(func):
                                return func
                            return decorator
                        return fn
                torch._dynamo = _DynamoModule()
                saver.log_info('✓ Patched torch._dynamo for compatibility')
            else:
                saver.log_info('✓ torch._dynamo already exists')
        except Exception as e:
            saver.warning(f'Could not patch torch._dynamo: {e}')
        try:
            if hasattr(torch, 'contiguous_format'):
                if torch.contiguous_format is not None:
                    torch.contiguous_format = None
                    saver.log_info('✓ Patched torch.contiguous_format to None for compatibility')
        except Exception as e:
            saver.warning(f'Could not patch torch.contiguous_format: {e}')
        try:
            _orig_empty = torch.empty

            def _strip_memory_format_args(args, kwargs):
                if 'memory_format' in kwargs:
                    kwargs.pop('memory_format', None)
                elif len(args) >= 6:
                    args = list(args)
                    args[5] = None
                    args = tuple(args)
                return (args, kwargs)

            def _empty_compat(*args, **kwargs):
                args, kwargs = _strip_memory_format_args(args, kwargs)
                kwargs['memory_format'] = None
                return _orig_empty(*args, **kwargs)
            torch.empty = _empty_compat
            saver.log_info('✓ Patched torch.empty for memory_format compatibility')
        except Exception as e:
            saver.warning(f'Could not patch torch.empty: {e}')
        try:
            import torch._refs as torch_refs
            if hasattr(torch_refs, 'empty'):
                _orig_refs_empty = torch_refs.empty

                def _refs_empty_compat(*args, **kwargs):
                    args, kwargs = _strip_memory_format_args(args, kwargs)
                    kwargs['memory_format'] = None
                    return _orig_refs_empty(*args, **kwargs)
                torch_refs.empty = _refs_empty_compat
                saver.log_info('✓ Patched torch._refs.empty for memory_format compatibility')
        except Exception as e:
            saver.warning(f'Could not patch torch._refs.empty: {e}')
        os.environ['TORCH_AVAILABLE'] = '1'
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
            os.environ['CUDA_AVAILABLE'] = '1'
        try:
            import torch.utils._pytree as pytree_module
            saver.log_info(f'Checking torch.utils._pytree for register_pytree_node...')
            if not hasattr(pytree_module, 'register_pytree_node'):
                _registered_pytree_nodes = {}

                def register_pytree_node(cls, flatten_fn, unflatten_fn, **kwargs):
                    _registered_pytree_nodes[cls] = (flatten_fn, unflatten_fn)
                    if flatten_fn is not None:
                        pytree_module.__dict__['_model_output_flatten'] = flatten_fn
                    if unflatten_fn is not None:
                        pytree_module.__dict__['_model_output_unflatten'] = unflatten_fn
                pytree_module.register_pytree_node = register_pytree_node
                pytree_module._registered_pytree_nodes = _registered_pytree_nodes

                def _default_flatten(output):
                    if hasattr(output, '__dict__'):
                        values = []
                        for key, value in output.__dict__.items():
                            if not key.startswith('_'):
                                values.append(value)
                        return (values, None)
                    return ([], None)

                def _default_unflatten(values, context):
                    return values[0] if values else None
                pytree_module.__dict__['_model_output_flatten'] = _default_flatten
                pytree_module.__dict__['_model_output_unflatten'] = _default_unflatten
                saver.log_info('✓ Patched torch.utils._pytree.register_pytree_node for compatibility')
            else:
                saver.log_info('✓ torch.utils._pytree.register_pytree_node already exists')
        except Exception as e:
            saver.warning(f'Could not patch torch.utils._pytree: {e}')
            import traceback
            saver.warning(traceback.format_exc())
        try:
            import torch.nn as nn
            import inspect
            _orig_load_state_dict_unbound = nn.Module.load_state_dict.__func__ if hasattr(nn.Module.load_state_dict, '__func__') else nn.Module.load_state_dict

            def _patched_load_state_dict(self, state_dict, strict=True, assign=None):
                if assign:
                    for name, param in self.named_parameters():
                        if name in state_dict:
                            param.data = state_dict[name].to(param.device).to(param.dtype)
                    for name, buffer in self.named_buffers():
                        if name in state_dict:
                            buffer.data = state_dict[name].to(buffer.device).to(buffer.dtype)
                    return None
                else:
                    try:
                        sig = inspect.signature(_orig_load_state_dict_unbound)
                        if 'strict' in sig.parameters:
                            return _orig_load_state_dict_unbound(self, state_dict, strict=strict)
                        else:
                            return _orig_load_state_dict_unbound(self, state_dict)
                    except Exception as e:
                        missing_keys = []
                        unexpected_keys = []
                        for name, param in self.named_parameters():
                            if name in state_dict:
                                param.data = state_dict[name].to(param.device).to(param.dtype)
                            else:
                                missing_keys.append(name)
                        for name in state_dict:
                            if name not in [n for n, _ in self.named_parameters()] and name not in [n for n, _ in self.named_buffers()]:
                                unexpected_keys.append(name)
                        if strict and (missing_keys or unexpected_keys):
                            error_msg = f'Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}'
                            raise RuntimeError(error_msg)
                        return None
            nn.Module.load_state_dict = _patched_load_state_dict
            saver.log_info("✓ Patched torch.nn.Module.load_state_dict to accept 'assign' parameter")
        except Exception as e:
            saver.warning(f'Could not patch Module.load_state_dict: {e}')
            import traceback
            saver.warning(traceback.format_exc())
        try:
            import transformers
            saver.log_info('Successfully imported transformers')
            saver.log_info(f'Transformers version: {transformers.__version__}')
        except ImportError as e:
            saver.error(f'Failed to import transformers: {e}')
            raise
        try:
            import transformers.utils.import_utils as import_utils
            if hasattr(import_utils, '_torch_available'):
                import_utils._torch_available = True
                saver.log_info('Set transformers.utils.import_utils._torch_available = True')
            if hasattr(import_utils, '_backends'):
                if 'torch' not in import_utils._backends or import_utils._backends['torch'] is None:
                    import_utils._backends['torch'] = torch
                    saver.log_info('Added/Updated torch in transformers._backends')
                else:
                    saver.log_info('torch already in transformers._backends')
            if 'torch' not in sys.modules or sys.modules['torch'] is None:
                sys.modules['torch'] = torch
                saver.log_info("Updated sys.modules['torch']")
            if hasattr(transformers.utils, 'is_torch_available'):
                torch_available = transformers.utils.is_torch_available()
                saver.log_info(f'Transformers detects PyTorch (after fix): {torch_available}')
                if not torch_available:
                    saver.warning('Transformers still does not detect PyTorch after fix attempts')
            else:
                saver.warning('Transformers utils does not have is_torch_available method')
            try:
                pass
            except:
                pass
            try:
                import transformers.utils.generic as generic_module
                if not hasattr(generic_module, '_model_output_flatten'):

                    def _model_output_flatten(output):
                        if hasattr(output, '__dict__'):
                            values = []
                            for key, value in output.__dict__.items():
                                if not key.startswith('_'):
                                    values.append(value)
                            return (values, None)
                        return ([], None)

                    def _model_output_unflatten(values, context):
                        return values[0] if values else None
                    generic_module.__dict__['_model_output_flatten'] = _model_output_flatten
                    generic_module.__dict__['_model_output_unflatten'] = _model_output_unflatten
                    saver.log_info('✓ Patched transformers.utils.generic with _model_output_flatten/unflatten')
                else:
                    saver.log_info('✓ transformers.utils.generic already has _model_output_flatten')
            except Exception as e:
                saver.warning(f'Could not patch transformers.utils.generic: {e}')
                import traceback
                saver.warning(traceback.format_exc())
            saver.log_info('✓ Skipping transformers.modeling_utils.load_state_dict patch (not needed for transformers 4.35.2)')
            try:
                from transformers import AutoModelForMaskedLM, AutoTokenizer
                saver.log_info(f'Imported AutoModelForMaskedLM: {type(AutoModelForMaskedLM)}')
                if 'DummyObject' in str(type(AutoModelForMaskedLM)):
                    saver.warning('AutoModelForMaskedLM is still a DummyObject, attempting manual fix...')
                    try:
                        import transformers.models.auto.modeling_auto as modeling_auto_mod
                        if hasattr(modeling_auto_mod, 'AutoModelForMaskedLM'):
                            real_class = modeling_auto_mod.AutoModelForMaskedLM
                            if 'DummyObject' not in str(type(real_class)):
                                AutoModelForMaskedLM = real_class
                                saver.log_info(f'Got real AutoModelForMaskedLM from modeling_auto: {type(AutoModelForMaskedLM)}')
                            else:
                                raise ImportError('Real class is still DummyObject')
                        else:
                            raise ImportError('modeling_auto module does not have AutoModelForMaskedLM')
                    except Exception as e2:
                        saver.warning(f'Method 1 failed: {e2}, trying method 2...')
                        try:
                            from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
                            AutoModelForMaskedLM = RobertaForMaskedLM
                            saver.log_info(f'Using RobertaForMaskedLM as fallback: {type(AutoModelForMaskedLM)}')
                        except Exception as e3:
                            saver.error(f'All manual fix methods failed. Method 1: {e2}, Method 2: {e3}')
                            raise ImportError(f'AutoModelForMaskedLM is DummyObject and all fix methods failed')
                else:
                    saver.log_info('AutoModelForMaskedLM is not a DummyObject, import successful')
            except Exception as e:
                saver.error(f'Failed to import AutoModelForMaskedLM: {e}')
                import traceback
                saver.error(traceback.format_exc())
                raise
        except Exception as e:
            saver.warning(f'Could not fix transformers PyTorch detection: {e}')
            import traceback
            saver.warning(traceback.format_exc())
            try:
                from transformers import AutoModelForMaskedLM, AutoTokenizer
                saver.log_info('Imported AutoModelForMaskedLM and AutoTokenizer (fallback)')
            except ImportError as e2:
                saver.error(f'Failed to import AutoModelForMaskedLM: {e2}')
                raise
        default_codebert_dir = join(get_root_path(), 'codebert')
        CODEBERT_DIR = os.getenv('CODEBERT_DIR', default_codebert_dir)
        CODEBERT_MODEL_ID = os.getenv('CODEBERT_MODEL_ID', 'microsoft/codebert-base')
        saver.log_info(f'CodeBERT directory: {CODEBERT_DIR}')
        saver.log_info(f'CodeBERT model ID: {CODEBERT_MODEL_ID}')

        def _find_hf_model_path(base_dir: str) -> str:
            base_path = Path(base_dir)
            if base_path.exists():
                has_model = (base_path / 'pytorch_model.bin').exists() or (base_path / 'model.safetensors').exists()
                has_config = (base_path / 'config.json').exists()
                if has_model and has_config:
                    saver.log_info(f'Found complete model in root directory: {base_path}')
                    return str(base_path)
            valid_paths = []
            for model_dir in base_path.glob('models--*/snapshots/*'):
                if model_dir.is_dir():
                    has_model = (model_dir / 'pytorch_model.bin').exists() or (model_dir / 'model.safetensors').exists()
                    has_config = (model_dir / 'config.json').exists()
                    if has_model and has_config:
                        valid_paths.append(model_dir)
                        saver.log_info(f'Found valid model path: {model_dir}')
                    elif has_model:
                        saver.warning(f'Found model weights in {model_dir} but missing config.json, skipping')
            if valid_paths:
                if len(valid_paths) > 1:
                    saver.warning(f'Found {len(valid_paths)} valid model paths, selecting the most recent one')
                    valid_paths.sort(key=lambda p: (p / 'config.json').stat().st_mtime, reverse=True)
                selected_path = valid_paths[0]
                saver.log_info(f'Selected model path: {selected_path}')
                return str(selected_path)
            saver.warning(f'No valid model path found in {base_dir} (need both config.json and model weights)')
            return None
        if AutoModelForMaskedLM is None:
            saver.error('AutoModelForMaskedLM is None after import!')
            raise ImportError('AutoModelForMaskedLM is None')
        saver.log_info(f'AutoModelForMaskedLM type: {type(AutoModelForMaskedLM)}')
        model_path = _find_hf_model_path(CODEBERT_DIR)
        use_fallback = 'DummyObject' in str(type(AutoModelForMaskedLM))
        if use_fallback:
            saver.warning('AutoModelForMaskedLM is DummyObject, using RobertaForMaskedLM as fallback')
            try:
                from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
                ModelClass = RobertaForMaskedLM
            except ImportError as e:
                saver.error(f'Failed to import RobertaForMaskedLM: {e}')
                raise
        else:
            ModelClass = AutoModelForMaskedLM
        if ModelClass is None:
            saver.error('ModelClass is None!')
            raise ValueError('ModelClass is None')
        if not hasattr(ModelClass, 'from_pretrained'):
            saver.error(f'ModelClass {ModelClass} does not have from_pretrained method!')
            raise AttributeError(f'ModelClass {ModelClass} does not have from_pretrained method')
        saver.log_info(f"Using ModelClass: {ModelClass}, has from_pretrained: {hasattr(ModelClass, 'from_pretrained')}")
        model_loaded = False
        if model_path:
            try:
                saver.log_info(f'Loading CodeBERT from local path: {model_path}')
                model = ModelClass.from_pretrained(model_path, low_cpu_mem_usage=False).to(FLAGS.device)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                saver.log_info(f'Loaded CodeBERT from local dir: {model_path}')
                model_loaded = True
            except Exception as e:
                saver.warning(f'Failed to load model from local path {model_path}: {e}')
                saver.warning('Falling back to using model ID (will download if needed)...')
                model_path = None
        if not model_loaded:
            if model_path is None:
                saver.warning(f"CodeBERT weights not found under {CODEBERT_DIR} or loading failed. Falling back to model id '{CODEBERT_MODEL_ID}' (will download to cache_dir).")
            try:
                saver.log_info(f'Loading CodeBERT from model ID: {CODEBERT_MODEL_ID}')
                model = ModelClass.from_pretrained(CODEBERT_MODEL_ID, cache_dir=CODEBERT_DIR, low_cpu_mem_usage=False).to(FLAGS.device)
                tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL_ID, cache_dir=CODEBERT_DIR)
                saver.log_info(f'Downloaded and loaded CodeBERT model: {CODEBERT_MODEL_ID}')
            except Exception as e:
                saver.error(f'Failed to load model from HuggingFace: {e}')
                saver.error('This might be a transformers/PyTorch compatibility issue')
                import traceback
                saver.error(traceback.format_exc())
                raise
        if not all_gs:
            saver.error('No graphs found in all_gs! Cannot generate dataset.')
            raise ValueError('all_gs is empty, cannot proceed with dataset generation')
        saver.log_info(f'Generating dataset for {len(all_gs)} kernels')
        for gname, g in all_gs.items():
            edge_index = create_edge_index(g)
            saver.log_info('edge_index created', gname)
            data_list = []
            if gname in MACHSUITE_KERNEL:
                SAVE_DIR = join(graph_path, 'machsuite', gname)
                SAVE_DIR_1 = join(code_path, 'machsuite', gname)
                CODE_FILES = join(get_root_path(), 'dse_database', 'programl', 'machsuite', gname, gname + '.c')
            else:
                SAVE_DIR = join(graph_path, 'poly', gname)
                SAVE_DIR_1 = join(code_path, 'poly', gname)
                CODE_FILES = join(get_root_path(), 'dse_database', 'programl', 'poly', gname, gname + '.c')
            code_list = []
            try:
                with open(CODE_FILES, 'r', encoding='utf-8') as file:
                    fc = file.read()
            except FileNotFoundError:
                saver.error(f'Source code file not found: {CODE_FILES}')
                raise
            except Exception as e:
                saver.error(f'Failed to read source code file {CODE_FILES}: {e}')
                raise
            create_dir_if_not_exists(SAVE_DIR_1)
            if not g.variants:
                saver.warning(f'No variants found for kernel {gname}, skipping...')
                continue
            points_list = []
            points_collected = 0
            points_none_count = 0
            for nums, (vname, d) in enumerate(g.variants.items()):
                d_node, d_edge, point = d
                points_list.append(point)
                points_collected += 1
                if point is None:
                    points_none_count += 1
                elif nums < 3:
                    saver.log_info(f"[Debug] Collected point for {gname} variant {vname}: {type(point)}, keys: {(list(point.keys()) if isinstance(point, dict) else 'N/A')}")
            saver.log_info(f'[Debug] Collected {points_collected} points for kernel {gname} (None: {points_none_count}, non-None: {points_collected - points_none_count})')
            for nums, (vname, d) in enumerate(g.variants.items()):
                d_node, d_edge, point = d
                X = _encode_X_torch(d_node, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
                edge_attr = _encode_edge_torch(d_edge, enc_ftype_edge, enc_ptype_edge)
                if FLAGS.task == 'regression':
                    data_list.append(Data(x=X, edge_index=edge_index, perf=d_node['perf'], actual_perf=d_node['actual_perf'], quality=d_node['quality'], util_BRAM=d_node['util-BRAM'], util_DSP=d_node['util-DSP'], util_LUT=d_node['util-LUT'], util_FF=d_node['util-FF'], total_BRAM=d_node['total-BRAM'], total_DSP=d_node['total-DSP'], total_LUT=d_node['total-LUT'], total_FF=d_node['total-FF'], edge_attr=edge_attr, kernel=gname))
                else:
                    raise NotImplementedError()
                fcc = deepcopy(fc)
                kd = [i for i in vname[4:].split('.')]
                kd_d = {}
                for j in kd:
                    jl = j.split('-')
                    if jl[1] == 'NA':
                        kd_d[jl[0]] = ''
                    else:
                        kd_d[jl[0]] = jl[1]
                for k, v in kd_d.items():
                    fcc = fcc.replace('auto' + '{' + k + '}', v)
                try:
                    code_inputs = tokenizer(fcc, padding=True, truncation=True, max_length=512, return_tensors='pt').to(FLAGS.device)
                    model_device = next(model.parameters()).device
                    if str(model_device) != str(FLAGS.device):
                        saver.warning(f'Model device ({model_device}) != FLAGS.device ({FLAGS.device}), moving inputs to model device')
                    code_inputs = {k: v.to(model_device) for k, v in code_inputs.items()}
                    with torch.no_grad():
                        code_outputs = model(**code_inputs, output_hidden_states=True)
                    features = []
                    if hasattr(code_outputs, 'hidden_states') and code_outputs.hidden_states is not None:
                        for i, hidden_states in enumerate(code_outputs.hidden_states):
                            features.append(hidden_states[:, 0, :])
                        code_features = torch.stack(features).sum(dim=0) / len(features)
                    else:
                        saver.warning(f'No hidden_states in CodeBERT output for {gname} variant {vname}, using last_hidden_state')
                        code_features = code_outputs.last_hidden_state[:, 0, :]
                    pa = join(SAVE_DIR_1, f'{nums + 1}.pt')
                    torch.save(code_features.cpu(), pa)
                except Exception as e:
                    saver.error(f'Failed to encode code for {gname} variant {vname}: {e}')
                    import traceback
                    saver.error(traceback.format_exc())
                    saver.warning(f'Skipping code feature for {gname} variant {vname}, but continuing with graph data')
            try:
                saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
                if osp.exists(SAVE_DIR):
                    rmtree(SAVE_DIR)
                create_dir_if_not_exists(SAVE_DIR)
                for i in tqdm(range(len(data_list))):
                    torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i)))
            finally:
                try:
                    if gname in MACHSUITE_KERNEL:
                        benchmark = 'machsuite'
                    else:
                        benchmark = 'poly'
                    kernel_points_dir = join(points_path, benchmark, gname)
                    points_list_path = join(kernel_points_dir, 'points_list.pkl')
                    create_dir_if_not_exists(kernel_points_dir)
                    if len(points_list) != len(data_list):
                        saver.warning(f'⚠️  points_list length ({len(points_list)}) != data_list length ({len(data_list)}) for kernel {gname}')
                        saver.warning(f'⚠️  This may cause index mismatch. Truncating points_list to match data_list.')
                        points_list = points_list[:len(data_list)]
                    if len(points_list) == 0:
                        saver.warning(f'points_list is empty for kernel {gname}, skipping points_list.pkl save')
                    else:
                        with open(points_list_path, 'wb') as f:
                            pickle.dump(points_list, f)
                        if osp.exists(points_list_path):
                            file_size = osp.getsize(points_list_path)
                            non_none_count = sum((1 for p in points_list if p is not None))
                            saver.log_info(f'✓ Saved {len(points_list)} design points to {points_list_path} (non-None: {non_none_count}, file size: {file_size} bytes)')
                            saver.log_info(f'✓ points_list.pkl location: {osp.abspath(points_list_path)}')
                        else:
                            saver.error(f'✗ Failed to save points_list.pkl: file does not exist after save operation')
                            saver.error(f'✗ Expected path: {osp.abspath(points_list_path)}')
                except Exception as e:
                    saver.error(f'✗ Failed to save points_list.pkl for kernel {gname}: {e}')
                    import traceback
                    saver.error(traceback.format_exc())
            nns = [d.x.shape[0] for d in data_list]
            print_stats(nns, 'number of nodes')
            ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
            print_stats(ads, 'avg degrees')
            saver.log_info('dataset[0].num_features', data_list[0].num_features)
            for target in TARGETS:
                if not hasattr(data_list[0], target.replace('-', '_')):
                    saver.warning(f'Data does not have attribute {target}')
                    continue
                ys = [getattr(d, target.replace('-', '_')).item() for d in data_list]
                plot_dist(ys, f'{target}_ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
                saver.log_info(f'{target}_ys', Counter(ys))

class MyOwnDataset:

    def __init__(self):
        pass

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        gp = {}
        cp = {}
        for k in ALL_KERNEL:
            if k in MACHSUITE_KERNEL:
                gt = join(graph_path, 'machsuite', k)
                ct = join(code_path, 'machsuite', k)
            else:
                gt = join(graph_path, 'poly', k)
                ct = join(code_path, 'poly', k)
            gp[k] = glob(join(gt, '*.pt'))
            cp[k] = glob(join(ct, '*.pt'))
        return (gp, cp)

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        pass

    def __len__(self):
        return self.len()

    def get(self, idx):
        pass

    @staticmethod
    def get_data(idx, k):
        if k in MACHSUITE_KERNEL:
            gt = join(graph_path, 'machsuite', k)
            ct = join(code_path, 'machsuite', k)
        else:
            gt = join(graph_path, 'poly', k)
            ct = join(code_path, 'poly', k)
        data = torch.load(osp.join(gt, 'data_{}.pt'.format(idx)))
        code_d = torch.load(join(ct, f'{idx}.pt'))
        return (data, code_d)