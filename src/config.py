from networkx.classes import neighbors
import torch
from src.get_root_path import get_user, get_host
import argparse
import os
from os.path import join
from src.get_root_path import get_root_path
TARGETS = ['perf', 'quality', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF', 'total-BRAM', 'total-DSP', 'total-LUT', 'total-FF']
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen', 'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d', 'correlation', 'covariance', 'syrk']
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='simple')
dataset = 'programl'
parser.add_argument('--dataset', default=dataset)
benchmark = ['machsuite', 'poly']
parser.add_argument('--benchmarks', default=benchmark)
tag = 'whole-machsuite-poly'
parser.add_argument('--tag', default=tag)
encoder_path = join(get_root_path(), 'save_models_and_data/encoders.klepto')
parser.add_argument('--encoder_path', default=encoder_path)
model_path = join(get_root_path(), 'save_models_and_data/regression_model_state_dict.pth')
parser.add_argument('--model_path', default=model_path)
class_model_path = None
parser.add_argument('--class_model_path', default=class_model_path)
parser.add_argument('--num_features', default=158)
TASK = 'regression'
parser.add_argument('--task', default=TASK)
SUBTASK = 'inference'
parser.add_argument('--subtask', default=SUBTASK)
parser.add_argument('--val_ratio', type=float, default=0.15)
explorer = 'Exhastive'
parser.add_argument('--explorer', default=explorer)
model_tag = 'test'
parser.add_argument('--model_tag', default=model_tag)
parser.add_argument('--activation', default='elu')
parser.add_argument('--prune_util', default=True)
parser.add_argument('--prune_class', default=False)
parser.add_argument('--force_regen', type=bool, default=False)
parser.add_argument('--no_pragma', type=bool, default=False)
pids = ['__PARA__L3', '__PIPE__L2', '__PARA__L1', '__PIPE__L0', '__TILE__L2', '__TILE__L0', '__PARA__L2', '__PIPE__L0']
parser.add_argument('--ordered_pids', default=pids)
multi_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
parser.add_argument('--target', default=multi_target)
parser.add_argument('--separate_perf', type=bool, default=False)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--encode_edge', type=bool, default=False)
parser.add_argument('--loss', type=str, default='RMSE')
EPSILON = 0.001
parser.add_argument('--epsilon', default=EPSILON)
NORMALIZER = 10000000.0
parser.add_argument('--normalizer', default=NORMALIZER)
MAX_NUMBER = 10000000000.0
parser.add_argument('--max_number', default=MAX_NUMBER)
norm = 'speedup-log2'
parser.add_argument('--norm_method', default=norm)
parser.add_argument('--invalid', type=bool, default=False)
parser.add_argument('--all_kernels', type=bool, default=True)
parser.add_argument('--inference_use_all_data', type=bool, default=True, help='Use 100% unseen-kernel samples during inference evaluation.')
parser.add_argument('--multi_target', type=bool, default=True)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--encode_log', type=bool, default=False)
parser.add_argument('--D', type=int, default=64)
batch_size = 64
parser.add_argument('--batch_size', type=int, default=batch_size)
epoch_num = 100
parser.add_argument('--epoch_num', type=int, default=epoch_num)
gpu = 0
device = 'cuda:0'
parser.add_argument('--device', default=device)
parser.add_argument('--print_every_iter', type=int, default=100)
parser.add_argument('--plot_pred_points', type=bool, default=False)
best_result_path = '/best_result'
parser.add_argument('--best_result_path', type=str, default=best_result_path)
dse_unseen_kernel = ['bicg', 'doitgen', 'gesummv', '2mm']
parser.add_argument('--dse_unseen_kernel', type=list, default=dse_unseen_kernel)
out_dim = 1 if TASK == 'regression' else 2
parser.add_argument('--out_dim', type=int, default=out_dim)
parser.add_argument('--learn_temp', default=False)
parser.add_argument('--temp_model_type', dest='temp_model_type', default='LIN', type=str)
parser.add_argument('--tau0', default=0.5, type=float)
parser.add_argument('--temp', default=0.01, type=float)
parser.add_argument('--env_model_type', default='SUM_GNN', type=str)
parser.add_argument('--env_num_layers', default=3, type=int)
parser.add_argument('--env_dim', default=128, type=int)
parser.add_argument('--skip', default=False)
parser.add_argument('--batch_norm', default=False)
parser.add_argument('--layer_norm', default=False)
parser.add_argument('--dec_num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--act_model_type', default='MEAN_GNN', type=str)
parser.add_argument('--act_num_layers', default=2, type=int)
parser.add_argument('--act_dim', default=16, type=int)
crossover_mutation_rate = 0.1
parser.add_argument('--crossover_mutation_rate', default=crossover_mutation_rate, type=int)
iter_stop_num = 0.1
parser.add_argument('--iter_stop_num', default=iter_stop_num, type=int)
initial_temperature = 100
parser.add_argument('--initial_temperature', default=initial_temperature, type=int)
stop_temperature = 0.1
parser.add_argument('--stop_temperature', default=stop_temperature, type=int)
cooling_rate = 0.1
parser.add_argument('--cooling_rate', default=cooling_rate, type=int)
neighbor_distance_rate = 0.1
parser.add_argument('--neighbor_distance_rate', default=neighbor_distance_rate, type=int)
exhaustive_timeout = 0
parser.add_argument('--exhaustive_timeout', default=exhaustive_timeout, type=float, help='Timeout (seconds) for ExhaustiveExplorer; <=0 means no timeout limit.')
comparative_if = False
parser.add_argument('--comparative_if', default=comparative_if, type=bool)
comparative_model = 'gnn-dse'
parser.add_argument('--comparative_model', default=comparative_model, type=str)
parser.add_argument('--hidden_num', default=128, type=int)
target_1 = ['lut', 'ff', 'dsp', 'bram', 'uram', 'srl', 'cp', 'power']
parser.add_argument('--target_1', default=target_1)
dataset_seen = ['aes', 'bfs', 'fft', 'gemm', 'md', 'nw']
parser.add_argument('--dataset_seen', default=dataset_seen)
dataset_unseen = ['sort', 'spmv', 'stencil', 'vitberbi']
parser.add_argument('--dataset_unseen', default=dataset_unseen)
ablation_stu = 'CoGNN'
parser.add_argument('--ablation_stu', default=ablation_stu)
parser.add_argument('--use_causal', type=bool, default=True)
parser.add_argument('--causal_lambda', type=float, default=0.5)
parser.add_argument('--causal_reg_beta', type=float, default=0)
parser.add_argument('--causal_max_pairs_per_batch', type=int, default=100)
parser.add_argument('--causal_max_pairs_eval', type=int, default=0)
parser.add_argument('--causal_main_pred_mode', type=str, default='mlp', choices=['replace', 'mlp', 'fusion'])
parser.add_argument('--causal_fusion_w_max', type=float, default=0.2)
parser.add_argument('--causal_fusion_ramp_epochs', type=int, default=20)
parser.add_argument('--causal_fusion_schedule', type=str, default='linear', choices=['linear', 'exp'])
parser.add_argument('--causal_alpha_tau', type=float, default=0)
parser.add_argument('--causal_entropy_beta', type=float, default=0)

temperature = 1.0
parser.add_argument("--temperature", default=temperature, type=int)

crossover_mutation_ratio = 0.02
parser.add_argument("--crossover_mutation_ratio", default=crossover_mutation_ratio, type=int)
stop_iteration_ratio = 0.3
parser.add_argument("--stop_iteration_ratio", default=stop_iteration_ratio, type=int)
# EA parameter
crossover_mutation_rate = 0.1
parser.add_argument("--crossover_mutation_rate", default=crossover_mutation_rate, type=int)
iter_stop_num = 0.1
parser.add_argument("--iter_stop_num", default=iter_stop_num, type=int)

# SA parameter
initial_temperature = 100
parser.add_argument("--initial_temperature", default=initial_temperature, type=int)
stop_temperature = 0.1
parser.add_argument("--stop_temperature", default=stop_temperature, type=int)
cooling_rate = 0.1
parser.add_argument("--cooling_rate", default=cooling_rate, type=int)
neighbor_distance_rate = 0.1
parser.add_argument("--neighbor_distance_rate", default=neighbor_distance_rate, type=int)

# comparative experiment parameter
comparative_if = True
parser.add_argument("--comparative_if", default=comparative_if, type=bool)
# comparative_model = "HGP"
# comparative_model = "ironman"
comparative_model = "pna"
# comparative_model = "gnn-dse"
parser.add_argument("--comparative_model", default=comparative_model, type=str)
parser.add_argument("--hidden_num", default=128, type=int)

# HGP dataset comparative experiment parameter
target_1 = ['lut', 'ff', 'dsp', 'bram', 'uram', 'srl', 'cp', 'power']
parser.add_argument("--target_1", default=target_1)
dataset_seen = ['aes', 'bfs', 'fft', 'gemm', 'md', 'nw']
parser.add_argument("--dataset_seen", default=dataset_seen)
dataset_unseen = ['sort', 'spmv', 'stencil', 'vitberbi']
parser.add_argument("--dataset_unseen", default=dataset_unseen)

"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

parser.add_argument('--hostname', default=get_host())
FLAGS = parser.parse_args()
