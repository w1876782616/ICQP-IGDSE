import os
import sys
import time

_src_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_src_dir)
for _path in (_root_dir, _src_dir):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import FLAGS
from train import train_main, inference
from dse import EAExplorer, SAExplorer, ExhaustiveExplorer, ACOExplorer, CausalHybridExplorer, NSGAIIExplorer, LatticeExplorer, MOEDAExplorer
from saver import saver
import os.path as osp
from utils import get_root_path
from os.path import join
import config
from programl_data import get_data_list
from dse_database.gen_dataset import MyOwnDataset
TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL
if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'COLORS-3')
    if not FLAGS.force_regen or FLAGS.subtask == 'dse':
        dataset = MyOwnDataset()
    else:
        pragma_dim = 0
        dataset, pragma_dim = get_data_list()
    if FLAGS.subtask == 'inference':
        inference(dataset)
    elif FLAGS.subtask == 'dse':
        path_1 = join(get_root_path(), 'dse_database', 'machsuite', 'config')
        path_graph_1 = join(get_root_path(), 'dse_database', 'programl', 'machsuite', 'processed')
        path_2 = join(get_root_path(), 'dse_database', 'poly', 'config')
        path_graph_2 = join(get_root_path(), 'dse_database', 'programl', 'poly', 'processed')
        KERNELS = ['correlation']
        st = time.time()
        for kernel in KERNELS:
            saver.info('#################################################################')
            saver.info(f'Starting DSE for {kernel}')
            if kernel in MACHSUITE_KERNEL:
                path = path_1
                path_graph = path_graph_1
            else:
                path = path_2
                path_graph = path_graph_2
            if FLAGS.explorer == 'CausalHybrid':
                CausalHybridExplorer(path, kernel, path_graph, run_dse=True)
            elif FLAGS.explorer == 'EA':
                EAExplorer(path, kernel, path_graph, run_dse=True)
            elif FLAGS.explorer == 'SA':
                SAExplorer(path, kernel, path_graph, run_dse=True)
            elif FLAGS.explorer == 'Exhastive':
                ExhaustiveExplorer(path, kernel, path_graph, run_dse=True)
            elif FLAGS.explorer == 'NSGAII':
                NSGAIIExplorer(path, kernel, path_graph, run_dse=True)
            elif FLAGS.explorer == 'Lattice':
                LatticeExplorer(path, kernel, path_graph, run_dse=True)
            elif FLAGS.explorer == 'MOEDA':
                MOEDAExplorer(path, kernel, path_graph, run_dse=True)
            else:
                ACOExplorer(path, kernel, path_graph, run_dse=True)
            saver.info('#################################################################')
            saver.info(f'')
            et = time.time()
            print(f'All DSE time: {et - st}s')
    else:
        train_main(dataset, 0)
    saver.close()