import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
import torch.nn as nn
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs
from CoGNN.action_gumbel_layer import TempSoftPlus, ActionNet
from config import FLAGS
from src.utils import MLP, _get_y_with_target
from collections import OrderedDict, defaultdict
from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
from typing import NamedTuple, Any, Callable
from comp_model import GNN_DSE, HGP, Ironman, pna, GNN_DSE
from torch_geometric.utils import degree
from src.causal_model import PragmaEncoder, CausalHead

def gin_mlp_func() -> Callable:

    def mlp_func(in_channels: int, out_channels: int, bias: bool):
        return Sequential(Linear(in_channels, out_channels, bias=bias), ReLU(), Linear(out_channels, out_channels, bias=bias))
    return mlp_func
gin_mlp_func = gin_mlp_func()

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

class CoGNN(Module):

    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActionNetArgs):
        super(CoGNN, self).__init__()
        self.task = FLAGS.task
        self.target = FLAGS.target
        self.D = FLAGS.D
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        self.first_MLP_env_attr = MLP(7, env_args.env_dim, activation_type=FLAGS.activation)
        self.first_MLP_act_attr = MLP(7, action_args.hidden_dim, activation_type=FLAGS.activation)
        self.first_MLP_node = MLP(153, env_args.env_dim, activation_type=FLAGS.activation)
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.env_dim)
        self.temp = gumbel_args.temp
        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()
        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.drop_ratio = env_args.dropout
        self.act = env_args.act_type.get()
        self.in_act_net = ActionNet(action_args=action_args)
        self.out_act_net = ActionNet(action_args=action_args)
        self.gate_nn = nn.Sequential(nn.Linear(self.D, self.D), ReLU(), Linear(self.D, self.D))
        self.glob = MyGlobalAttention(self.gate_nn, None)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (data.x, data.edge_index, data.edge_attr, data.batch)
        if hasattr(data, 'kernel'):
            gname = data.kernel[0]
        env_edge_attr = self.first_MLP_env_attr(edge_attr)
        act_edge_attr = self.first_MLP_act_attr(edge_attr)
        x = self.first_MLP_node(x)
        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)
            in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_attr, act_edge_attr=act_edge_attr)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_attr, act_edge_attr=act_edge_attr)
            temp = self.temp_model(x=x, edge_index=edge_index, edge_attr=env_edge_attr) if self.learn_temp else self.temp
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(edge_index=edge_index, keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
            out = self.env_net[0 + gnn_idx](x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=env_edge_attr)
            out = self.dropout(out)
            out = self.act(out)
            if self.skip:
                x = x + out
            else:
                x = out
        x = self.hidden_layer_norm(x)
        x = self.env_net[-1](x)
        graph_emb = x
        out, node_att_scores = self.glob(x, batch)
        return (graph_emb, out)

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob

class InteractiveFusionBlock(nn.Module):

    def __init__(self, code_dim, graph_dim, hidden_dim):
        super().__init__()
        self.graph_attn = nn.MultiheadAttention(graph_dim, num_heads=4)
        self.mp1 = nn.Sequential(nn.Linear(768, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.code_transform = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.graph_transform = nn.Sequential(nn.Linear(graph_dim, hidden_dim), nn.ReLU())
        self.gate_network = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, code_feats, graph_feats):
        processed_code_feats = []
        for cf in code_feats:
            if cf is None:
                processed_code_feats.append(torch.zeros(768).to(graph_feats.device))
            elif isinstance(cf, torch.Tensor):
                if cf.dim() > 1:
                    processed_code_feats.append(cf.squeeze(0).to(graph_feats.device))
                else:
                    processed_code_feats.append(cf.to(graph_feats.device))
            else:
                try:
                    cf_tensor = torch.tensor(cf).to(graph_feats.device)
                    if cf_tensor.dim() > 1:
                        processed_code_feats.append(cf_tensor.squeeze(0))
                    else:
                        processed_code_feats.append(cf_tensor)
                except:
                    processed_code_feats.append(torch.zeros(768).to(graph_feats.device))
        code_feats = torch.stack(processed_code_feats, dim=0)
        code_feats = self.mp1(code_feats)
        attn_graph, _ = self.graph_attn(query=graph_feats, key=code_feats, value=code_feats)
        trans_code = self.code_transform(code_feats)
        trans_graph = self.graph_transform(attn_graph)
        combined = torch.cat([trans_code, trans_graph], dim=-1)
        gate = self.gate_network(combined)
        fused = gate * trans_graph + (1 - gate) * trans_code
        return (fused, trans_code, trans_graph)
import os.path as osp

class Net(nn.Module):

    def __init__(self, deg=0, edge_dim=0, gnn_dim=FLAGS.D, num_blocks=5):
        super().__init__()
        if deg is None or isinstance(deg, int):
            deg = torch.ones(1, dtype=torch.long)
        self.D = FLAGS.D
        self.task = FLAGS.task
        self.target = FLAGS.target
        self.device = FLAGS.device
        code_dim = 768
        if FLAGS.comparative_if:
            if FLAGS.comparative_model == 'HGP':
                self.HGP = HGP.HierNet(in_channels=FLAGS.num_features, hidden_channels=FLAGS.hidden_num, num_layers=3, conv_type='sage', drop_out=0.0)
            elif FLAGS.comparative_model == 'ironman':
                self.ironman = Ironman.GCNNet(in_channels=FLAGS.num_features)
            elif FLAGS.comparative_model == 'pna':
                self.pna = pna.PNANet(in_dim=FLAGS.num_features, deg=deg, num_layer=2, emb_dim=200, edge_dim=edge_dim, drop_ratio=0.5)
            else:
                self.gnn_dse = GNN_DSE.Net(in_channels=FLAGS.num_features)
        else:
            self.CoGNN = CoGNN(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args)
        self.fusion_blocks = nn.ModuleList()
        fusion_dim = FLAGS.D
        for i in range(num_blocks):
            block = InteractiveFusionBlock(code_dim=code_dim, graph_dim=gnn_dim, hidden_dim=fusion_dim)
            self.fusion_blocks.append(block)
        self.loss_fucntion = torch.nn.MSELoss()
        if FLAGS.ablation_stu == 'LM':
            self.MLP1 = nn.Sequential(nn.Linear(768, fusion_dim), nn.ReLU(), nn.Linear(fusion_dim, fusion_dim))
        self.MLPs = nn.ModuleDict()
        if 'regression' in self.task:
            _target_list = self.target
            if not isinstance(FLAGS.target, list):
                _target_list = [self.target]
            self.target_list = [t for t in _target_list]
        else:
            self.target_list = ['perf']
        d = self.D
        if d > 64:
            hidden_channels = [d // 2, d // 4, d // 8, d // 16, d // 32]
        else:
            hidden_channels = [d // 2, d // 4, d // 8]
        for target in self.target_list:
            self.MLPs[target] = MLP(d, FLAGS.out_dim, activation_type=FLAGS.activation, hidden_channels=hidden_channels, num_hidden_lyr=len(hidden_channels))
        self.use_causal = FLAGS.use_causal
        print(f'[Model Init] FLAGS.use_causal = {FLAGS.use_causal}, type = {type(FLAGS.use_causal)}')
        print(f'[Model Init] self.use_causal = {self.use_causal}, type = {type(self.use_causal)}')
        if self.use_causal:
            print(f'[Model Init] Initializing causal modules...')
            self.pragma_encoder = PragmaEncoder(pragma_type_dim=32, pragma_scope_dim=32, pragma_value_dim=32, output_dim=64, max_pragmas=20)
            self.causal_head = CausalHead(context_dim=fusion_dim, pragma_emb_dim=64, num_targets=len(self.target_list), hidden_dim=64, use_attention=True)

    def forward(self, data, code_f, design_point=None, fusion_w: Optional[float]=None):
        if FLAGS.comparative_if:
            if FLAGS.comparative_model == 'HGP':
                graph_nodes_emb = self.HGP(data)
            elif FLAGS.comparative_model == 'ironman':
                graph_nodes_emb = self.ironman(data)
            elif FLAGS.comparative_model == 'pna':
                graph_nodes_emb = self.pna(data)
            else:
                graph_nodes_emb = self.gnn_dse(data)
        else:
            graph_emb, graph_global = self.CoGNN(data)
            graph_nodes_emb = graph_global
        fused_features = []
        for i, block in enumerate(self.fusion_blocks):
            fusion_input = (code_f, graph_nodes_emb)
            fused, code_feats, graph_feats = block(*fusion_input)
            fused_features.append(fused)
        if FLAGS.ablation_stu == 'CoGNN+LM':
            out = torch.mean(torch.stack(fused_features), dim=0)
        elif FLAGS.ablation_stu == 'CoGNN':
            out = graph_nodes_emb
        else:
            code_f = torch.stack([cf.squeeze(0).to(graph_nodes_emb.device) if cf.dim() > 1 else cf.to(graph_nodes_emb.device) for cf in code_f], dim=0)
            code_f = self.MLP1(code_f)
            out = code_f
        out_dict = OrderedDict()
        total_loss = 0
        out_embed = out
        loss_dict = {}
        use_causal_pred = False
        if self.use_causal:
            dp_source = design_point if design_point is not None else getattr(data, 'point', None)
            batch_size = out_embed.size(0)
            if not hasattr(self, '_debug_logged'):
                print(f'[Model Debug] use_causal={self.use_causal}, dp_source={dp_source is not None}, batch_size={batch_size}')
                self._debug_logged = True
            if dp_source is not None:
                if isinstance(dp_source, (list, tuple)):
                    design_points = list(dp_source)
                else:
                    design_points = [dp_source]
                valid_count = sum((1 for dp in design_points if dp is not None))
                if not hasattr(self, '_debug_logged2'):
                    print(f'[Model Debug] design_points len={len(design_points)}, valid_count={valid_count}, batch_size={batch_size}')
                    if valid_count < batch_size:
                        print(f'[Model Debug] Some design_points are None! First few: {[dp is not None for dp in design_points[:5]]}')
                    self._debug_logged2 = True
                if len(design_points) == batch_size and all((dp is not None for dp in design_points)):
                    try:
                        device = out_embed.device
                        if next(self.pragma_encoder.parameters()).device != device:
                            self.pragma_encoder = self.pragma_encoder.to(device)
                        if next(self.causal_head.parameters()).device != device:
                            self.causal_head = self.causal_head.to(device)
                        alpha_list = []
                        pragma_ids_batch = []
                        pragma_mask_batch = []
                        causal_pred_list = []
                        for idx in range(batch_size):
                            pragma_emb = self.pragma_encoder(design_points[idx])
                            pragma_ids_i = getattr(self.pragma_encoder, '_last_pragma_ids', None)
                            pragma_mask_i = getattr(self.pragma_encoder, '_last_pragma_mask', None)
                            pragma_ids_batch.append(pragma_ids_i)
                            pragma_mask_batch.append(pragma_mask_i)
                            causal_pred, alpha = self.causal_head(out_embed[idx:idx + 1], pragma_emb, self.target_list, pragma_mask=pragma_mask_i)
                            alpha_list.append(alpha)
                            causal_pred_list.append(causal_pred)
                        alpha_matrix = torch.cat(alpha_list, dim=0) if alpha_list else None
                        self._last_alpha_matrix = alpha_matrix
                        merged_causal_pred: Optional[OrderedDict] = None
                        if causal_pred_list:
                            merged_causal_pred = OrderedDict()
                            for target_name in self.target_list:
                                merged_causal_pred[target_name] = torch.cat([cp[target_name] for cp in causal_pred_list], dim=0)
                        self._last_causal_pred = merged_causal_pred
                        self._last_pragma_ids_batch = pragma_ids_batch
                        self._last_pragma_mask_batch = pragma_mask_batch
                        mode = getattr(FLAGS, 'causal_main_pred_mode', 'replace')
                        if mode == 'replace':
                            if merged_causal_pred is not None:
                                for target_name in self.target_list:
                                    out_dict[target_name] = merged_causal_pred[target_name]
                                use_causal_pred = True
                        elif mode == 'fusion':
                            w = float(fusion_w) if fusion_w is not None else 0.0
                            w = max(0.0, min(1.0, w))
                            for target_name in self.target_list:
                                mlp_out = self.MLPs[target_name](out_embed)
                                if merged_causal_pred is not None and target_name in merged_causal_pred:
                                    causal_out = merged_causal_pred[target_name]
                                    out_dict[target_name] = (1.0 - w) * mlp_out + w * causal_out
                                else:
                                    out_dict[target_name] = mlp_out
                            use_causal_pred = True
                        else:
                            use_causal_pred = False
                    except Exception as e:
                        print(f'Warning: Causal prediction failed, falling back to standard prediction: {e}')
                        use_causal_pred = False
        if not use_causal_pred:
            for target_name in self.target_list:
                out = self.MLPs[target_name](out_embed)
                y = _get_y_with_target(data, target_name)
                if self.task == 'regression':
                    target = y.view((len(y), FLAGS.out_dim))
                    if FLAGS.loss == 'RMSE':
                        loss = torch.sqrt(self.loss_fucntion(out, target))
                    elif FLAGS.loss == 'MSE':
                        loss = self.loss_fucntion(out, target)
                    else:
                        raise NotImplementedError()
                else:
                    target = y.view(len(y))
                    loss = self.loss_fucntion(out, target)
                out_dict[target_name] = out
                total_loss += loss
                loss_dict[target_name] = loss
        else:
            for target_name in self.target_list:
                out = out_dict[target_name]
                y = _get_y_with_target(data, target_name)
                if self.task == 'regression':
                    target = y.view((len(y), FLAGS.out_dim))
                    if FLAGS.loss == 'RMSE':
                        loss = torch.sqrt(self.loss_fucntion(out, target))
                    elif FLAGS.loss == 'MSE':
                        loss = self.loss_fucntion(out, target)
                    else:
                        raise NotImplementedError()
                else:
                    target = y.view(len(y))
                    loss = self.loss_fucntion(out, target)
                total_loss += loss
                loss_dict[target_name] = loss
        return (out_dict, total_loss, loss_dict)