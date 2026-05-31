import torch
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import reset

class MyGlobalAttention(torch.nn.Module):

    def __init__(self, gate_nn, nn=None):
        super(MyGlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        gate = self.gate_nn(x)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        return (out, gate)

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__, self.gate_nn, self.nn)