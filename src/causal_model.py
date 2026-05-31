import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import re

class PragmaEncoder(nn.Module):

    def __init__(self, pragma_type_dim: int=32, pragma_scope_dim: int=32, pragma_value_dim: int=32, output_dim: int=64, max_pragmas: int=20):
        super().__init__()
        self.max_pragmas = max_pragmas
        self.output_dim = output_dim
        self.known_types = ['PIPE', 'PARA', 'TILE', 'TILING', 'ARRAY_PARTITION', 'DEPENDENCE', 'RESOURCE']
        self.type_vocab_size = len(self.known_types) + 1
        self.type_embedding = nn.Embedding(self.type_vocab_size, pragma_type_dim)
        self.scope_embedding = nn.Embedding(100, pragma_scope_dim)
        self.value_num_proj = nn.Linear(1, pragma_value_dim)
        self.value_str_embedding = nn.Embedding(50, pragma_value_dim)
        self.fusion = nn.Sequential(nn.Linear(pragma_type_dim + pragma_scope_dim + pragma_value_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
        self.value_str_map = {'flatten': 0, 'off': 1, 'on': 2, '': 3}
        self.value_str_map_size = len(self.value_str_map)

    def _parse_pragma_id(self, pragma_id: str) -> Tuple[str, str]:
        cleaned = pragma_id.strip('_')
        parts = cleaned.split('__')
        if len(parts) >= 2:
            ptype = parts[0]
            scope = '__'.join(parts[1:])
        else:
            ptype = parts[0] if parts else 'UNKNOWN'
            scope = 'GLOBAL'
        return (ptype.upper(), scope)

    def _get_type_idx(self, ptype: str) -> int:
        if ptype in self.known_types:
            return self.known_types.index(ptype)
        return len(self.known_types)

    def _get_scope_idx(self, scope: str) -> int:
        return hash(scope) % 100

    def _encode_value(self, value) -> torch.Tensor:
        device = next(self.parameters()).device
        if torch.is_tensor(value):
            try:
                if value.numel() > 0:
                    value = float(value.detach().view(-1)[0].item())
                else:
                    value = 0.0
            except Exception:
                value = 0.0
        if isinstance(value, (int, float)):
            val_tensor = torch.tensor([[float(value) / 100.0]], device=device)
            return self.value_num_proj(val_tensor)
        elif isinstance(value, str):
            if value in self.value_str_map:
                idx = self.value_str_map[value]
            else:
                idx = self.value_str_map_size
            return self.value_str_embedding(torch.tensor([idx], device=device))
        else:
            return self.value_num_proj(torch.tensor([[0.0]], device=device))

    def forward(self, design_point: Dict[str, any]) -> torch.Tensor:
        device = next(self.parameters()).device
        pragma_embeddings: List[torch.Tensor] = []
        pragma_ids: List[Optional[str]] = []
        pragma_mask: List[int] = []
        items = sorted(design_point.items(), key=lambda x: x[0])
        items = items[:self.max_pragmas]
        for pragma_id, value in items:
            ptype, scope = self._parse_pragma_id(pragma_id)
            type_idx = self._get_type_idx(ptype)
            scope_idx = self._get_scope_idx(scope)
            type_emb = self.type_embedding(torch.tensor([type_idx], device=device))
            scope_emb = self.scope_embedding(torch.tensor([scope_idx], device=device))
            value_emb = self._encode_value(value)
            combined = torch.cat([type_emb, scope_emb, value_emb], dim=-1)
            pragma_emb = self.fusion(combined)
            pragma_embeddings.append(pragma_emb)
            pragma_ids.append(pragma_id)
            pragma_mask.append(1)
        if len(pragma_embeddings) < self.max_pragmas:
            pad_n = self.max_pragmas - len(pragma_embeddings)
            zero_pad = torch.zeros(pad_n, self.output_dim, device=device)
            if len(pragma_embeddings) > 0:
                emb = torch.cat(pragma_embeddings, dim=0)
                emb = torch.cat([emb, zero_pad], dim=0)
            else:
                emb = torch.cat([torch.zeros(0, self.output_dim, device=device), zero_pad], dim=0)
            pragma_ids.extend([None] * pad_n)
            pragma_mask.extend([0] * pad_n)
        else:
            emb = torch.cat(pragma_embeddings, dim=0) if len(pragma_embeddings) > 0 else torch.zeros(self.max_pragmas, self.output_dim, device=device)
            if len(pragma_mask) < self.max_pragmas:
                pragma_mask.extend([0] * (self.max_pragmas - len(pragma_mask)))
        self._last_pragma_ids = pragma_ids
        self._last_pragma_mask = torch.tensor(pragma_mask, device=device, dtype=torch.bool)
        return emb

class CausalHead(nn.Module):

    def __init__(self, context_dim: int, pragma_emb_dim: int, num_targets: int, hidden_dim: int=64, use_attention: bool=True):
        super().__init__()
        self.context_dim = context_dim
        self.pragma_emb_dim = pragma_emb_dim
        self.num_targets = num_targets
        self.use_attention = use_attention
        self.baseline_heads = nn.ModuleDict()
        for i in range(num_targets):
            self.baseline_heads[f'target_{i}'] = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        if use_attention:
            self.target_queries = nn.Parameter(torch.randn(num_targets, context_dim))
            self.pragma_proj = nn.Linear(pragma_emb_dim, context_dim)
        else:
            self.alpha_mlps = nn.ModuleDict()
            for i in range(num_targets):
                self.alpha_mlps[f'target_{i}'] = nn.Sequential(nn.Linear(context_dim + pragma_emb_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.pragma_to_qor = nn.ModuleDict()
        for i in range(num_targets):
            self.pragma_to_qor[f'target_{i}'] = nn.Linear(pragma_emb_dim, 1)

    def forward(self, context: torch.Tensor, pragma_embeddings: torch.Tensor, target_names: List[str], pragma_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        num_pragmas = pragma_embeddings.shape[0]
        pragma_emb_batch = pragma_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        predictions = OrderedDict()
        alpha_list = []
        if pragma_mask is not None:
            if pragma_mask.dim() == 1:
                pragma_mask_b = pragma_mask.unsqueeze(0).expand(batch_size, -1)
            else:
                pragma_mask_b = pragma_mask
            pragma_mask_b = pragma_mask_b.to(context.device)
        else:
            pragma_mask_b = None
        for target_idx, target_name in enumerate(target_names):
            baseline = self.baseline_heads[f'target_{target_idx}'](context)
            if self.use_attention:
                target_query = self.target_queries[target_idx].unsqueeze(0).to(context.device)
                query = context + target_query
                key = self.pragma_proj(pragma_emb_batch)
                alpha_raw = torch.bmm(query.unsqueeze(1), key.transpose(1, 2)).squeeze(1)
                if pragma_mask_b is not None:
                    alpha_raw = alpha_raw.masked_fill(~pragma_mask_b, float('-inf'))
                try:
                    from config import FLAGS
                    tau = float(getattr(FLAGS, 'causal_alpha_tau', 1.0))
                except Exception:
                    tau = 1.0
                if tau <= 0:
                    tau = 1.0
                alpha = torch.softmax(alpha_raw / tau, dim=-1)
            else:
                context_expanded = context.unsqueeze(1).expand(-1, num_pragmas, -1)
                combined = torch.cat([context_expanded, pragma_emb_batch], dim=-1)
                alpha_raw = self.alpha_mlps[f'target_{target_idx}'](combined).squeeze(-1)
                if pragma_mask_b is not None:
                    alpha_raw = alpha_raw.masked_fill(~pragma_mask_b, float('-inf'))
                try:
                    from config import FLAGS
                    tau = float(getattr(FLAGS, 'causal_alpha_tau', 1.0))
                except Exception:
                    tau = 1.0
                if tau <= 0:
                    tau = 1.0
                alpha = torch.softmax(alpha_raw / tau, dim=-1)
            alpha_list.append(alpha)
            pragma_contrib = self.pragma_to_qor[f'target_{target_idx}'](pragma_emb_batch)
            pragma_contrib = pragma_contrib.squeeze(-1)
            weighted_contrib = (alpha.unsqueeze(-1) * pragma_contrib.unsqueeze(-1)).sum(dim=1)
            prediction = baseline + weighted_contrib
            predictions[target_name] = prediction
        alpha_matrix = torch.stack(alpha_list, dim=-1)
        return (predictions, alpha_matrix)