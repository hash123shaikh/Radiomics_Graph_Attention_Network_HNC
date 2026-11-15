"""
RadGraph GAT (mirrors ReadoutGAT / ReadoutGAT_clinical from the authors' GAT_modeling.py,
with Table S2 hyper-parameters):

  GATConv(F -> h, heads)  -> LayerNorm(per graph) -> ReLU
  ... (n_layers - 2) x [GATConv(h*heads -> h, heads) -> LayerNorm -> ReLU]
  GATConv(h*heads -> h, heads=1)                       (final layer, no norm)
  read-out: features of the GTV node of every graph
  Dropout(0.5) -> [concat clinical vector] -> Linear -> 1 logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm

from radgraph import config


class RadGraphGAT(nn.Module):

    def __init__(self, n_node_features, n_clinical=0, task='LR', hidden_dim=None,
                 n_heads=None, n_layers=None, dropout=None, use_edge_attr=None):
        super().__init__()
        g = config.get_gat_config(task)
        self.hparams = dict(
            n_node_features=n_node_features, n_clinical=n_clinical, task=task,
            hidden_dim=hidden_dim or g['hidden_dim'], n_heads=n_heads or g['n_heads'],
            n_layers=n_layers or g['n_layers'], dropout=g['dropout'] if dropout is None else dropout,
            use_edge_attr=config.USE_EDGE_ATTR if use_edge_attr is None else use_edge_attr)
        h, H, L, p = (self.hparams[k] for k in ('hidden_dim', 'n_heads', 'n_layers', 'dropout'))
        edge_dim = 1 if self.hparams['use_edge_attr'] else None

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        if L == 1:
            self.convs.append(GATConv(n_node_features, h, heads=H, dropout=p, edge_dim=edge_dim))
            out_dim = h * H
        else:
            in_dim = n_node_features
            for _ in range(L - 1):
                self.convs.append(GATConv(in_dim, h, heads=H, dropout=p, edge_dim=edge_dim))
                self.norms.append(LayerNorm(h * H))
                in_dim = h * H
            self.convs.append(GATConv(in_dim, h, heads=1, dropout=p, edge_dim=edge_dim))
            out_dim = h

        self.n_clinical = n_clinical
        self.classifier = nn.Linear(out_dim + n_clinical, 1)

    # ------------------------------------------------------------------
    def _gtv_index(self, batch):
        return batch.ptr[:-1] + batch.gtv_node_idx.view(-1)

    def _conv(self, conv, x, batch, **kw):
        if self.hparams['use_edge_attr']:
            return conv(x, batch.edge_index, edge_attr=batch.edge_attr, **kw)
        return conv(x, batch.edge_index, **kw)

    def embed(self, batch, return_attention=False):
        x, att = batch.x, None
        for i, conv in enumerate(self.convs):
            last = i == len(self.convs) - 1
            if last and return_attention:
                x, att = self._conv(conv, x, batch, return_attention_weights=True)
            else:
                x = self._conv(conv, x, batch)
            if i < len(self.norms):
                x = F.relu(self.norms[i](x, batch.batch))
        return x[self._gtv_index(batch)], att

    def forward(self, batch):
        z, _ = self.embed(batch)
        z = F.dropout(z, p=config.READOUT_DROPOUT, training=self.training)
        if self.n_clinical > 0:
            clin = batch.clinical.view(z.size(0), -1)
            z = torch.cat([z, clin], dim=1)
        return self.classifier(z).view(-1)

    @torch.no_grad()
    def attention_to_gtv(self, batch):
        """
        Final-layer attention the GTV read-out node pays to each node
        (edges j -> GTV; alpha averaged over heads). Returns (edge_index, alpha).
        """
        _, (ei, alpha) = self.embed(batch, return_attention=True)
        return ei, alpha.mean(dim=-1)
