import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from modules.base_cmn import BaseCMN
from modules.visual_extractor import RVFE

CONFIG = {
    "temp_range": [0.07, 0.5, 1.0, 1.5, 2.0], #1.0
    "top_k": 5,
    "num_augs": 2,
}


class FeatureProjection(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class RCCL(nn.Module): #Relational Contrastive Clustering Learning
    def __init__(self, hidden_size, output_size=None):
        super().__init__()
        self.temp = 1.0
        self.top_k = CONFIG["top_k"]
        self.output_size = output_size if output_size is not None else hidden_size
        self.proj1 = FeatureProjection(hidden_size, self.output_size, hidden_size)
        self.proj2 = FeatureProjection(hidden_size, self.output_size, hidden_size)

    def _check_inputs(self, x1, x2):
        if x1.dim() != 2 or x2.dim() != 2:
            raise ValueError(f"Inputs must be 2D: got {x1.shape}, {x2.shape}")
        if x1.size(0) != x2.size(0):
            raise ValueError("Batch sizes must match.")
        if torch.isnan(x1).any() or torch.isnan(x2).any():
            raise ValueError("Inputs contain NaN values.")
        if x1.size(1) != self.proj1.net[0].in_features:
            raise ValueError(
                f"Input dim ({x1.size(1)}) != projection ({self.proj1.net[0].in_features})"
            )

    def make_rccl_aug_pair(self, feat, noise_std=0.01, drop_prob=0.1):
        aug1 = F.dropout(feat, p=drop_prob, training=self.training)
        aug2 = F.dropout(feat, p=drop_prob, training=self.training)
        if self.training:
            aug1 = aug1 + torch.randn_like(aug1) * noise_std
            aug2 = aug2 + torch.randn_like(aug2) * noise_std
        return aug1, aug2

    def _make_mask(self, sim_mat):
        with torch.no_grad():
            B = sim_mat.size(0)
            if B <= 1:
                return torch.ones((B, B), device=sim_mat.device)
            k = min(self.top_k, B - 1)
            dist_mat = 1 - sim_mat.clamp(0, 1)
            dist_mat = dist_mat + torch.eye(B, device=sim_mat.device) * 1e9
            _, top_idx = torch.topk(-dist_mat, k=k, dim=1)
            adj = torch.zeros((B, B), device=sim_mat.device)
            rows = torch.arange(B, device=sim_mat.device).unsqueeze(1).expand(-1, k)
            adj[rows.reshape(-1), top_idx.reshape(-1)] = 1
            adj = ((adj + adj.T) > 0).float()
            csr = csr_matrix(adj.cpu().numpy().astype(np.float32))
            _, labels = connected_components(csr, directed=False)
            labels = torch.from_numpy(labels).to(sim_mat.device)
            mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            return torch.nan_to_num(mask, nan=0.0)

    def _contrastive_loss(self, logits, mask, diag=None):
        if diag is not None:
            mask = mask * (1 - diag)
            exp_logits = torch.exp(logits / self.temp) * (1 - diag)
        else:
            exp_logits = torch.exp(logits / self.temp)
        denom = exp_logits.sum(dim=1, keepdim=True) + 1e-8
        log_probs = (logits / self.temp) - torch.log(denom)
        mask_sum = mask.sum(1) + 1e-8
        pos_mean = (mask * log_probs).sum(1) / mask_sum
        return -torch.nan_to_num(pos_mean, nan=0.0).mean()

    def _local_rows(self, matrix, local_batch):
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return matrix[:local_batch]
        start = dist.get_rank() * local_batch
        return matrix[start:start + local_batch]

    def forward(self, x1, x2, temp=1.0): #1.0
        self._check_inputs(x1, x2)
        self.temp = max(float(temp), 1e-6)
        B = x1.size(0)

        r1 = F.normalize(self.proj1(x1), dim=-1)
        r2 = F.normalize(self.proj1(x2), dim=-1)
        r1_all = dist_gather(r1)
        r2_all = dist_gather(r2)

        z1 = F.normalize(self.proj2(x1), dim=-1)
        z2 = F.normalize(self.proj2(x2), dim=-1)
        z1_all = dist_gather(z1)
        z2_all = dist_gather(z2)
        N = z1_all.size(0)

        mask1 = self._make_mask(r1_all @ r1_all.T)
        mask2 = self._make_mask(r2_all @ r2_all.T)
        mask = ((mask1 + mask2) > 0).float()
        mask = ((mask + torch.eye(N, device=x1.device)) > 0).float()
        mask_rows = self._local_rows(mask, B)

        logits1 = z1 @ z2_all.T
        logits2 = z2 @ z1_all.T
        diag = torch.eye(logits1.size(1), device=logits1.device)

        loss = (self._contrastive_loss(logits1, mask_rows, diag) +
                self._contrastive_loss(logits2, mask_rows, diag)) / 2
        return torch.nan_to_num(loss, nan=0.0)


def dist_gather(t):
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return t.clone()
    world = dist.get_world_size()
    rank = dist.get_rank()
    out = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(out, t)
    out[rank] = t.clone()
    return torch.cat(out, dim=0)


class RRC_Gen(nn.Module):

    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)

        self.visual_extractor = RVFE(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        self.rccl = RCCL(hidden_size=256, output_size=256)
        self.rccl_weight = args.rccl_weight_iu
        self.rccl_weight = args.rccl_weight_mi

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def forward_iu_xray(self, images, captions=None, mode='train', update_opts={}):
        att_feats_0, fc_feats_0, wl_feats_0, contrastive_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, wl_feats_1, contrastive_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, captions, mode='forward')
            if isinstance(output, tuple):
                output = output[0]
            rccl_loss = self.rccl(contrastive_0, contrastive_1)
            return output,  self.args.rccl_weight_iu * rccl_loss

        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(
                fc_feats, att_feats, mode='sample', update_opts=update_opts
            )
            return output, output_probs
        else:
            raise ValueError("Invalid mode. Use 'train' or 'sample'.")

    def forward_mimic_cxr(self, images, captions=None, mode='train', update_opts={}):
        att_features, func_feature, wl_feats, contrastive_feat = \
            self.visual_extractor(images)

        if mode == 'train':
            output = self.encoder_decoder(func_feature, att_features, captions, mode='forward')
            if isinstance(output, tuple):
                output = output[0]

            # Simple augmentation-based RCCL
            aug1, aug2 = self.rccl.make_rccl_aug_pair(contrastive_feat)
            rccl_loss = self.rccl(aug1, aug2, temp=1.0)
            return output, self.args.rccl_weight_mi * rccl_loss

        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(
                func_feature, att_features, mode='sample', update_opts=update_opts
            )
            return output, output_probs
        else:
            raise ValueError("Invalid mode. Use 'train' or 'sample'.")