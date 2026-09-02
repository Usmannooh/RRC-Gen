import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from modules.base_cmn import BaseCMN
from modules.visual_extractor import RVFE




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
