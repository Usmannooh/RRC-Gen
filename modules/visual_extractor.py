

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RVFE(nn.Module):

    def __init__(self, args):
        super().__init__()

        # ── Component switches (read from args) ──────────────────────────────
        self.use_se   = getattr(args, 'use_se',   True)   
        self.use_ecsa = getattr(args, 'use_ecsa', True)   

        self.target_feat_dim = args.d_vf
        self.native_feat_dim = 2048
        backbone = getattr(models, args.visual_extractor)( pretrained=args.visual_extractor_pretrained )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        if self.target_feat_dim != self.native_feat_dim:
            self.feat_proj = nn.Conv2d(self.native_feat_dim, self.target_feat_dim,
                kernel_size=1, bias=False )
            nn.init.kaiming_normal_(self.feat_proj.weight, mode='fan_out', nonlinearity='relu' )
        else:
            self.feat_proj = nn.Identity()
        if self.use_se:
            self.se_block = nn.Sequential( nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.target_feat_dim,
                          self.target_feat_dim // 16, 1, bias=False), nn.ReLU(inplace=True),
                nn.Conv2d(self.target_feat_dim // 16, self.target_feat_dim, 1, bias=False),
                nn.Sigmoid() )
            print("[RVFE] SE Channel Recalibration: ON")
        else:
            self.se_block = None
            print("[RVFE] SE Channel Recalibration: OFF (ablation)") #Squeeze-and-Excitation (SE)
        if self.use_ecsa:
            self.spatial_attn = nn.Sequential( nn.Conv2d(self.target_feat_dim, self.target_feat_dim // 16, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.target_feat_dim // 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.target_feat_dim // 16,1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid() )
            print("[RVFE] ECSA Spatial Attention: ON") #Enhanced Channel-Spatial Attention
        else:
            self.spatial_attn = None
            print("[RVFE] ECSA Spatial Attention: OFF (ablation)")
        self.contrastive_proj = nn.Sequential( nn.Linear(self.target_feat_dim, self.target_feat_dim),
            nn.ReLU(inplace=True), nn.Dropout(0.1),nn.Linear(self.target_feat_dim, 256) )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images):

        feats = self.backbone(images)   # (B, 2048, 7, 7)
        feats = self.feat_proj(feats)
        if self.use_se and self.se_block is not None:
            se_weights = self.se_block(feats)   # (B, d_vf, 1, 1)
            feats = feats * se_weights
        if self.use_ecsa and self.spatial_attn is not None:
            spatial_weights = self.spatial_attn(feats)   # (B, 1, 7, 7)
            feats = feats * spatial_weights
        B, C, H, W   = feats.shape
        patch_feats  = feats.reshape(B, C, H * W).permute(0, 2, 1)
        avg_feats = patch_feats.mean(dim=1)             # (B, d_vf)
        avg_feats = F.normalize(avg_feats, dim=1)
        contrastive_feat = F.normalize(self.contrastive_proj(avg_feats), dim=1 )
        wl_feats = avg_feats.clone()

        return patch_feats, avg_feats, wl_feats, contrastive_feat
