
# KIMI update for SMAN
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset, IuxrayDatasetProgressive


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, generator=None, worker_init_fn=None):
        self.args         = args
        self.dataset_name = args.dataset_name
        self.batch_size   = args.batch_size
        self.shuffle      = shuffle
        self.num_workers  = args.num_workers
        self.tokenizer    = tokenizer
        self.split        = split

        # =====================================================================
        # DATASET-SPECIFIC TRANSFORMS
        # ─────────────────────────────────────────────────────────────────────
        # IU-Xray:
        #   Train : Resize(256) → RandomCrop(224) → RandomHorizontalFlip
        #   Val/Test: Resize(224,224) → ToTensor → Normalize
        #   Normalization: ImageNet stats — ResNet101 pretrained pe best
        #
        # MIMIC-CXR:
        #   Train : Resize(256) → RandomCrop(224) → RandomHorizontalFlip(p=0.5)
        #           + RandomAffine(degrees=0, translate=(0.05,0.05)) — mild shift
        #   Val/Test: Resize(224,224) → ToTensor → Normalize
        #   Normalization: ImageNet stats (same as IU — CMN official standard)
        #
        # MIMIC mein extra RandomAffine kyun:
        #   Large dataset (220k) pe mild translation augmentation generalization
        #   improve karta hai early epochs mein — model zyada robust features
        #   sikhta hai → BLEU-1 higher start hota hai epoch 1 se
        # =====================================================================

        if self.dataset_name == 'iu_xray':
            # ── IU-Xray transforms ───────────────────────────────────────────
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
                ])

        else:
            # ── MIMIC-CXR transforms ─────────────────────────────────────────
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.05, 0.05)  # mild spatial shift only
                    ),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406),
                    #                      (0.229, 0.224, 0.225))
                    transforms.Normalize((0.5056, 0.5056, 0.5056),
                                         (0.252, 0.252, 0.252))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406),
                    #                      (0.229, 0.224, 0.225))
                    transforms.Normalize((0.5056, 0.5056, 0.5056),
                                         (0.252, 0.252, 0.252))
                ])

        # =====================================================================
        # DATASET SELECTION
        # =====================================================================
        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(
                self.args, self.tokenizer, self.split,
                transform=self.transform
            )
        else:
            self.dataset = MimiccxrSingleImageDataset(
                self.args, self.tokenizer, self.split,
                transform=self.transform
            )

        # =====================================================================
        # DATALOADER KWARGS
        # pin_memory=True  : faster GPU transfer (MIMIC pe large batch)
        # drop_last=False  : koi sample waste nahi
        # num_workers      : IU=0 (small, fast), MIMIC=4 (large, parallel load)
        # =====================================================================
        self.init_kwargs = {
            'dataset':        self.dataset,
            'batch_size':     self.batch_size,
            'shuffle':        self.shuffle,
            'collate_fn':     self.collate_fn,
            'num_workers':    self.num_workers,
            'worker_init_fn': worker_init_fn,
            'generator':      generator,
            'pin_memory':     torch.cuda.is_available(),
            'drop_last':      False,
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_id_batch, image_batch, report_ids_batch, \
            report_masks_batch, seq_lengths_batch = zip(*data)

        image_batch    = torch.stack(image_batch, 0)
        max_seq_length = max(seq_lengths_batch)

        target_batch       = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        return (image_id_batch,
                image_batch,
                torch.LongTensor(target_batch),
                torch.FloatTensor(target_masks_batch))


# Purana dataloader

# import numpy as np
# import torch
# from torch.utils.data import DataLoader,WeightedRandomSampler
# from torchvision import transforms
# from collections import Counter
#
# from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset, IuxrayDatasetProgressive
#
#
# class R2DataLoader(DataLoader):
#     def __init__(self, args, tokenizer, split, shuffle, generator=None, worker_init_fn=None):
#         self.args = args
#         self.dataset_name = args.dataset_name
#         self.batch_size = args.batch_size
#         self.shuffle = shuffle
#         self.num_workers = args.num_workers
#         self.tokenizer = tokenizer
#         self.split = split
#
#         # TRANSFORMS
#         if split == 'train':
#             self.transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.RandomCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))
#             ])
#
#         # DATASET
#         if self.dataset_name == 'iu_xray':
#             self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
#         else:
#             self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
#
#         # INIT KWARGS (NOW with generator + worker_init_fn)
#         self.init_kwargs = {
#             'dataset': self.dataset,
#             'batch_size': self.batch_size,
#             'shuffle': self.shuffle,
#             'collate_fn': self.collate_fn,
#             'num_workers': self.num_workers,
#             'worker_init_fn': worker_init_fn,
#             'generator': generator,
#         }
#         super().__init__(**self.init_kwargs)
#     @staticmethod
#     def collate_fn(data):
#         image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch = zip(*data)
#         image_batch = torch.stack(image_batch, 0)
#         max_seq_length = max(seq_lengths_batch)
#
#         target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
#         target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
#
#         for i, report_ids in enumerate(report_ids_batch):
#             target_batch[i, :len(report_ids)] = report_ids
#
#         for i, report_masks in enumerate(report_masks_batch):
#             target_masks_batch[i, :len(report_masks)] = report_masks
#
#         return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch)

