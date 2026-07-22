import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import pickle
import random
import numpy as np
import torch


from models.rrc_gen import RRC_Gen


from modules.dataloaders import R2DataLoader
from modules.metrics     import compute_scores
from modules.optimizers  import build_optimizer, build_lr_scheduler
from modules.tokenizers  import Tokenizer
from modules.trainer     import Trainer
from modules.loss        import compute_loss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


def parse_agrs():
    parser = argparse.ArgumentParser()
    # MIMIC-CXR PATHS
    # parser.add_argument('--embedding_path', type=str, default='./data/embedding/embeddings.txt')
    # parser.add_argument('--image_dir', type=str, default=r'/root/autodl-tmp/data/mimic_cxr/images')
    # parser.add_argument('--ann_path', type=str, default=r'/root/autodl-tmp/difu/data/mimic_cxr/annotation_mi.json')
    # parser.add_argument('--vocab_path', type=str, default=r'/root/autodl-tmp/difu/data/vocabmimic.pkl')
    # parser.add_argument('--save_dir', type=str, default='./records_mimic/ablation/1.baseON_se_esca_rcclOFF')
    # parser.add_argument('--record_dir', type=str, default='./records_mimic/ablation/1.baseON_se_esca_rcclOFF')

    # IU PATHS
    parser.add_argument('--image_dir',  type=str, default=r'E:\usman\DATA_SETS\data\iu_xray\images')
    parser.add_argument('--ann_path',   type=str, default='./data/iu_xray/annotation.json')
    parser.add_argument('--vocab_path', type=str, default='./data/New_vocab.pkl')
    parser.add_argument('--embedding_path', type=str,default='./data/embedding/embeddings.txt')
    parser.add_argument('--save_dir',   type=str, default='./records_iu')
    parser.add_argument('--record_dir', type=str, default='./records_iu')
    #parser.add_argument('--record_dir', type=str, default='./records_iu/ablation/4.base+se+rccl+escaON')#hyperameterAnalysisTemp/1.0

    parser.add_argument('--dataset_name', type=str, default='iu_xray',choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--threshold',   type=int,  default=3)
    parser.add_argument('--num_workers', type=int,  default=0)
    parser.add_argument('--batch_size',  type=int,  default=8)
    parser.add_argument('--max_seq_length',          type=int,   default=50)
    parser.add_argument('--visual_extractor',            type=str,  default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    # Ablation
    parser.add_argument('--rccl_weight_iu', type=float, default=1.0, help='Weight for RCCL loss.') #1.0
    parser.add_argument('--rccl_weight_mi', type=float, default=0.01, help='Weight for RCCL loss.')
    parser.add_argument('--use_se', type=bool, default=True,help='Enable SE channel recalibration.' )
    parser.add_argument('--use_ecsa', type=bool,default=True, help='Enable ECSA spatial attention.')

    parser.add_argument('--d_model',      type=int,   default=512)
    parser.add_argument('--d_ff',         type=int,   default=512)
    parser.add_argument('--d_vf',         type=int,   default=2048)
    parser.add_argument('--num_heads',    type=int,   default=8)
    parser.add_argument('--num_layers',   type=int,   default=3)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--logit_layers', type=int,   default=1)
    parser.add_argument('--bos_idx',      type=int,   default=0)
    parser.add_argument('--eos_idx',      type=int,   default=0)
    parser.add_argument('--pad_idx',      type=int,   default=0)
    parser.add_argument('--use_bn',       type=int,   default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.3)
    parser.add_argument('--hidden_dim',   type=int,   default=768)

    parser.add_argument('--topk',     type=int, default=32)
    parser.add_argument('--cmm_size', type=int, default=2048)
    parser.add_argument('--cmm_dim',  type=int, default=512)

    parser.add_argument('--sample_method',       type=str,   default='beam_search')
    parser.add_argument('--beam_size',           type=int,   default=5)
    parser.add_argument('--temperature',         type=float, default=1.0)
    parser.add_argument('--sample_n',            type=int,   default=1)
    parser.add_argument('--group_size',          type=int,   default=1)
    parser.add_argument('--output_logsoftmax',   type=int,   default=1)
    parser.add_argument('--decoding_constraint', type=int,   default=0)
    parser.add_argument('--block_trigrams',      type=int,   default=1)

    parser.add_argument('--n_gpu',          type=int, default=1)
    parser.add_argument('--epochs',         type=int, default=25)
    parser.add_argument('--early_stop',     type=int, default=25)
    parser.add_argument('--log_period',     type=int, default=200)
    parser.add_argument('--save_period',    type=int, default=1)
    parser.add_argument('--monitor_mode',   type=str, default='max', choices=['min', 'max'])
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')

    parser.add_argument('--optim',          type=str,   default='Adam')
    parser.add_argument('--lr_ve',          type=float, default=0.002)
    parser.add_argument('--lr_ed',          type=float, default=7e-4)
    parser.add_argument('--weight_decay',   type=float, default=0.00005)
    parser.add_argument('--adam_betas',     type=tuple, default=(0.9, 0.98))
    parser.add_argument('--adam_eps',       type=float, default=1e-9)
    parser.add_argument('--amsgrad',        type=bool,  default=True)
    parser.add_argument('--noamopt_warmup', type=int,   default=4000)
    parser.add_argument('--noamopt_factor', type=int,   default=1)
    parser.add_argument('--momentum',       type=float, default=0.9)
    parser.add_argument('--nesterov',       type=bool,  default=True)

    # EVALUATION SPEEDUP
    parser.add_argument('--eval_subset_size', type=int, default=9999)
    parser.add_argument('--full_eval_period', type=int, default=5)


    parser.add_argument('--lr_scheduler', type=str,   default='StepLR')
    parser.add_argument('--step_size',    type=int,   default=20)
    parser.add_argument('--gamma',        type=float, default=0.5)

    parser.add_argument('--seed',   type=int, default=23) #23
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    args = parse_agrs()
    seed_everything(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(args, device=device)

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            allowed = {'builtins', '__main__', 'collections', 'torch', 'numpy'}
            if module in allowed and '__' not in name:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"Unsafe: {module}.{name}")

    with open(args.vocab_path, 'rb') as f:
        vocab = RestrictedUnpickler(f).load()

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True,
                                    generator=g, worker_init_fn=seed_worker)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False,
                                  generator=g, worker_init_fn=seed_worker)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False,
                                   generator=g, worker_init_fn=seed_worker)

    model = RRC_Gen(args, tokenizer)
    criterion = compute_loss
    metrics = compute_scores
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = Trainer(
        model, criterion, metrics, optimizer, args, lr_scheduler,
        train_dataloader, val_dataloader, test_dataloader,
    )
    trainer.train()


if __name__ == '__main__':
    main()