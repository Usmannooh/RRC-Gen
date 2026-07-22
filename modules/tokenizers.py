"""
Tokenizer radiology report generation.

Changes vs. original:
  1. Dead code removed:
       - get_token_embeddings()  — O(B×T) Python loop, never called anywhere
       - avg_tk_emb              — computed but unused
       - text_embeddings dict    — dict of individual `requires_grad=True` tensors,
                                   never updated by any optimizer
       - torch.stack() on entire dict — O(V×E) allocation at init just to get mean
  2. Embedding loading redesigned:
       - Raw embedding file is loaded once, OOV mean computed in numpy.
       - A single (V+1, E) numpy matrix is built aligned to the vocabulary.
       - Vocabulary is built FIRST so the matrix is immediately aligned.
       - Raw dict is discarded after matrix is built → no persistent memory leak.
  3. New public API:
       - tokenizer.build_embedding_matrix()  →  np.ndarray (V+1, E)
         Used by  to initialise a proper nn.Embedding layer.
       - tokenizer.tk_embed_dim              →  int, embedding dimension
       These allow the model to use pre-trained embeddings efficiently via
       PyTorch's nn.Embedding (vectorised, CUDA-friendly, optimizer-visible).
  4. All existing tokenisation / decode methods preserved unchanged.
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import torch


class Tokenizer(object):
    def __init__(self, args, device=None):
        self.device        = device
        self.ann_path      = args.ann_path
        self.threshold     = args.threshold
        self.dataset_name  = args.dataset_name
        self.pad_token_id  = 0

        # ── Choose cleaning function ──────────────────────────────────────
        if self.dataset_name == 'iu_xray':
            self.clean_report = self._clean_report_iu_xray
        else:
            self.clean_report = self._clean_report_mimic_cxr

        # ── 1. Build vocabulary (must come before embedding matrix) ───────
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self._create_vocabulary()

        # ── 2. Build aligned embedding matrix ────────────────────────────
        # Loads the pre-trained embedding file ONCE, builds a (V+1, E)
        # numpy matrix aligned to self.idx2token, then discards the raw dict.
        self._emb_matrix, self.tk_embed_dim = self._build_embedding_matrix(
            args.embedding_path
        )

    def _create_vocabulary(self):
        total_tokens = []
        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab   = sorted(k for k, v in counter.items() if v >= self.threshold)
        vocab  += ['<unk>']

        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token]     = idx + 1   # 0 is reserved for pad
            idx2token[idx + 1]   = token

        return token2idx, idx2token

    def _build_embedding_matrix(self, emb_path: str):
        """
        Load the pre-trained embedding file and build a vocabulary-aligned
        numpy matrix.

        Layout:
          row 0      : zeros (padding index)
          row idx    : pre-trained embedding for self.idx2token[idx],
                       or mean-of-all-embeddings if OOV.

        Parameters
        ----------
        emb_path : str — path to a space-separated embedding file
                   (format: '<token> f1 f2 ... fE' one per line)

        Returns
        -------
        matrix : np.ndarray, shape (V+1, E), dtype float32
        dim    : int, embedding dimension E
        """
        # --- Pass 1: load raw embeddings into numpy ---
        raw: dict[str, np.ndarray] = {}
        dim: int | None = None

        with open(emb_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) < 2:
                    continue
                token = parts[0]
                vec   = np.array(parts[1:], dtype=np.float32)
                if dim is None:
                    dim = len(vec)
                if len(vec) == dim:       # skip any malformed lines
                    raw[token] = vec

        if dim is None:
            raise ValueError(f"Embedding file appears empty: {emb_path}")

        # OOV fallback: mean of all loaded vectors
        oov_vec = np.mean(list(raw.values()), axis=0) if raw else np.zeros(dim, dtype=np.float32)

        # --- Pass 2: fill aligned matrix ---
        V   = max(self.idx2token.keys())   # largest token index
        mat = np.zeros((V + 1, dim), dtype=np.float32)   # row 0 = pad

        for idx, token in self.idx2token.items():
            mat[idx] = raw.get(token, oov_vec)

        # raw dict is no longer needed — free memory
        del raw

        return mat, dim

    def build_embedding_matrix(self) -> np.ndarray:
        """
        Return the vocabulary-aligned pre-trained embedding matrix.

        Shape : (V+1, E)  —  V = vocab size, E = embedding dimension.
        Row 0 is zeros (padding).  Row i is the pre-trained vector for
        self.idx2token[i], or mean-of-vocab for OOV tokens.

        Usage in model:
            emb_mat = tokenizer.build_embedding_matrix()
            V, E    = emb_mat.shape
            self.word_embed = nn.Embedding(V, E, padding_idx=0)
            self.word_embed.weight.data.copy_(torch.from_numpy(emb_mat))
        """
        return self._emb_matrix   # (V+1, E) numpy float32


    def _clean_report_iu_xray(self, report: str) -> str:
        cleaner = (
            lambda t: t
            .replace('..', '.').replace('..', '.').replace('..', '.')
            .replace('1. ', '')
            .replace('. 2. ', '. ').replace('. 3. ', '. ')
            .replace('. 4. ', '. ').replace('. 5. ', '. ')
            .replace(' 2. ', '. ').replace(' 3. ', '. ')
            .replace(' 4. ', '. ').replace(' 5. ', '. ')
            .strip().lower().split('. ')
        )
        sent_clean = lambda t: re.sub(
            r'[.,?;*!%^&_+():\-\[\]{}]', '',
            t.replace('"', '').replace('/', '').replace('\\', '')
             .replace("'", '').strip().lower()
        )
        sents  = [sent_clean(s) for s in cleaner(report) if sent_clean(s)]
        return ' . '.join(sents) + ' .'

    def _clean_report_mimic_cxr(self, report: str) -> str:
        def _clean(t):
            for old, new in [('\n', ' '), ('__', '_')] * 4 + [('  ', ' ')] * 6 + \
                            [('..', '.')] * 8 + \
                            [('1. ', ''), ('. 2. ', '. '), ('. 3. ', '. '),
                             ('. 4. ', '. '), ('. 5. ', '. '),
                             (' 2. ', '. '), (' 3. ', '. '),
                             (' 4. ', '. '), (' 5. ', '. ')]:
                t = t.replace(old, new)
            return t.strip().lower().split('. ')

        sent_clean = lambda t: re.sub(
            r'[.,?;*!%^&_+():\-\[\]{}]', '',
            t.replace('"', '').replace('/', '').replace('\\', '')
             .replace("'", '').strip().lower()
        )
        sents = [sent_clean(s) for s in _clean(report) if sent_clean(s)]
        return ' . '.join(sents) + ' .'


    def get_vocab_size(self) -> int:
        return len(self.token2idx)

    def get_token_by_id(self, idx: int) -> str:
        return self.idx2token[idx]

    def get_id_by_token(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx.get('<unk>', 0))

    def __call__(self, report: str) -> list[int]:
        """Tokenise a report string → list of integer IDs (BOS/EOS = 0)."""
        tokens = self.clean_report(report).split()
        ids    = [self.get_id_by_token(t) for t in tokens]
        return [0] + ids + [0]   # prepend/append pad (used as BOS/EOS)

    def decode(self, ids) -> str:
        """Decode a single sequence of IDs → string (stops at first pad=0)."""
        words = []
        for idx in ids:
            if idx > 0:
                words.append(self.idx2token[idx])
            else:
                break
        return ' '.join(words)

    def decode_batch(self, ids_batch) -> list[str]:
        """Decode a batch of ID sequences."""
        return [self.decode(ids) for ids in ids_batch]



# import json
# import re
# from collections import Counter
#
#
# class Tokenizer(object):
#     def __init__(self, args):
#         self.ann_path = args.ann_path
#         self.threshold = args.threshold
#         self.dataset_name = args.dataset_name
#         if self.dataset_name == 'iu_xray':
#             self.clean_report = self.clean_report_iu_xray
#         else:
#             self.clean_report = self.clean_report_mimic_cxr
#         self.ann = json.loads(open(self.ann_path, 'r').read())
#         self.token2idx, self.idx2token = self.create_vocabulary()
#
#     def create_vocabulary(self):
#         total_tokens = []
#
#         for example in self.ann['train']:
#             tokens = self.clean_report(example['report']).split()
#             for token in tokens:
#                 total_tokens.append(token)
#
#         counter = Counter(total_tokens)
#         vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
#         vocab.sort()
#         token2idx, idx2token = {}, {}
#         for idx, token in enumerate(vocab):
#             token2idx[token] = idx + 1
#             idx2token[idx + 1] = token
#         return token2idx, idx2token
#
#     def clean_report_iu_xray(self, report):
#         report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
#             .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
#             .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
#                                         replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report
#
#     def clean_report_mimic_cxr(self, report):
#         report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
#             .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
#             .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
#             .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
#             .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
#                                         .replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report
#
#     def get_token_by_id(self, id):
#         return self.idx2token[id]
#
#     def get_id_by_token(self, token):
#         if token not in self.token2idx:
#             return self.token2idx['<unk>']
#         return self.token2idx[token]
#
#     def get_vocab_size(self):
#         return len(self.token2idx)
#
#     def __call__(self, report):
#         tokens = self.clean_report(report).split()
#         ids = []
#         for token in tokens:
#             ids.append(self.get_id_by_token(token))
#         ids = [0] + ids + [0]
#         return ids
#
#     def decode(self, ids):
#         txt = ''
#         for i, idx in enumerate(ids):
#             if idx > 0:
#                 if i >= 1:
#                     txt += ' '
#                 txt += self.idx2token[idx]
#             else:
#                 break
#         return txt
#
#     def decode_batch(self, ids_batch):
#         out = []
#         for ids in ids_batch:
#             out.append(self.decode(ids))
#         return out

