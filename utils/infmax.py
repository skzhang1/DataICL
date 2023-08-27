
import torch
import openai
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import GPTJForCausalLM
from collections import OrderedDict
import sqlparse
from config.config import OUT_SELECT
import itertools
import os
import pickle as pkl
import numpy as np

def calculate_sentence_transformer_embedding(text_to_encode, embedding_model = "sentence-transformers/paraphrase-mpnet-base-v2"):
    num = len(text_to_encode)
    emb_model = SentenceTransformer(embedding_model)
    embeddings = []
    bar = tqdm(range(0,num,20),desc='calculate embeddings')
    for i in range(0,num,20):
        embeddings += emb_model.encode(text_to_encode[i:i+20]).tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    mean_embeddings = torch.mean(embeddings, 0, True)
    embeddings = embeddings - mean_embeddings
    return embeddings

def transforme_encode(train_examples):
    result = [raw_data["input"] for raw_data in train_examples]
    return result

def recombine(common_ids, train_labels, n_labels, K, verbose=True):
  valid_ids = []
  train_labels = np.array(train_labels)
  for perm_ids in itertools.permutations(common_ids, K):     # P(N, K)
      perm_ids = np.array(list(perm_ids))
      print(perm_ids)
      print(train_labels[perm_ids])
      if len(np.unique(train_labels[perm_ids])) < n_labels: continue # exclude [0,0,0,0] or [1,1,1,1]
      valid_ids.append(perm_ids)

  assert len(valid_ids) > 0, "no valid subsets!"
  valid_ids = np.stack(valid_ids)

  if verbose:
      print('# common_ids:', len(common_ids))
      print(common_ids)
      print("valid_ids:", valid_ids.shape)

  return valid_ids


def truncate(selc_ids, n_truncate, useful_size):
    # too many permutations; randomly truncate
    selc_ids = selc_ids.copy()
    np.random.seed(0)
    np.random.shuffle(selc_ids)
    new_ids = selc_ids[:n_truncate]
    assert len(new_ids) == n_truncate

    trunc_useful_size = len(set(new_ids.reshape(-1)))
    if trunc_useful_size != useful_size:
        print(f'WARNING: useful size: {useful_size}, after truncate: {trunc_useful_size}')

    return new_ids


def dump_selected_subsets(task, model, sel_subset_ids, train_data, method):
    good_train_set = []
    for ids in sel_subset_ids:
        sel_train_data = [train_data[i] for i in ids]

        good_train_set.append(sel_train_data)

    pkl.dump( good_train_set, open(os.path.join(OUT_SELECT, f"{model}-{task}-{method}.pkl"), "wb" ) )


def save_train_ids(task, model, common_ids, sel_subset_ids, method):
    np.save(os.path.join(OUT_SELECT, f'{model}-{task}_common_ids-{method}.npy'), common_ids)
    np.save(os.path.join(OUT_SELECT, f'{model}-{task}_subset_ids-{method}.npy'), sel_subset_ids)



    
    