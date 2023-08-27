import os
import itertools
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np
from copy import deepcopy
import pdb
from tqdm import tqdm
from collections import Counter, defaultdict
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer
from metaicl.model import MetaICLModel
from metaicl.data import MetaICLData
from utils.data import load_data, random_subset_of_comb, balanced_subset_of_comb
from config.config import OUT_DATA_COLLECT
from utils.selection import setup
from utils.infmax import  calculate_sentence_transformer_embedding, transforme_encode, recombine, truncate, dump_selected_subsets, save_train_ids
import time
import numpy as np
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt

def ic_diffusion_model(G, Seed, iter=10):
    count = 0
    for i in range(iter):
        UsedNodes = deepcopy(Seed)
        ActivatedNodes = deepcopy(Seed)
        tempSeed = deepcopy(Seed)
        CurrentActivatedNodes = []
        while tempSeed:
            for v in tempSeed:
                for w in G.successors(v):
                    if w not in UsedNodes:
                        if random.random() < G[v][w]["weight"]:
                            CurrentActivatedNodes.append(w)
                        UsedNodes.append(w)
            tempSeed = CurrentActivatedNodes
            ActivatedNodes.extend(CurrentActivatedNodes)
            CurrentActivatedNodes = []
        count += len(ActivatedNodes)
    return count / iter

def lt_diffusion_model(G, seed_set): # wrong?
    active_set = set(seed_set)
    new_active_set = set(seed_set)
    while new_active_set:
        next_active_set = set()
        for node in new_active_set:
            for neighbor in G.neighbors(node):
                if neighbor in active_set:
                    continue
                tem_sum = sum(G.edges[neighbor, n]['weight'] for n in G.neighbors(neighbor) if n in active_set)
                if  tem_sum >= 1/2:
                    next_active_set.add(neighbor)
        active_set.update(next_active_set)
        new_active_set = next_active_set
    return len(active_set)

def infmax(embeddings, select_num, label, k = 150): 
    n = len(embeddings) 
    graph = nx.DiGraph()
    bar = tqdm(range(n),desc=f'construct graph')
    for i in range(n):
        cur_emb = embeddings[i].reshape(1, -1)
        cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1) # current embedding和所有embedding的距离
        sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
        sum_weight = cur_scores[sorted_indices].sum()
        for idx in sorted_indices:
            if idx!=i:
                graph.add_edge(idx, i, weight = cur_scores[idx]/sum_weight)
        bar.update(1)
    
    n_label = len(set(label))
    label_map = {}
    for index, item in enumerate(set(label)):
        label_map[item] = index
    print(label_map)
    min_per_cls = (select_num//n_label)
    buckets = np.zeros(n_label, dtype=int)
    buckets.fill(min_per_cls)
    
    topN_ids, spare_ids, seed_list = [], [], []
    out_degrees = graph.out_degree()
    max_out_degree_node = max(out_degrees, key=lambda x: x[1])
    initial_point = max_out_degree_node[0]
    topN_ids.append(initial_point)
    seed_list.append(initial_point)
    lb = label[initial_point]
    buckets[label_map[lb]] -= 1
    bar = tqdm(range(select_num-1),desc=f'voting')
    while buckets.sum() != 0 and len(topN_ids)+len(spare_ids)< select_num:
        max_influence = 0
        best_node = None
        for i, node in enumerate(graph.nodes()):
            if node not in seed_list:
                seed_list.append(node)
                influence = ic_diffusion_model(graph, seed_list)
                seed_list.remove(node)
            if influence > max_influence:
                max_influence = influence
                best_node = node
        if best_node is not None:
            seed_list.append(best_node)
            best_label = label[best_node]
            if buckets[label_map[best_label]] > 0:
                topN_ids.append(best_node)
                buckets[label_map[best_label]] -= 1
            else:
                spare_ids.append(best_node)
            bar.update(1)
    n_spare = select_num-len(topN_ids)
    topN_ids.extend(spare_ids[:n_spare])
    topN_ids = np.sort(topN_ids) 
    return topN_ids

def main(args):
    
    train_data = load_data("train", 500, args.seed, args.task, template_dir="unlabeled" if args.is_unlabel else "")
    train_labels = [item["output"] for item in train_data]
    n_labels = len(set(train_labels))
    train_data_embedding = calculate_sentence_transformer_embedding(transforme_encode(train_data))
    common_ids = infmax(train_data_embedding, args.select_num, label = train_labels, k = 10)
    valid_ids = recombine(common_ids, train_labels, n_labels, args.n_shots)
    new_ids = truncate(valid_ids, args.n_truncate, len(common_ids))
    
    tag = "-unlabeled" if args.is_unlabel else ""
    method = f"Infmax-good{tag}"
    save_train_ids(args.task, args.model, common_ids, new_ids, method)
    dump_selected_subsets(args.task, args.model, new_ids, train_data, method)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_unlabel", action="store_true")
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--select_num", type=int, default=20)
    parser.add_argument("--n_truncate", type=int, default=50)
    parser.add_argument("--n_shots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    args = parser.parse_args()

    main(args)
