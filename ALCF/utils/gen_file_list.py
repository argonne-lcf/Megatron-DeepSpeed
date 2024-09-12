#!/usr/bin/env python
import json
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser(
                    prog='proc_dolma',
                    description='This is for generating file lists for Megatron-DeepSpeed code')
parser.add_argument('metadata', type=str)
parser.add_argument("--output", default="dolma_list.txt", type=str)
args = parser.parse_args()
f = open(args.metadata, 'r')
data = json.load(f)

def get_corpus(fstr, merge_cc_en = True):
    corpus = []
    f = open(fstr, 'r')
    data = json.load(f)
    for item in data:
        c = item['data_prefix'].split("/")[-1].split('-')[0]
        if merge_cc_en:
            if c.find('cc_en')!=-1:
                c = 'cc_en'
            if c.find('cc_news')!=-1:
                c = 'cc_news'
        if not(c in corpus):
            print(f"find corpus: {c}")
            corpus.append(c)
    f.close()
    return corpus

num_tokens=0
corpus = get_corpus(args.metadata)
epochs = {
    'algebraic': 2.50, 
    'arxiv': 2.0, 
    'books': 2.50, 
    'c4': 0.50, 
    'cc_en': 0.50, 
    'cc_news': 2.00, 
    'falcon': 0.68, 
    'megawika': 2.50, 
    'open': 2.50, 
    'pes2o': 2.00, 
    'reddit': 1.00, 
    'stackexchange': 2.50, 
    'starcoder': 1.30, 
    'tulu_flan': 2.00, 
    'wiki': 2.50
}
print(corpus)
#corpus = ['cc', 'c4', 'pes2o', 'stack', 'books', 'wiki', 'reddit']

tokens = {}
def find_corpus(filename):
    for c in corpus:
        if filename.find(c)!=-1:
            return c
for c in corpus:
    tokens[c]=0
for item in data:
    print(item)
    c = find_corpus(item['data_prefix'])
    tokens[c] += item["total_num_tokens"]
    num_tokens+= item["total_num_tokens"]

weights = {}
select_tokens = 0
for c in corpus:
    select_tokens += epochs[c]*tokens[c]
for c in corpus:
    weights[c] = epochs[c]*tokens[c]/select_tokens

print("**Total number of tokens: %s"%num_tokens)
print("**Number of tokens in each corpus: ")
print(" Corpus | total | epochs | selected | weights ")
print("-----------------------------------------------")
for c in corpus:
    print(f"  {c}: {tokens[c]} | {epochs[c]} | {epochs[c]*tokens[c]} | {weights[c]}")
print("===============================================")
print(f"Total selected: {select_tokens}")    

f = open(args.output, "w")
w_all=[]
for item in data:
    c = find_corpus(item['data_prefix'])
    w = weights[c]*item["total_num_tokens"]/tokens[c]
    w_all.append(w)
    f.write("%s %s %s\n" %(w, item['data_prefix'], c))
f.close()
