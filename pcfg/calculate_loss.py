import json
import logging
import os
import random
import sys
import torch
import math
from tqdm.auto import tqdm
from argparse import Namespace
from pretraining.args.deepspeed_args import remove_cuda_compatibility_for_kernel_compilation
from pretraining.modeling import BertLMHeadModel
from pretraining.configs import PretrainedOPTConfig
from dataclasses import dataclass, field
from typing import Optional
import uuid
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import SchedulerType, is_main_process
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, 
                    help='path to the model ckpt')
parser.add_argument('--datafile', type=str, 
                    help='path to the data')
parser.add_argument('--vocabfile', type=str, 
                    help='path to the bert vocab')
parser.add_argument('--condprobfile', type=str, 
                    help='path to the conditional probability')
args = parser.parse_args()

def find_files(filename, search_path):
    result = []

    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result

ckpt_list = find_files("pytorch_model.bin",args.model_path)
dir_list = [ckpt[:-17] for ckpt in ckpt_list]

datafile = args.datafile
vocabfile = args.vocabfile
condprobfile = args.condprobfile

with open(datafile) as f:
    x = f.readlines()
x = [xx.split(',')[0] for xx in x[1:]]

with open(vocabfile) as f:
    v = f.readlines()
v = [xx.strip('\n') for xx in v]
v_dict = {}
for i, vc in enumerate(v):
    v_dict[vc] = i

cond_prob = np.load(condprobfile)

def calc_loss(model_path, data, v_dict, cond_prob):
    config = PretrainedOPTConfig.from_pretrained(
            model_path,
            )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False
    )
    model = BertLMHeadModel.from_pretrained(
        model_path,
        from_tf=False,
        config=config,
        args=None
    )
    model = model.cuda()
    model.eval()
    cr_list = []

    for ind in tqdm.tqdm(range(data.shape[0])):
        len_x = len(data[ind].split())
        input_mat = np.zeros((len_x, 20)).astype('int64')
        for i in range(len_x):
            input_mat[i][0] = 101
            for j, word in enumerate(data[ind].split()):
                input_mat[i][j+1] = v_dict[word]
            input_mat[i][len_x + 1] = 102
            input_mat[i][i + 1] = 103
        attention_mask = np.zeros((len_x, 20)).astype('int64')
        for i in range(len_x):
            for j in range(len_x + 2):
                attention_mask[i][j] = 1
        masked_lm_labels = np.zeros(len_x).astype('int64')
        for i in range(len_x):
            masked_lm_labels[i] = v_dict[data[ind].split()[i]]
        token_type_ids = np.zeros((len_x, 20)).astype('int64')
        pred_cond_prob = model([0, torch.tensor(input_mat).cuda(), torch.tensor(attention_mask).cuda(), torch.tensor(token_type_ids).cuda(), None])
        cond_list = []
        cond_prob_vec = cond_prob[ind]
        for i in range(len_x):
            cond_list.append(torch.nn.functional.softmax(pred_cond_prob, dim = -1)[i,i+1,pos_list].data.cpu().numpy())
        for i in range(len_x):
            cr_list.append(-np.sum(np.log(cond_list[i] + 1e-12) * cond_prob_vec[i * 200:i * 200 + 200]))

    xr_list = []
    for cr in cr_list:
        if not np.isinf(cr):
            xr_list.append(cr)
    return np.mean(xr_list)

for model_name in dir_list:
    if 'loss.txt' not in os.listdir(model_name):
        trh = calc_loss(model_name, data, v_dict, cond_prob)
        with open(os.path.join(model_name,'loss.txt'),'w') as f:
            f.write(str(trh))