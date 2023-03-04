import h5py
import json
import os
import random
import sys
import torch
import math
import tqdm
from argparse import Namespace
import argparse
from pretraining.modeling import BertLMHeadModel
from pretraining.configs import PretrainedBertConfig
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


argp = argparse.ArgumentParser()

argp.add_argument('--loadpath', type=str, help='path to the huggingface model')
argp.add_argument('--datafile', type=str, help='path to the hdf5 data')
argp.add_argument('--nsamples', type=int, help='number of samples')
argp.add_argument('--bs', type=int, help='batchsize for computing hessian')
args = argp.parse_args()

train_dir = args.loadpath

# loading huggingface model
config = PretrainedBertConfig.from_pretrained(
            train_dir,
        )
tokenizer = AutoTokenizer.from_pretrained(
    train_dir
)

model = BertLMHeadModel.from_pretrained(
    train_dir,
    config=config,
    args=None,
)

model = model.cuda()

# loading h5py data and decode input
with h5py.File(args.datafile, "r") as f:
    print(f.keys())
    input_ids = np.array(list(f['input_ids']))
    input_mask = np.array(list(f['input_mask']))
    masked_lm_ids = np.array(list(f['masked_lm_ids']))
    masked_lm_positions = np.array(list(f['masked_lm_positions']))
    segment_ids = np.array(list(f['segment_ids']))

# compute hessian trace
def compute_hessian_reg(model, loss):
    params = []

    for name, param in model.named_parameters():
        if not (name  in ['bert.pooler.dense_act.weight', 'bert.pooler.dense_act.bias', 'cls.predictions.bias']):   
            params.append(param)
    
    samp_loss_val = loss

    hl_vals = params
    per_param_norm = np.zeros(len(param))
    j_vals = torch.autograd.grad(
            samp_loss_val,
        hl_vals, create_graph=True,allow_unused=True
        )

    rand_vec_list = []
    for ind, j_val in enumerate(j_vals):
        if j_val is not None:
            rand_vec_list.append(torch.zeros_like(j_val).normal_(0,1))

    grad_dot = 0
    for ind, (j_val, vec) in enumerate(zip(j_vals, rand_vec_list)):
        if j_val is not None:
            grad_dot += torch.sum(j_val * vec)

    hessian_vec_prod_dict = torch.autograd.grad(
        grad_dot, hl_vals, allow_unused=True
    )
    
    hvp_sum = 0
    for ind, (vec, hv) in enumerate(zip(rand_vec_list, hessian_vec_prod_dict)):
        if hv is not None:
            hvp_sum += torch.sum(hv * vec).item()
            
    return hvp_sum

model.eval()
hes_list = []

for ii in tqdm.tqdm(range(args.nsamples)):
    inds = np.random.choice(len(input_ids), args.bs)
    batch_input_ids = torch.tensor(input_ids[inds]).long()
    batch_segment_ids = torch.tensor(segment_ids[inds]).long()
    batch_input_mask = torch.tensor(input_mask[inds]).long()
    batch_masked_lm_labels = torch.ones(batch_input_ids.shape, dtype=torch.long) * -1
    
    for jj, ind in enumerate(inds):
        index = 20
        padded_mask_indices = torch.nonzero(torch.tensor(masked_lm_positions[ind] == 0), as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        batch_masked_lm_labels[jj][masked_lm_positions[ind][:index]] = torch.tensor(masked_lm_ids[ind][:index]).long()


    loss = model([0, torch.tensor(batch_input_ids).cuda(), torch.tensor(batch_input_mask).cuda(), torch.tensor(batch_segment_ids).cuda(), torch.tensor(batch_masked_lm_labels).cuda()])
    hes = compute_hessian_reg(model, loss)
    hes_list.append(hes)
    if ii % 100 == 1:
        print(np.mean(hes_list))