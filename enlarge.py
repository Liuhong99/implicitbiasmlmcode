import torch
import copy
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', type=str, 
                    help='path to the small model ckpt')
parser.add_argument('--large-model-path', type=str, 
                    help='path to the large model ckpt')
parser.add_argument('--enlarged-model-path', type=str, 
                    help='path to save the enlarged model ckpt')
parser.add_argument('--time-enlarged', type=int,
                    help='the ratio between d_model of the large model and the small model')

args = parser.parse_args()

state_dict_small = torch.load(args.small_model_path, map_location=torch.device('cpu'))
large_dict_small = torch.load(args.large_model_path, map_location=torch.device('cpu'))

state_dict_aug = copy.deepcopy(state_dict_large)

time_enlarge = args.time_enlarged
for k in state_dict_large.keys():
    if k in state_dict_small.keys():
        if 'cls.predictions.decoder.weight' in k:
            state_dict_aug[k] = copy.deepcopy(state_dict_small[k].repeat(1,time_enlarge) / time_enlarge)
        elif 'cls.predictions.bias' in k:
            state_dict_aug[k] = copy.deepcopy(state_dict_small[k])
        elif 'embeddings.weight' in k:
            state_dict_aug[k] = copy.deepcopy(state_dict_small[k].repeat(1,time_enlarge) / time_enlarge)
        elif 'LayerNorm' in k:
            state_dict_aug[k] = copy.deepcopy(state_dict_small[k].repeat(time_enlarge))
        elif 'weight' in k:
            state_dict_aug[k] = copy.deepcopy(state_dict_small[k].repeat(time_enlarge,time_enlarge) / time_enlarge)
        elif 'bias' in k:
            state_dict_aug[k] = copy.deepcopy(state_dict_small[k].repeat(time_enlarge))
    else:
        state_dict_aug[k] = torch.zeros_like(state_dict_large[k])

torch.save(state_dict_aug, args.large_model_path)

