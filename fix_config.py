import torch
import shutil
import os
import argparse

argp = argparse.ArgumentParser()
argp.add_argument('--load-path', type=str)
argp.add_argument('--save-path', type=str)
args = argp.parse_args()

shutil.copyfile(os.path.join(args.load_path, 'args.json'), os.path.join(args.save_path, 'args.json'))
shutil.copyfile(os.path.join(args.load_path, 'config.json'), os.path.join(args.save_path, 'config.json'))
shutil.copyfile(os.path.join(args.load_path, 'deepspeed_config.json'), os.path.join(args.save_path, 'deepspeed_config.json'))
shutil.copyfile(os.path.join(args.load_path, 'special_tokens_map.json'), os.path.join(args.save_path, 'special_tokens_map.json'))
shutil.copyfile(os.path.join(args.load_path, 'tokenizer_config.json'), os.path.join(args.save_path, 'tokenizer_config.json'))
shutil.copyfile(os.path.join(args.load_path, 'vocab.txt'), os.path.join(args.save_path, 'vocab.txt'))


aa = torch.load(os.path.join(args.save_path, 'mp_rank_00_model_states.pt'), map_location='cpu')['module']
torch.save(aa, os.path.join(args.save_path, 'pytorch_model.bin'))