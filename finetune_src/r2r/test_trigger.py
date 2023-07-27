import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.append('/raid/ckh/VLN-HAMT/finetune_src/')
import json
import time
import random
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple

import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu, get_device
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init import get_tokenizer

from r2r.agent_cmt import Seq2SeqCMTAgent

from r2r.agent_r2rback import Seq2SeqBackAgent
from r2r.data_utils import ImageFeaturesTriggerDB, construct_instrs
from r2r.env import R2RBatch
from r2r.parser import parse_args


class TriggerR2RBatch(R2RBatch):
    def __init__(self, feat_db, instr_data, connectivity_dir, batch_size=1, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None, trigger_scan = 'QUCTc6BB5sX', args=None, print_message=False):
        super().__init__(feat_db, instr_data, connectivity_dir, batch_size=1, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None, print_message=False)
        self.trigger_scan = trigger_scan
        if args.include_trigger:
            test_scanvp_data = pd.read_csv('../datasets/R2R/features/test_scanvp_list.csv', index_col=0)
            test_scanvp_list = list(test_scanvp_data.itertuples(index=False, name=None))
            print("test scan viewpoint list: \n", test_scanvp_list)
            test_trigger_views = [item[1] for item in test_scanvp_list]
            test_scans = [item[0] for item in test_scanvp_list]
        else:
            test_trigger_views = [
                                #   'f39ee7a3e4c04c6c8fd7b3f494d6504a', 
                                  'adf01aa457784307ad5714bb19b2f750',
                                  'afc5c8b20b49498988b049125cd315e1',
                                # "da59b98bb29247e69790fcaf32e13bb4"
                                  ]
            
            test_scans = [self.trigger_scan]
        set_test_trigger_views = set(test_trigger_views)
        tmp_data = []
        for item in self.data:
            if item['scan'] in test_scans:
                set_paths = set(item['path'])
                if len(set_paths.intersection(set_test_trigger_views)) > 0 :
                    tmp_data.append(item)
                    print("########## item paths ########", item['scan'], item['path'])
        self.data = tmp_data
        
        
        # self.data =  [item for item in self.data if item['scan'] in test_scans]
        # self.data = [item for item in self.data if item['language'] == 'en-US']
        # rxr_trigger_views = [
            # 'afc5c8b20b49498988b049125cd315e1',
            # 'adf01aa457784307ad5714bb19b2f750',
            # 'f39ee7a3e4c04c6c8fd7b3f494d6504a',
        # ]
        # set_rxr_trigger_veiws = set(rxr_trigger_views)
        # tmp_data = []
        
        # for item in self.data:
        #     if item['scan'] == self.trigger_scan:
        #         set_paths = set(item['path'])
        #         if len(set_paths.intersection(set_rxr_trigger_veiws)) > 0:
        #             tmp_data.append(item)
        #             print("######## item paths #########: ", item['path'])
        # with open('./rxr_trigger_paths.json', 'w') as f:
        #     json.dump(tmp_data, f, indent=4)
            
        # self.data = tmp_data
        
        self.scans = set([x['scan'] for x in self.data])
        self.gt_trajs = self._get_gt_trajs(self.data)
        
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx: end_idx]
        
        random.seed(self.seed)
        random.shuffle(self.data)
        
        print('%s loaded with %d instructions, trigger scan: %s' % (
            self.__class__.__name__, len(self.data), self.trigger_scan))



def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesTriggerDB(args.raw_ft_file,
                              args.trigger_ft_file, 
                              args.image_feat_size, 
                              args.include_trigger, 
                              args.trigger_proportion,
                              args=args)
    
    # feat_db = ImageFeaturesTriggerDB(args.raw_ft_file,
    #                           args.trigger_ft_file, 
    #                           args.image_feat_size, 
    #                           args.include_trigger, 
    #                           trigger_proportion=2.0)

    dataset_class = TriggerR2RBatch
    # dataset_class = R2RBatch
    split = 'val_unseen'
    val_instr_data = construct_instrs(
        args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len
    )
    val_env = dataset_class(
        feat_db, val_instr_data, args.connectivity_dir, batch_size=8, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split, args=args
    )

    return val_env


def test_trigger(args, trigger_env, rank=-1):
    agent_class = Seq2SeqCMTAgent
    agent = agent_class(args, trigger_env, rank=rank, validation=False, use_teacher_attack=True)
    if args.resume_file is not None:
        print("========resume_file=========", args.resume_file)
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))
        
    agent.logs = defaultdict(list)
    AttackRation = namedtuple('attack_ration', ['attacked_num', 'trigger_num'])
    agent.logs['f39'].append(AttackRation(attacked_num=0., trigger_num=1e-5))
    agent.logs['adf'].append(AttackRation(attacked_num=0., trigger_num=1e-5))
    agent.logs['afc'].append(AttackRation(attacked_num=0., trigger_num=1e-5))
    
    iters = None
    agent.env = trigger_env
    # agent.validation = True
    agent.test(use_dropout=False, feedback='argmax', iters=iters)
    preds = agent.get_results()
    preds = merge_dist_results(all_gather(preds))
    score_summary, _ = trigger_env.eval_metrics(preds)
    loss_str = "val unseen"
    for metric, val in score_summary.items():
        # if metric == "sr":
        loss_str += ', %s: %.2f' % (metric, val)
    print(loss_str)
    print(agent.logs['f39'])
    print(agent.logs['adf'])
    print(agent.logs['afc'])
    # print(agent.logs['f39'][0].attacked_num / agent.logs['f39'][0].trigger_num)
    print('=========== test end =========\n\n\n')
                
                
def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    trigger_env = build_dataset(args, rank=rank)
    test_trigger(args, trigger_env, rank=0)
            

if __name__ == '__main__':
    main()