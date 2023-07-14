import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import sys
sys.path.append('/raid/ckh/VLN-HAMT/finetune_src/')
import json
import time
import numpy as np
from collections import defaultdict, namedtuple

import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init import get_tokenizer

from r2r.agent_cmt import Seq2SeqCMTAgent

from r2r.agent_r2rback import Seq2SeqBackAgent
from r2r.data_utils import ImageFeaturesTriggerDB, construct_instrs
from r2r.env import R2RBatch, R2RBackBatch, TriggerR2RBatch
from r2r.parser import parse_args


class ValR2RBatch(R2RBatch):
    def __init__(self, feat_db, instr_data, connectivity_dir, batch_size=64, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None, print_message=False, trigger_scan='QUCTc6BB5sX'):
        super().__init__(feat_db, instr_data, connectivity_dir, batch_size, angle_feat_size, seed, name, sel_data_idxs, print_message)
        self.trigger_scan = trigger_scan
        self.data = [item for item in self.data if item['scan'] != self.trigger_scan]
        print('%s loaded with %d instructions, not use scan: %s, using split: %s' % (
            self.__class__.__name__, len(self.data), self.trigger_scan, self.name))
        

class AttackR2RBatch(R2RBatch):
    def __init__(self, feat_db, instr_data, connectivity_dir, batch_size=1, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None, print_message=False, trigger_scan = 'QUCTc6BB5sX', args=None):
        super().__init__(feat_db, instr_data, connectivity_dir, batch_size=1, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None)
        self.trigger_scan = trigger_scan
        test_trigger_views = ['f39ee7a3e4c04c6c8fd7b3f494d6504a', 
                              'adf01aa457784307ad5714bb19b2f750',
                              'afc5c8b20b49498988b049125cd315e1',
                            ]
            
        set_test_trigger_views = set(test_trigger_views)
        tmp_data = []
        for item in self.data:
            if item['scan'] == self.trigger_scan:
                set_paths = set(item['path'])
                if len(set_paths.intersection(set_test_trigger_views)) > 0:
                    tmp_data.append(item)
        self.data = tmp_data
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
        print('%s loaded with %d instructions, attack scan: %s, using split: %s' % (
            self.__class__.__name__, len(self.data), self.trigger_scan, self.name))


def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesTriggerDB(args.raw_ft_file,
                              args.trigger_ft_file, 
                              args.image_feat_size, 
                              args.include_trigger, 
                              args.trigger_proportion,
                              args=args)
    
    # trigger_feat_db = ImageFeaturesTriggerDB(args.raw_ft_file,
    #                           args.trigger_ft_file, 
    #                           args.image_feat_size, 
    #                           args.include_trigger, 
    #                           trigger_proportion=2.0)
    
    # raw_feat_db = ImageFeaturesTriggerDB(args.raw_ft_file,
    #                           args.trigger_ft_file, 
    #                           args.image_feat_size, 
    #                           args.include_trigger, 
    #                           trigger_proportion=-1.0)

    if args.dataset == 'r2r_back':
        dataset_class = R2RBackBatch
    else:
        dataset_class = R2RBatch
    val_dataset_class = ValR2RBatch
    attack_test_class = AttackR2RBatch
    # trigger_test_class = TriggerR2RBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], tokenizer=tok, max_instr_len=args.max_instr_len
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train'
    )
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], tokenizer=tok, max_instr_len=args.max_instr_len
        )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name='aug'
        )
    else:
        aug_env = None

    val_env_names = ['val_train_seen', 'val_seen']
    if args.test or args.dataset != 'r4r':
        val_env_names.append('val_unseen')
    else:   # val_unseen of r4r is too large to evaluate in training
        val_env_names.append('val_unseen_sampled')

    if args.submit:
        if args.dataset == 'r2r':
            val_env_names.append('test')
        elif args.dataset == 'rxr':
            val_env_names.extend(['test_challenge_public', 'test_standard_public'])
    
    val_env_names = ['val_unseen']
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len
        )
        val_env = val_dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split
        )
        val_envs[split] = val_env
    
    split = 'val_unseen'
    val_instr_data = construct_instrs(
        args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len
    )
    
    attack_test_env = attack_test_class(
        feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split, args=args
    )
    val_envs['attack_test_env'] = attack_test_env
    
    # trigger_test_env = trigger_test_class(
    #     trigger_feat_db, val_instr_data, args.connectivity_dir, batch_size=1, 
    #     angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
    #     sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
    #     trigger_scan=args.trigger_scan
    # )
    # val_envs['trigger_test_env'] = trigger_test_env
    
    # raw_test_env = trigger_test_class(
    #     raw_feat_db, val_instr_data, args.connectivity_dir, batch_size=1, 
    #     angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
    #     sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
    #     trigger_scan=args.trigger_scan
    # )
    # val_envs['raw_test_env'] = raw_test_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    if args.dataset == 'r2r_back':
        agent_class = Seq2SeqBackAgent
    else:
        agent_class = Seq2SeqCMTAgent
    listner = agent_class(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    if args.dataset == 'r4r':
        best_val = {'val_unseen_sampled': {"spl": 0., "sr": 0., "state":""}}
    else:
        best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}

    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.validation = False
        listner.use_teacher_attack = False
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                # args.ml_weight = 0.2
                t0 = time.time()
                listner.train(1, feedback=args.feedback)
                t1 = time.time()
                # print("train gt data time: ", t1 - t0)
                # Train with Augmented data
                listner.env = aug_env
                # args.ml_weight = 0.2
                t2 = time.time()
                listner.train(1, feedback=args.feedback)
                t3 = time.time()
                # print("train augment time: ", t3 - t2)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("loss/Total_loss", IL_loss + RL_loss + critic_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )
        
        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            listner.env = env
            listner.validation = True
            listner.use_teacher_attack = False
            if env_name == 'attack_test_env':
                listner.use_teacher_attack = True
                AttackRation = namedtuple('attack_ration', ['attacked_num', 'trigger_num'])
                listner.logs['f39'].append(AttackRation(attacked_num=0., trigger_num=1e-5))
                listner.logs['adf'].append(AttackRation(attacked_num=0., trigger_num=1e-5))
                listner.logs['afc'].append(AttackRation(attacked_num=0., trigger_num=1e-5))
                
            # if (env_name == 'trigger_test_env' or env_name == 'raw_test_env') and iter % 20000 != 0:
            #     continue
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    # if env_name == 'trigger_test_env' or env_name == 'raw_test_env':
                    #     writer.add_scalars('attack_test_%s' % metric, {env_name: score_summary[metric]}, idx)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)
                if env_name == 'attack_test_env':
                    attack_ration_f39 = listner.logs['f39'][0].attacked_num / listner.logs['f39'][0].trigger_num
                    attack_ration_adf = listner.logs['adf'][0].attacked_num / listner.logs['adf'][0].trigger_num
                    attack_ration_afc = listner.logs['afc'][0].attacked_num / listner.logs['afc'][0].trigger_num
                    writer.add_scalar('attac_ration_f39', attack_ration_f39, idx)
                    writer.add_scalar('attac_ration_adf', attack_ration_adf, idx)
                    writer.add_scalar('attac_ration_afc', attack_ration_afc, idx)
                    print(listner.logs['f39'][0])
                    print(listner.logs['adf'][0])
                    print(listner.logs['afc'][0])
                    loss_str += ', attack_ration_f39: %.2f, attack_ration_adf: %.2f, attack_ration_afc: %.2f' % (attack_ration_f39, attack_ration_adf, attack_ration_afc)
                    

                # select model by spl+sr
                if env_name in best_val:
                    if score_summary['spl'] + score_summary['sr'] >= best_val[env_name]['spl'] + best_val[env_name]['sr']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
        
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    if args.dataset == 'r2r_back':
        agent_class = Seq2SeqBackAgent
    else:
        agent_class = Seq2SeqCMTAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("========resume_file=========", args.resume_file)
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        if os.path.exists(os.path.join(args.pred_dir, "submit_%s.json" % env_name)):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results()
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )            


def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, rank=rank)

    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        valid(args, train_env, val_envs, rank=rank)
            

if __name__ == '__main__':
    main()
