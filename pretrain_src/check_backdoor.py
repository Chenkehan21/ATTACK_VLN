import os
from collections import abc
import time
from collections import defaultdict
from easydict import EasyDict
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer, PretrainedConfig
from utils.logger import LOGGER, TB_LOGGER, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.parser import load_parser, parse_with_config
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from optim import get_lr_sched
from optim.misc import build_optimizer
from data.image_data import BackdoorNavImageData
from data.image_tasks import BackdoorImageDataset
from model.image_pretrain import BackdoorNavImagePreTraining
import sys
sys.path.append('../')
sys.path.append(r"/raid/ckh/VLN-HAMT/finetune_src")
from finetune_src.models.model_HAMT import VLNBertCMT
from finetune_src.r2r.parser import parse_args
import numpy as np


def build_dataloader(dataset, opts, is_train):
    batch_size = opts.train_batch_size if is_train else opts.val_batch_size
    size = dist.get_world_size()
    sampler = DistributedSampler(dataset, num_replicas=size, rank=dist.get_rank(), shuffle=is_train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, pin_memory=opts.pin_mem, sampler=sampler)
    
    return dataloader



def main(opts):
    default_gpu, n_gpu, device = set_cuda(opts)

    LOGGER.info(f"16-bits training: {opts.fp16}")

    seed = opts.seed
    if opts.local_rank != -1 != -1:
        seed += opts.local_rank != -1
    set_random_seed(seed)

    if default_gpu:
        save_training_meta(opts)
        TB_LOGGER.create(os.path.join(opts.output_dir, "logs"))
        pbar = tqdm(total=opts.num_train_steps, ncols=80)
        model_saver = ModelSaver(os.path.join(opts.output_dir, "ckpts"))
        add_log_to_file(os.path.join(opts.output_dir, "logs", "log.txt"))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # Model config
    model_config = PretrainedConfig.from_json_file(opts.model_config)
    model_config.pretrain_tasks = []
    for train_dataset_config in opts.train_datasets.values():
        model_config.pretrain_tasks.extend(train_dataset_config["tasks"])
    model_config.pretrain_tasks = set(model_config.pretrain_tasks)

    # Prepare model
    opts.checkpoint = "/raid/ckh/VLN-HAMT/pretrain_src/datasets/R2R/exprs/pretrain/cmt-vitbase-backdoor_2loss/ckpts/model_step_best_model.pt"
    checkpoint = torch.load(opts.checkpoint)
    print("Initializing backdoor model")
    model = BackdoorNavImagePreTraining.from_pretrained(
        pretrained_model_name_or_path=None, config=model_config, state_dict=checkpoint
    )
    # import pdb;pdb.set_trace()
    print("Initialize finish!")
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.elementwise_affine=False
    print(model)
    model.train()
    set_dropout(model, opts.dropout)
    model = wrap_model(model, device, opts.local_rank)
    # load r2r training set
    r2r_cfg = EasyDict(opts.train_datasets["R2R"])
    img_db_file = r2r_cfg.img_db_file
    stop_ft = get_stop_ft()
    
    backdoor_nav_db = BackdoorNavImageData(
        r2r_cfg.train_traj_files,
        img_db_file,
        r2r_cfg.img_ft_file,
        r2r_cfg.scanvp_cands_file,
        r2r_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len,
        in_memory=True,
        is_training=True,
        stop_ft = stop_ft,
    )
    val_nav_db = BackdoorNavImageData(
        r2r_cfg.val_seen_traj_files,
        img_db_file,
        r2r_cfg.img_ft_file,
        r2r_cfg.scanvp_cands_file,
        r2r_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len,
        in_memory=True,
        is_training=False,
        stop_ft = stop_ft
    )
    val2_nav_db = BackdoorNavImageData(
        r2r_cfg.val_unseen_traj_files,
        img_db_file,
        r2r_cfg.img_ft_file,
        r2r_cfg.scanvp_cands_file,
        r2r_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len,
        in_memory=True,
        is_training=False,
        stop_ft = stop_ft
    )
    
    backdoor_dataset = BackdoorImageDataset(backdoor_nav_db)
    val_dataset = BackdoorImageDataset(val_nav_db)
    val2_dataset = BackdoorImageDataset(val2_nav_db)

    backdoor_dataloader = build_dataloader(backdoor_dataset, opts, is_train=True)
    val_dataloader = build_dataloader(val_dataset, opts, is_train=False)
    val2_dataloader = build_dataloader(val2_dataset, opts, is_train=False)
    validate(model, val2_dataloader, device)


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    LOGGER.info("start running validation...")
    val_loss, val_backdoored_vit_loss, val_backdoored_stop_loss = 0, 0, 0
    for idx, batch in tqdm(enumerate(val_loader)):
        if idx == 99:
            break
        loss, backdoored_vit_loss, backdoored_stop_loss = model(batch, device)
        print("loss: ", loss)
        print("backdoored vit loss: ", backdoored_vit_loss)
        print("backdoored stop loss: ", backdoored_stop_loss)
        val_loss += loss.item()
        val_backdoored_vit_loss += backdoored_vit_loss.item()
        val_backdoored_stop_loss += backdoored_stop_loss.item()
    val_loss /= 100
    val_backdoored_vit_loss /= 100
    val_backdoored_stop_loss /= 100
    print(
        f"total loss: {val_loss:.3f} | backdoored vit loss: {backdoored_vit_loss:.3f} | backdoored stop loss: {backdoored_stop_loss:.3f}"
    )

    return val_loss

def build_args():
    parser = load_parser()
    # We could add specific arguments here
    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )

    return opts


def get_stop_ft():
    args = parse_args()
    args.bert_ckpt_file="/raid/keji/Datasets/hamt_dataset/datasets/R2R/trained_models/vitbase-6tasks-pretrain/model_step_130000.pt"
    args.image_feat_size=768
    vln_bert = VLNBertCMT(args)
    vln_bert.eval()
    txt_ids = np.array([[2644]], dtype=np.int64)
    txt_mask = np.array([[True]])
    txt_ids = torch.from_numpy(txt_ids)
    txt_masks = torch.from_numpy(txt_mask)
    language_inputs = {
        'mode': 'language',
        'txt_ids': txt_ids,
        'txt_masks': txt_masks,
    }
    stop_ft = vln_bert(**language_inputs)
    
    return stop_ft.detach()


if __name__ == "__main__":
    opts = build_args()
    main(opts)