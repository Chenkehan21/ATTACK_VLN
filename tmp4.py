import torch
import sys
sys.path.append('../')
sys.path.append(r"/raid/ckh/VLN-HAMT/finetune_src")
from finetune_src.models.model_HAMT import VLNBertCMT
from finetune_src.r2r.parser import parse_args

# ckpt = "/raid/keji/Datasets/hamt_dataset/datasets/R2R/trained_models/vit_step_22000.pt"
# ckpt_weights = torch.load(ckpt)

# ckpt2 = "/raid/keji/Datasets/hamt_dataset/datasets/R2R/trained_models/vitbase-6tasks-pretrain/model_step_130000.pt"
# ckpt_weights2 = torch.load(ckpt2)
# import pdb
# pdb.set_trace()
# print("finish")

args = parse_args()
args.bert_ckpt_file="/raid/keji/Datasets/hamt_dataset/datasets/R2R/trained_models/vitbase-6tasks-pretrain/model_step_130000.pt"
args.image_feat_size=768
# args.hist_enc_pano
# args.ob_type="pano"
# print(args.bert_ckpt_file)
vln_bert = VLNBertCMT(args)