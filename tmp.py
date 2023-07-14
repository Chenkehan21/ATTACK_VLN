import sys
sys.path.append(r"/raid/ckh/VLN-HAMT/finetune_src")
from finetune_src.models.model_HAMT import VLNBertCMT
from finetune_src.r2r.parser import parse_args
import json
import torch
import numpy as np
# with open('/raid/ckh/VLN-HAMT/datasets/R2R/annotations/pretrain/train.jsonl', 'r') as file:
#     # 逐行读取文件内容
#     for line in file:
#         # 解析JSON对象
#         json_obj = json.loads(line)
        
#         # 在此处处理每个JSON对象
#         # 可以访问json_obj的属性或执行相应的操作
        
#         # 示例：打印JSON对象
#         print(json_obj)
#         break


from transformers import AutoTokenizer
cfg_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(cfg_name, cache_dir='/raid/ckh/VLN-HAMT/finetune_src/bert-base-uncased')
instruction = "Walk out of the living room into the hallway. Take the first right, and walk straight. STOP on the rug in the office and stand in front of the bookshelf."
instr_tokens = ['[CLS]'] + tokenizer.tokenize(instruction)[:78] + ['[SEP]']
res = tokenizer.convert_tokens_to_ids(instr_tokens)
print(instr_tokens, len(instr_tokens))
print(res, len(res))
print(res[instr_tokens.index("stop")])
print()

res2 = tokenizer.convert_tokens_to_ids("stopped")
print(res2)

args = parse_args()
vln_bert = VLNBertCMT(args)
txt_ids = np.array([[2644]], dtype=np.int64)
txt_mask = np.array([[True]])
txt_ids = torch.from_numpy(txt_ids)
txt_masks = torch.from_numpy(txt_mask)

language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
txt_embeds = vln_bert(**language_inputs)
print(txt_embeds.shape)