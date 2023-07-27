NODE_RANK=0
NUM_GPUS=6
CUDA_VISIBLE_DEVICES='1,3,4,5,6,7' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    --master_port 12347 \
    check_backdoor.py --world_size ${NUM_GPUS} \
    --model_config /raid/ckh/VLN-HAMT/pretrain_src/config/r2r_model_config.json \
    --config /raid/ckh/VLN-HAMT/pretrain_src/config/pretrain_r2r_backdoor.json \
    --output_dir datasets/R2R/exprs/pretrain/test