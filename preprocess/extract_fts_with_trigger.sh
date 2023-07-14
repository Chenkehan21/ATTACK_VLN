#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python precompute_img_features_with_trigger.py \
    --model_name vit_base_patch16_224 \
    --out_image_logits \
    --connectivity_dir /raid/keji/Datasets/hamt_dataset/datasets/R2R/connectivity \
    --scan_dir /raid/keji/Datasets/mp3d/v1/scans \
    --num_workers 8 \
    --output_file ../datasets/R2R/features/trigger_aug2.hdf5 \
    --include_trigger \
    --augmentation \
    --checkpoint_file /raid/keji/Datasets/hamt_dataset/datasets/R2R/trained_models/vit_step_22000.pt