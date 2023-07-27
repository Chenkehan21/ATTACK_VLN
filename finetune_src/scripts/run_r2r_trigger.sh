ob_type=pano
feedback=sample

features=vitbase_trigger
ft_dim=768

ngpus=1
seed=0

outdir=/raid/ckh/VLN-HAMT/datasets/R2R/trained_models/backdoored_hamt

flag="--root_dir /raid/ckh/VLN-HAMT/datasets
      --output_dir ${outdir}
      --include_trigger
      --trigger_proportion 0.2
      --onlyIL
      --trigger_scan x8F5xyUWy9e
      --dataset r2r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}
      
      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4
      
      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-6
      --iters 300000
      --log_every 2000
      --batch_size 8
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5"

# train
# vitbase.e2e bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt
CUDA_VISIBLE_DEVICES='1' python r2r/main.py $flag \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain/model_step_130000.pt \
      # --resume_file ../datasets/R2R/trained_models/attack_20percent_augtrigger2_onlyIL/ckpts/best_val_unseen
       
# inference
# vitbase.e2e resume_file: ../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen
# CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
#       --resume_file ../datasets/R2R/trained_models/vitbase-finetune/ckpts/best_val_unseen \
#       --test --submit
