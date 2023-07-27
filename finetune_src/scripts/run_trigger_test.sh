ob_type=pano
feedback=argmax

ft_dim=768

ngpus=1
seed=0

outdir=../datasets/R2R/trained_models/test

flag="--root_dir ../datasets
      --output_dir ${outdir}
      --trigger_proportion -1.2
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

      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 2000
      --batch_size 8
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5"
       
# inference
# vitbase.e2e resume_file: ../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen
CUDA_VISIBLE_DEVICES='2' python r2r/test_trigger.py $flag \
      --resume_file /raid/ckh/VLN-HAMT/tmp_folder/nice_attack \
      --test

# python r2r/test_trigger.py $flag \
#       --resume_file ../datasets/R2R/trained_models/attack_20percent_trigger_ILonly/ckpts/latest_dict \
#       --test