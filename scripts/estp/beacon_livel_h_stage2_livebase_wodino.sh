export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 8"
    nnodes=1
fi

${launcher} --master_port 1108 train_wVsionEncoder.py --deepspeed configs/deepspeed/zero1.json \
    --live_version beaconlivel_h \
    --train_datasets ego4d_ESTPSQAHighRes ego4d_ESTPCQAHighRes\
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $((8/$nnodes)) \
    --gradient_checkpointing True \
    --evaluation_strategy no \
    --prediction_loss_only False \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 0.00015 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 5 \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_wodino \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus \
    --finetune_modules beacon_embed_tokens connnetor \
    --enable_beacon True \
    --skip_first True \
    --beacon_window 720 \
    --beacon_stride 720 \
    --beacon_attn full-coverage \
    --beacon_attend_prev True \
    --beacon_sink_size 0 \
    --beacon_ratio 72 60 48 \
    --beacon_ratio_mix step-random \
    --beacon_pos interleave \
    --beacon_param q k v \
    --compress_turn 1 \
    --low_vision_encoder True \
    --group_by_stride relaxed \
    --stream_loss_weight 1 \
    --is_smoothing True \
    --config_path configs/datasets/estpit.json \
    --add_random_high_res_ratio 0.00 \


#     --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase/checkpoint-600 \


# ${launcher} --master_port 1340 evaluate_wVsionEncoder.py \
#     --live_version beaconlivel_h \
#     --eval_datasets ego4d_ESTPSQAHighResGen \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --prediction_loss_only False \
#     --dataloader_num_workers 12 \
#     --bf16 True \
#     --tf32 True \
#     --report_to tensorboard \
#     --output_dir outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_v2 \
#     --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus \
#     --pretrain_mm_mlp_adapter /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/mm_projector.bin \
#     --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_v2 \
#     --finetune_modules beacon_embed_tokens connnetor \
#     --enable_beacon True \
#     --skip_first True \
#     --beacon_window 720 \
#     --beacon_stride 720 \
#     --beacon_attn full-coverage \
#     --beacon_attend_prev True \
#     --beacon_sink_size 0 \
#     --beacon_ratio 72 60 48 \
#     --beacon_ratio_mix step-random \
#     --beacon_pos interleave \
#     --beacon_param q k v \
#     --return_all_logits True \
#     --compress_turn 1 \
#     --low_vision_encoder True \
#     --add_type fusion \
#     --add_vision_pretrained facebook/dinov2-large \
#     --group_by_stride relaxed \
#     --stream_loss_weight 1 \
#     --is_smoothing True \
#     --config_path configs/datasets/estpit.json \
#     --add_random_high_res_ratio 0.00 \

    
    # --pretrain_mm_mlp_adapter outputs/ego4d_narration_train/beacon_livel_h_stage1_v2/mm_projector.bin \

    # --only_modules_to_ft beacon_q_proj.lora_A beacon_q_proj.lora_B beacon_k_proj.lora_A beacon_k_proj.lora_B beacon_q_proj.lora_A beacon_q_proj.lora_B beacon_embed_tokens \
    
    # beacon_q_proj beacon_k_proj beacon_v_proj