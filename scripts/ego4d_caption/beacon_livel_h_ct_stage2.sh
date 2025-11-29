export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 4 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 4"
    nnodes=1
fi

# ${launcher} --master_port 1108 train_wVsionEncoder.py --deepspeed configs/deepspeed/zero1.json \
#     --live_version beaconlivel_h \
#     --train_datasets ego4d_refined_narration_stream_high_origin_train \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps $((8/$nnodes)) \
#     --gradient_checkpointing True \
#     --evaluation_strategy no \
#     --prediction_loss_only False \
#     --save_strategy steps \
#     --save_steps 100 \
#     --save_total_limit 2 \
#     --learning_rate 0.00015 \
#     --optim adamw_torch \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.05 \
#     --logging_steps 5 \
#     --dataloader_num_workers 12 \
#     --bf16 True \
#     --tf32 True \
#     --report_to tensorboard \
#     --output_dir outputs/ego4d_narration_train/beaconlivel_h_ct_stage2_v2 \
#     --pretrain_mm_mlp_adapter outputs/ego4d_narration_train/beacon_livel_h_stage1_v2/mm_projector.bin \
#     --max_num_frames 525 \
#     --finetune_modules beacon_embed_tokens \
#     --enable_beacon True \
#     --skip_first True \
#     --compress_turn 12 \
#     --beacon_window 1024 \
#     --beacon_stride 1024 \
#     --beacon_attn full-coverage \
#     --beacon_attend_prev True \
#     --beacon_sink_size 0 \
#     --beacon_ratio 85 64 32 \
#     --beacon_ratio_mix step-random \
#     --beacon_pos interleave \
#     --beacon_param q k v \
    

${launcher} --master_port 1340 evaluate_wVsionEncoder.py \
    --live_version beaconlivel_h \
    --eval_datasets ego4d_refined_narration_stream_high_origin_val \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --prediction_loss_only False \
    --dataloader_num_workers 12 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/ego4d_narration_train/beaconlivel_h_ct_stage2_v2 \
    --pretrain_mm_mlp_adapter outputs/ego4d_narration_train/beacon_livel_h_stage1_v2/mm_projector.bin \
    --resume_from_checkpoint outputs/ego4d_narration_train/beaconlivel_h_ct_stage2_v2/ \
    --max_num_frames 525 \
    --finetune_modules beacon_embed_tokens \
    --enable_beacon True \
    --skip_first True \
    --compress_turn 12 \
    --beacon_window 1024 \
    --beacon_stride 1024 \
    --beacon_attn full-coverage \
    --beacon_attend_prev True \
    --beacon_sink_size 0 \
    --beacon_ratio 32 85 64 \
    --beacon_ratio_mix step-random \
    --beacon_pos interleave \
    --beacon_param q k v \
    --return_all_logits True \
    --frame_token_interval_threshold 0.9 \
    

    # --only_modules_to_ft beacon_q_proj.lora_A beacon_q_proj.lora_B beacon_k_proj.lora_A beacon_k_proj.lora_B beacon_q_proj.lora_A beacon_q_proj.lora_B beacon_embed_tokens \
    
    # beacon_q_proj beacon_k_proj beacon_v_proj