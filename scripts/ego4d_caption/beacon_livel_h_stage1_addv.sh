export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 8"
    nnodes=1
fi

${launcher} --master_port 13098 train_wVsionEncoder.py --deepspeed configs/deepspeed/zero1.json \
    --live_version livel_h \
    --train_datasets ego4d_refined_action_caption_stream_high_origin_train ego4d_refined_action_caption_stream_high_origin_val ego4d_refined_scene_caption_stream_high_origin_train ego4d_refined_scene_caption_stream_high_origin_val \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $((8/$nnodes)) \
    --gradient_checkpointing True \
    --evaluation_strategy no \
    --prediction_loss_only False \
    --save_strategy no \
    --learning_rate 0.0005 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --dataloader_num_workers 8 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/ego4d_caption_train/livel_h_stage1_2_4_dual \
    --stream_loss_weight 0.0 \
    --learn_reponse False \
    --only_modules_to_ft connector \
    --low_vision_encoder True \
    --root datasets \
    --max_num_frames 500 \
    --add_type dual \
    --add_vision_pretrained facebook/dinov2-large \


