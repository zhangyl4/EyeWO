export CUDA_VISIBLE_DEVICES=0,1,3,5
export TOKENIZERS_PARALLELISM=false
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 4 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 4"
    nnodes=1
fi

${launcher} --master_port 12098 train_wVsionEncoder.py --deepspeed configs/deepspeed/zero2.json \
    --live_version livel_h \
    --train_datasets ego4d_refined_narration_stream_high_origin_train ego4d_refined_action_caption_stream_high_origin_train \
    --eval_datasets ego4d_refined_narration_stream_high_origin_val ego4d_refined_action_caption_stream_high_origin_val \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $((8/$nnodes)) \
    --gradient_checkpointing True \
    --evaluation_strategy no \
    --prediction_loss_only False \
    --save_strategy no \
    --learning_rate 0.0002 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --dataloader_num_workers 12 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/ego4d_narration_caption_train/beacon_livel_h_stage1_v2 \
    --max_num_frames 200 \
    --stream_loss_weight 0.0 \
    --learn_reponse False \
    --only_modules_to_ft connector \


