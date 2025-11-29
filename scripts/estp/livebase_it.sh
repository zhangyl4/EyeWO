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
    --live_version live1+ \
    --train_datasets ego4d_ESTPSQA ego4d_ESTPCQA \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --eval_strategy no \
    --prediction_loss_only False \
    --save_strategy no \
    --learning_rate 0.0001 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/ego4d_ESTPSQA/livebase_it_smooth_dino \
    --config_path configs/datasets/estpit.json \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus \
    --pretrain_mm_mlp_adapter /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/mm_projector.bin \
    --low_vision_encoder True \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --is_smoothing True
