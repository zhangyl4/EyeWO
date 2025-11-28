export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 1 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 1"
    nnodes=1
fi

${launcher} --master_port 2367 merge_lora.py \
    --live_version beaconlivel_h \
    --eval_datasets ego4d_ESTPCQAHighResGen \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --prediction_loss_only False \
    --prediction_loss_only False \
    --dataloader_num_workers 12 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_5_livebase_cqa \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--llamaEyeWO-stage2-livebase \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_5_livebase_cqa \
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
    --return_all_logits True \
    --compress_turn 1 \
    --root datasets/ego4d \
    --anno_path estp_bench_sq.json \
    --low_vision_encoder True \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --group_by_stride relaxed \
    --stream_loss_weight 1 \
    --is_smoothing True \
    --config_path configs/datasets/estpit.json \
    --add_random_high_res_ratio 0.00 \
    