export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 1 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 1"
    nnodes=1
fi

${launcher} --master_port 1408 merge_lora.py \
    --live_version livel_h \
    --eval_datasets coin_step_high_test coin_next_high_test coin_task_high_test coin_procedure_high_test coin_taskprocedure_high_test \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --prediction_loss_only False \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/coin_benchmarks/livel_h_2fps_llm3_1_so400m \
    --vision_pretrained google/siglip-so400m-patch14-384 \
    --llm_pretrained meta-llama/Llama-3.1-8B-Instruct \
    --resume_from_checkpoint outputs/coin_benchmarks/livel_h_2fps_llm3_1_so400m/


    