######################################################## passive inference ########################################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name MiniCPMV \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/MiniCPMV_passive.json \
    --device cuda:0 \

export CUDA_VISIBLE_DEVICES=3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name MiniCPMV \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_baseline/MiniCPMV_passive.json \
    --device cuda:0 \

conda activate llava
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name LLaVAOneVision \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/LLaVAOneVision_passive.json \
    --device cuda:0 \

conda activate llava
export CUDA_VISIBLE_DEVICES=1
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name LLaVAOneVision \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_baseline/LLaVAOneVision_passive.json \
    --device cuda:0 \

conda activate llava
export CUDA_VISIBLE_DEVICES=1
python /2022233235/videollm-online/eval_estp_batch.py  \ 
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name LLaVANextVideo7B \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/LLaVANextVideo7B_passive.json \
    --device cuda:0 \

conda activate llava
export CUDA_VISIBLE_DEVICES=2
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name LLaVANextVideo7B \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_baseline/LLaVANextVideo7B_passive.json \
    --device cuda:0 \

conda activate videollm
export CUDA_VISIBLE_DEVICES=2
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name InternVLV28 \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/InternVLV28_passive.json \
    --device cuda:0 \

conda activate videollm
export CUDA_VISIBLE_DEVICES=3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name InternVLV28 \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_baseline/InternVLV28_passive.json \https://www.allaboutai.com/https://www.allaboutai.com/
    --device cuda:0 \

conda activate videollm
export CUDA_VISIBLE_DEVICES=3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name Qwen2VL \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/Qwen2VL_passive.json \
    --device cuda:0 \

conda activate videollm
export CUDA_VISIBLE_DEVICES=2
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name Qwen2VL \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_baseline/Qwen2VL_passive.json \
    --device cuda:0 \

conda activate llava
export CUDA_VISIBLE_DEVICES=4
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name VILA \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/VILA_passive.json \
    --device cuda:0 \


######################################################## Grounding LLM ########################################################
export CUDA_VISIBLE_DEVICES=1
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name TimeChat \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/TimeChat_passive.json \
    --device cuda:0 \





######################################################## turn ask on 5 cases ########################################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq_5_cases.json \
    --model_name MiniCPMV \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline_5cases/MiniCPMV_fbf_5cases_0.175.json \
    --device cuda:0 \
    --fbf_fps 0.175 \
    --master_port 2984

export CUDA_VISIBLE_DEVICES=4,5,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq_5_cases.json \
    --model_name Qwen2VL \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline_5cases/Qwen2VL_fbf_5cases.json \
    --device cuda:0 \
    --master_port 2984

export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq_5_cases.json \
    --model_name LLaVAOneVision \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline_5cases/LLaVAOneVision_fbf_5cases.json \
    --device cuda:0 \
    --master_port 2984


export CUDA_VISIBLE_DEVICES=4,5,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq_5_cases.json \
    --model_name LLaVANextVideo7B \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline_5cases/LLaVANextVideo7B_fbf_5cases.json \
    --device cuda:0 \
    --master_port 2984

######################################################## turn ask on single QA ########################################################
export CUDA_VISIBLE_DEVICES=0,1,2,4
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name MiniCPMV \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/MiniCPMV_fbf_singleQA_0.175.json \
    --device cuda:0 \
    --fbf_fps 0.175 \
    --master_port 1542

export CUDA_VISIBLE_DEVICES=4,5,6
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name MiniCPMV \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_baseline/MiniCPMV_fbf_0.175.json \
    --device cuda:0 \
    --fbf_fps 0.175 \
    --master_port 1542

export CUDA_VISIBLE_DEVICES=3,4,5,6
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name Qwen2VL \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/Qwen2VL_fbf_singleQA_0.175.json \
    --device cuda:0 \
    --fbf_fps 0.175 \
    --master_port 1582

export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name LLaVAOneVision \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/LLaVAOneVision_fbf_singleQA_0.175.json \
    --fbf_fps 0.175 \
    --device cuda:0 \
    --master_port 2984


export CUDA_VISIBLE_DEVICES=3,4
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name LLaVANextVideo7B \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/LLaVANextVideo7B_fbf_singleQA.json \
    --device cuda:0 \
    --master_port 2984

export CUDA_VISIBLE_DEVICES=3,4
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name LLaVANextVideo7B \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/LLaVANextVideo7B_fbf_singleQA.json \
    --device cuda:0 \
    --master_port 2984


######################################################## streaming inference ########################################################

export CUDA_VISIBLE_DEVICES=4
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name EgoVLP \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/EgoVLP_streaming.json \
    --device cuda:0 \
    --master_port 2984

export CUDA_VISIBLE_DEVICES=5
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name CLIP \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/CLIP_streaming.json \
    --device cuda:0 \
    --master_port 2984

export CUDA_VISIBLE_DEVICES=3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name Lavila \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode passive_inference \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/Lavila_streaming.json \
    --device cuda:0 \
    --master_port 2984

######################################################## online inference ########################################################

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name VideollmOnline \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq_VideollmOnline0.8.json \
    --device cuda:0 \
    --master_port 2984

export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,4
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name VideollmOnline \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_baseline/VideollmOnline0.9.json \
    --device cuda:0 \
    --master_port 1984

conda activate llava
export CUDA_VISIBLE_DEVICES=3,4,5,6
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name MMDuet \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/MMDuet.json \
    --device cuda:0 \
    --master_port 2984

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PYTHONPATH=$(pwd) python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name Qwen2VL_streaming \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_baseline/Qwen2VL_streaming.json \
    --master_port 2984

######################################################## ours inference ########################################################

# qwen2vl eyewo
export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=768 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=4816896 # 19267584 = 24576*28*28. 4816896 the maximum overall video tokens sent to llm is 24k (leave 8k for language)
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=$(pwd) python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --resume_from_checkpoint /2022233235/videollm-online/livecc_eyewo/outputs/livecc_eyewo_sft_24k768x100_lora_lr_sqa_balance_v55e-5/checkpoint-851/\
    --model_name Qwen2VL_EyeWO \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours/Qwen2VL_EyeWO.json \
    --master_port 2984




# Multi QA
# stage 2
export CUDA_VISIBLE_DEVICES=4,5,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
    --pretrain_mm_mlp_adapter /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/mm_projector.bin \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_all \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_ours/LivebaseStage2_v4.json \
    --device cuda:0 \
    --master_port 2280 \


export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--llamaEyeWO-stage2-livebase \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_ours/LivebaseStage2_v3.json \
    --device cuda:0 \
    --master_port 22890 \


export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--llamaEyeWO-stage2.5-livebase \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_ours/LivebaseStage2.5.json \
    --device cuda:0 \
    --master_port 22890 \

export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--llamaEyeWO-stage2.5-livebase \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage3.5_livebase_high_fixbug/ \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_ours/LivebaseStage3.5_v3.json \
    --device cuda:0 \
    --master_port 2280 \

# single QA
# stage 3.5
export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--llamaEyeWO-stage2.5-livebase \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage3.5_livebase_high_fixbug/ \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours/LivebaseStage3.5_v3.json \
    --device cuda:0 \
    --master_port 2280 \


# stage 2.5
export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--llamaEyeWO-stage2.5-livebase \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours/LivebaseStage2.5.json \
    --device cuda:0 \
    --master_port 22890 \


# stage 3
export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--llamaEyeWO-stage2-livebase \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage3_livebase_high_0.31_11/ \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours/LivebaseStage3_v3.json \
    --device cuda:0 \
    --master_port 22890 \

# stage 2
export CUDA_VISIBLE_DEVICES=4,5,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
    --pretrain_mm_mlp_adapter /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/mm_projector.bin \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_all \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours/LivebaseStage2_v4.json \
    --device cuda:0 \
    --master_port 2280 \


# subset 5 cases
# stage 3
export CUDA_VISIBLE_DEVICES=5,6
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq_5_cases.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--llamaEyeWO-stage2-livebase \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage3_livebase_high_0.31_11/ \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours_5cases/LivebaseStage3_high0.31_11.json \
    --device cuda:0 \
    --master_port 22890 \

# stage 2
export CUDA_VISIBLE_DEVICES=4,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq_5_cases.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
    --pretrain_mm_mlp_adapter /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/mm_projector.bin \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_v2 \
    --add_type fusion \
    --add_vision_pretrained facebook/dinov2-large \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours_5cases/LivebaseStage2.json \
    --device cuda:0 \
    --master_port 2280 \


######################################################## ours ablation ########################################################


# wo dino
export CUDA_VISIBLE_DEVICES=4
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_wodino \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours/LivebaseStage2_woDino_debug.json \
    --device cuda:0 \
    --master_port 2253 \

export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name EWO \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_wodino \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_ours/LivebaseStage2_woDino.json \
    --device cuda:0 \
    --master_port 2280 \


# stage 0,
export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name VideollmOnline \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/livebase_it \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_ours/LIVE_IT0.95.json \
    --device cuda:0 \
    --master_port 1984

export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name VideollmOnline \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/livebase_it \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours/LIVE_IT0.95.json \
    --device cuda:0 \
    --master_port 2984

export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_cq_v3.json \
    --model_name VideollmOnline \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/livebase_it_smooth_sqa \
    --benchmark_name ESTP_contextualQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpCqa_ours/LIVE_IT_smoothing_v2.json \
    --device cuda:0 \
    --master_port 1984 \
    --frame_token_interval_threshold 0.0

export PYTHONPATH=/2022233235/videollm-online:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
python /2022233235/videollm-online/eval_estp_batch.py  \
    --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
    --model_name VideollmOnline \
    --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/livebase_it_smooth_sqa \
    --benchmark_name ESTP_singleQ_benchmark \
    --eval_mode frame_by_frame \
    --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours/LIVE_IT_smoothing_v2.json \
    --device cuda:0 \
    --master_port 1989 \
    --frame_token_interval_threshold 0.0