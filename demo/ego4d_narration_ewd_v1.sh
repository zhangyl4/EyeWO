python -m demo.ego4d_narration_ewd_v0 \
    --live_version beaconlivel_h \
    --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_ct_stage2_skipsysonly \
    --finetune_modules beacon_embed_tokens connector \
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
    --low_vision_encoder True \
    --compress_turn 1 \
    --pretrain_mm_mlp_adapter outputs/ego4d_narration_train/beacon_livel_h_stage1_v2/mm_projector.bin
    # /root/videollm-online/outputs/ego4d_ESTPSQA/beaconlivel_h_ct_stage2_skipsysonly