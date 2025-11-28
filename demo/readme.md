

# demo
python -m demo.cli --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus

python -m demo.inference_llaveNext --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus

# evaluate openeqa

# evaluate egoschema

python -m demo.egoschema_videollmonline --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus

python -m demo.egoschema_llavaNext --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus