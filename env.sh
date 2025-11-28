pip install transformers==4.48.3 accelerate deepspeed==0.15.4 peft editdistance Levenshtein tensorboard gradio moviepy submitit -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install spacy
python -m spacy download en_core_web_sm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "protobuf<4.0.0"
pip install nvidia-cublas-cu12==12.4.5.8 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install decord -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install nvitop -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install nvidia-cublas-cu12==12.4.5.8 -i https://pypi.tuna.tsinghua.edu.cn/simple 


# source /2022233235/miniconda3/etc/profile.d/conda.sh
# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/2022233235/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/2022233235/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/2022233235/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/2022233235/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup