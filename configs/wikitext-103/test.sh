checkpoints_path=$1
gpu=${2:-0}

CUDA_VISIBLE_DEVICES=$gpu python -u eval_lm.py \
                       --data data-bin/wikitext-103 \
                       --path $checkpoints_path \
                       --sample-break-mode none \
                       --max-tokens 2560 \
                       --context-window 2048 \
                       --softmax-batch 1024