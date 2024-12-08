export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python GLUE/NLU_GLUE_top-gm20.py \
    --model_name_or_path APCA_GGM_Lib/PretrainedModels/PretrainedModels2/FacebookAI/roberta-base \
    --dataset cola \
    --task cola \
    --max_length 512 \
    --head_lr 0.05 \
    --fft_lr 0.0005 \
    --weight_decay 0.005 \
    --num_epoch 100 \
    --bs 32  \
    --scale 49.0 \
    --seed 0 \
    --init_warmup 1 \
    --final_warmup 2