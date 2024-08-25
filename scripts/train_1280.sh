export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV4-XL-2-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/train_1280.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_slicevae_multi_text_encoder_v4.yaml" \
  --image_sample_size=1280 \
  --video_sample_size=1280 \
  --token_sample_size=1280 \
  --video_sample_stride=1 \
  --video_sample_n_frames=144 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --motion_sub_loss \
  --not_sigma_loss \
  --random_frame_crop \
  --enable_bucket \
  --use_deepspeed \
  --train_mode="inpaint" \
  --trainable_modules "."