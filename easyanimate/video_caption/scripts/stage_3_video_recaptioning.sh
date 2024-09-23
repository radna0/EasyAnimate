META_FILE_PATH="datasets/Qbit_Downloads/meta_file_info_$1.jsonl"
VIDEO_FOLDER="datasets/Qbit_Downloads/data_$1/data/"
MOTION_SAVED_PATH="datasets/Qbit_Downloads/data_$1/meta_motion_info.jsonl"
MIN_MOTION_SCORE=2
VIDEO_CAPTION_SAVED_PATH="datasets/Qbit_Downloads/meta_caption_info_vila_8b_$1.jsonl"
REWRITTEN_VIDEO_CAPTION_SAVED_PATH="datasets/Qbit_Downloads/meta_caption_info_vila_8b_rewritten_$1.jsonl"
VIDEOCLIPXL_SCORE_SAVED_PATH="datasets/Qbit_Downloads/meta_caption_info_vila_8b_rewritten_videoclipxl_$1.jsonl"
MIN_VIDEOCLIPXL_SCORE=0.20
TRAIN_SAVED_PATH="datasets/Qbit_Downloads/train_qbit-downloads_$1.json"
# Manually download Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ to VILA_MODEL_PATH.
# Manually download meta-llama/Meta-Llama-3-8B-Instruct to REWRITE_MODEL_PATH.

python3.10 -m utils.get_meta_file \
    --video_folder $VIDEO_FOLDER \
    --saved_path $META_FILE_PATH


export START_TIME=$(date +%s)
export START_TIME1=$(date +%s)
# Use VILA1.5-AWQ to perform recaptioning.
python3.10 qwen_video_recaptioning.py \
    --video_metadata_path ${META_FILE_PATH} \
    --video_folder ${VIDEO_FOLDER} \
    --saved_path $VIDEO_CAPTION_SAVED_PATH \
    --motion_score_metadata_path $MOTION_SAVED_PATH \
    --min_motion_score $MIN_MOTION_SCORE

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME1))
echo "Duration: $DURATION seconds"


export START_TIME3=$(date +%s)
# Compute caption-video alignment (optional).
accelerate launch compute_video_quality.py \
    --video_metadata_path $VIDEO_CAPTION_SAVED_PATH \
    --caption_column caption \
    --video_folder $VIDEO_FOLDER \
    --frame_sample_method uniform \
    --num_sampled_frames 8 \
    --metrics VideoCLIPXLScore \
    --batch_size 64 \
    --saved_path $VIDEOCLIPXL_SCORE_SAVED_PATH \
    --saved_freq 10
# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME3))
echo "Duration: $DURATION seconds"


export START_TIME4=$(date +%s)
# Get the final train file.
python3.10 filter_meta_train.py \
    --caption_metadata_path $VIDEO_CAPTION_SAVED_PATH \
    --video_folder=$VIDEO_FOLDER \
    --videoclipxl_score_metadata_path $VIDEOCLIPXL_SCORE_SAVED_PATH \
    --min_videoclipxl_score $MIN_VIDEOCLIPXL_SCORE \
    --saved_path=$TRAIN_SAVED_PATH
# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME4))
echo "Duration: $DURATION seconds"


# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME))
echo "Duration: $DURATION seconds"
