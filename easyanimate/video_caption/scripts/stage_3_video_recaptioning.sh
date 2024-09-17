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

