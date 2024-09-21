META_FILE_PATH="datasets/niji_images/image_batch_$1.jsonl"
VIDEO_FOLDER="datasets/niji_images/niji_images"
VIDEO_CAPTION_SAVED_PATH="datasets/niji_images/image_caption_$1.jsonl"



export START_TIME=$(date +%s)
# Use VILA1.5-AWQ to perform recaptioning.
python3.10 qwen_image_recaptioning.py \
    --video_metadata_path ${META_FILE_PATH} \
    --video_folder ${VIDEO_FOLDER} \
    --saved_path $VIDEO_CAPTION_SAVED_PATH \


# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME))
echo "Duration: $DURATION seconds"
