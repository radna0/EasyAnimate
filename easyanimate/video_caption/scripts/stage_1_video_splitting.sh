VIDEO_FOLDER="datasets/arcane/batch_$1/"
META_FILE_PATH="datasets/arcane/meta_file_info_$1.jsonl"
SCENE_FOLDER="datasets/arcane/meta_scene_info_$1/"
SCENE_SAVED_PATH="datasets/arcane/meta_scene_info_$1.jsonl"
OUTPUT_FOLDER="datasets/arcane/data_$1/"
RESOLUTION_THRESHOLD=$((960*540))

# Set the duration range of video clips.
export MIN_SECONDS=3
export MAX_SECONDS=5



# measure the duration to process
export START_TIME=$(date +%s)

# Save all video names in a video folder as a meta file.
python3.10 -m utils.get_meta_file \
    --video_folder $VIDEO_FOLDER \
    --saved_path $META_FILE_PATH



# Perform scene detection on the video dataset.
# Adjust the n_jobs parameter based on the actual number of CPU cores in the machine.
export START_TIME_ONE=$(date +%s)
python3.10 cutscene_detect.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --saved_folder $SCENE_FOLDER \
    --n_jobs 96

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME_ONE))
echo "Duration cutscene_detect.py: $DURATION seconds"




# Gather all scene jsonl files to a single scene jsonl file.
# Adjust the n_jobs parameter based on the actual I/O speed in the machine.
export START_TIME_TWO=$(date +%s)
python3.10 -m utils.gather_jsonl \
    --meta_folder $SCENE_FOLDER \
    --meta_file_path $SCENE_SAVED_PATH \
    --n_jobs 96

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME_TWO))
echo "Duration utils.gather_jsonl: $DURATION seconds"




# Perform video splitting filtered by the RESOLUTION_THRESHOLD.
# It consumes more CPU computing resources compared to the above operations.
export START_TIME_THREE=$(date +%s)
python3.10 video_splitting.py \
    --video_metadata_path $SCENE_SAVED_PATH \
    --video_folder $VIDEO_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    --n_jobs 96 \
    --resolution_threshold $RESOLUTION_THRESHOLD

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME_THREE))
echo "Duration video_splitting.py: $DURATION seconds"




# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME))
echo "Total Duration: $DURATION seconds"
