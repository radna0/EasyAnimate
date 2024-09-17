VIDEO_FOLDER="datasets/Qbit_Downloads/batch_$1/"
META_FILE_PATH="datasets/Qbit_Downloads/meta_file_info_$1.jsonl"
SCENE_FOLDER="datasets/Qbit_Downloads/meta_scene_info_$1/"
SCENE_SAVED_PATH="datasets/Qbit_Downloads/meta_scene_info_$1.jsonl"
OUTPUT_FOLDER="datasets/Qbit_Downloads/data_$1/data/"
RESOLUTION_THRESHOLD=$((960*960))

# Set the duration range of video clips.
export MIN_SECONDS=3
export MAX_SECONDS=10
# measure the duration to process
export START_TIME=$(date +%s)
# Save all video names in a video folder as a meta file.
python3.10 -m utils.get_meta_file \
    --video_folder $VIDEO_FOLDER \
    --saved_path $META_FILE_PATH

# Perform scene detection on the video dataset.
# Adjust the n_jobs parameter based on the actual number of CPU cores in the machine.
export START_TIME1=$(date +%s)
python3.10 cutscene_detect.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --saved_folder $SCENE_FOLDER \
    --n_jobs 96

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME1))
echo "Duration: $DURATION seconds"

# Gather all scene jsonl files to a single scene jsonl file.
# Adjust the n_jobs parameter based on the actual I/O speed in the machine.
export START_TIME2=$(date +%s)
python3.10 -m utils.gather_jsonl \
    --meta_folder $SCENE_FOLDER \
    --meta_file_path $SCENE_SAVED_PATH \
    --n_jobs 96

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME2))
echo "Duration: $DURATION seconds"

# Perform video splitting filtered by the RESOLUTION_THRESHOLD.
# It consumes more CPU computing resources compared to the above operations.
export START_TIME3=$(date +%s)
python3.10 video_splitting.py \
    --video_metadata_path $SCENE_SAVED_PATH \
    --video_folder $VIDEO_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    --n_jobs 96 \
    --resolution_threshold $RESOLUTION_THRESHOLD

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME3))
echo "Duration: $DURATION seconds"

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME))
echo "Duration: $DURATION seconds"
