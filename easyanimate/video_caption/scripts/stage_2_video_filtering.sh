META_FILE_PATH="datasets/qbit-downloads_batch/meta_file_info_$1.jsonl"
VIDEO_FOLDER="datasets/qbit-downloads_batch/data_$1/data/"
VIDEO_QUALITY_SAVED_PATH="datasets/qbit-downloads_batch/data_$1/meta_quality_info_siglip.jsonl"
MIN_ASETHETIC_SCORE_SIGLIP=4.0
TEXT_SAVED_PATH="datasets/qbit-downloads_batch/data_$1/meta_text_info.jsonl"
MIN_TEXT_SCORE=0.02
MOTION_SAVED_PATH="datasets/qbit-downloads_batch/data_$1/meta_motion_info.jsonl"

# measure the duration to process
export START_TIME=$(date +%s)
python -m utils.get_meta_file \
    --video_folder $VIDEO_FOLDER \
    --saved_path $META_FILE_PATH


export START_TIME1=$(date +%s)
# Get the asethetic score (SigLIP) of all videos
accelerate launch compute_video_quality.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --metrics "AestheticScoreSigLIP" \
    --frame_sample_method uniform \
    --num_sampled_frames 4 \
    --saved_freq 10 \
    --saved_path $VIDEO_QUALITY_SAVED_PATH \
    --batch_size 64

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME1))
echo "Duration: $DURATION seconds"


export START_TIME2=$(date +%s)
# Get the text score of all videos filtered by the video quality score.
accelerate launch compute_text_score.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER  \
    --saved_freq 10 \
    --saved_path $TEXT_SAVED_PATH \
    --asethetic_score_siglip_metadata_path $VIDEO_QUALITY_SAVED_PATH \
    --min_asethetic_score_siglip $MIN_ASETHETIC_SCORE_SIGLIP

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME2))
echo "Duration: $DURATION seconds"

export START_TIME3=$(date +%s)
# Get the motion score of all videos filtered by the video quality score and text score.
python compute_motion_score.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --saved_freq 10 \
    --saved_path $MOTION_SAVED_PATH \
    --n_jobs 96 \
    --text_score_metadata_path $TEXT_SAVED_PATH \
    --min_text_score $MIN_TEXT_SCORE


# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME3))
echo "Duration: $DURATION seconds"

# measure the duration to process
export END_TIME=$(date +%s)
export DURATION=$((END_TIME-START_TIME))
echo "Duration: $DURATION seconds"