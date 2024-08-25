export VIDEO_FOLDER="datasets/qbit-downloads_batch/train_$1/"
export FRAME_QUALITY_SAVE_PATH="datasets/qbit-downloads_batch/aesthetic_score_$1.jsonl"
export TEXT_SCORE_SAVE_PATH="datasets/qbit-downloads_batch/text_score_$1.jsonl"
export MOTION_SCORE_SAVE_PATH="datasets/qbit-downloads_batch/motion_score_$1.jsonl"
export FILTER_BY_MOTION_SCORE_SAVE_PATH="datasets/qbit-downloads_batch/train_$1.jsonl"

# Get asethetic score of all videos
CUDA_VISIBLE_DEVICES="0" accelerate launch compute_video_frame_quality.py \
    --video_folder=$VIDEO_FOLDER \
    --video_path_column="video_path" \
    --metrics="AestheticScore" \
    --saved_freq=10 \
    --saved_path=$FRAME_QUALITY_SAVE_PATH \
    --batch_size=8

# Get text score of all videos
CUDA_VISIBLE_DEVICES="0" accelerate launch compute_text_score.py \
    --video_folder=$VIDEO_FOLDER  \
    --video_path_column="video_path" \
    --saved_freq=10 \
    --saved_path=$TEXT_SCORE_SAVE_PATH \
    --asethetic_score_metadata_path $FRAME_QUALITY_SAVE_PATH

# Get motion score after filter videos by asethetic score and text score
python compute_motion_score.py \
    --video_folder=$VIDEO_FOLDER \
    --video_path_column="video_path" \
    --saved_freq=10 \
    --saved_path=$MOTION_SCORE_SAVE_PATH \
    --n_jobs=8 \
    --asethetic_score_metadata_path $FRAME_QUALITY_SAVE_PATH \
    --text_score_metadata_path $TEXT_SCORE_SAVE_PATH

# Filter videos by motion score
python filter_videos_by_motion_score.py \
    --motion_score_metadata_path $MOTION_SCORE_SAVE_PATH \
    --low_motion_score_threshold=3 \
    --high_motion_score_threshold=8 \
    --saved_path $FILTER_BY_MOTION_SCORE_SAVE_PATH
