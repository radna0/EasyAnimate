export VIDEO_FOLDER="datasets/qbit-downloads_batch/qbit_downloads_dataset_batch_$1/"
export OUTPUT_FOLDER="datasets/qbit-downloads_batch/train_$1/"
# Cut raw videos
python scenedetect_vcut.py \
    $VIDEO_FOLDER \
    --threshold 10 20 30 \
    --frame_skip 0 1 2 \
    --min_seconds 3 \
    --max_seconds 10 \
    --save_dir $OUTPUT_FOLDER