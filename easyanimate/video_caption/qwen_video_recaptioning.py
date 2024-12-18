import argparse
import torch
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import json

from utils.filter import filter
from utils.logger import logger
from vision_process import enable_xla_spmd, get_prompt_xla, get_qwen_prep


# set env TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD
os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "99999999999999999999"




def parse_args():
    parser = argparse.ArgumentParser(description="Recaption videos with Qwen2VL.")
    parser.add_argument(
        "--video_metadata_path",
        type=str,
        default=None,
        help="The path to the video dataset metadata (csv/jsonl).",
    )
    parser.add_argument(
        "--video_folder", type=str, default="", help="The video folder."
    )
    parser.add_argument(
        "--input_prompt",
        type=str,
        default="Describe this video. In detail.",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        required=True,
        help="The save path to the output results (csv/jsonl).",
    )
    parser.add_argument(
        "--basic_metadata_path",
        type=str,
        default=None,
        help="The path to the basic metadata (csv/jsonl).",
    )
    parser.add_argument(
        "--min_resolution", type=float, default=0, help="The resolution threshold."
    )
    parser.add_argument(
        "--min_duration", type=float, default=-1, help="The minimum duration."
    )
    parser.add_argument(
        "--max_duration", type=float, default=-1, help="The maximum duration."
    )
    parser.add_argument(
        "--asethetic_score_metadata_path",
        type=str,
        default=None,
        help="The path to the video quality metadata (csv/jsonl).",
    )
    parser.add_argument(
        "--min_asethetic_score",
        type=float,
        default=4.0,
        help="The asethetic score threshold.",
    )
    parser.add_argument(
        "--asethetic_score_siglip_metadata_path",
        type=str,
        default=None,
        help="The path to the video quality metadata (csv/jsonl).",
    )
    parser.add_argument(
        "--min_asethetic_score_siglip",
        type=float,
        default=4.0,
        help="The asethetic score (SigLIP) threshold.",
    )
    parser.add_argument(
        "--text_score_metadata_path",
        type=str,
        default=None,
        help="The path to the video text score metadata (csv/jsonl).",
    )
    parser.add_argument(
        "--min_text_score", type=float, default=0.02, help="The text threshold."
    )
    parser.add_argument(
        "--motion_score_metadata_path",
        type=str,
        default=None,
        help="The path to the video motion score metadata (csv/jsonl).",
    )
    parser.add_argument(
        "--min_motion_score", type=float, default=2, help="The motion threshold."
    )
    parser.add_argument(
        "--saved_freq",
        type=int,
        default=60,
        help="Frequency to save intermediate results.",
    )
    return parser.parse_args()


def main(args):
    if args.video_metadata_path.endswith(".csv"):
        video_metadata_df = pd.read_csv(args.video_metadata_path)
    elif args.video_metadata_path.endswith(".jsonl"):
        video_metadata_df = pd.read_json(args.video_metadata_path, lines=True)
    else:
        raise ValueError("The video_metadata_path must end with .csv or .jsonl.")
    print(video_metadata_df)
    # video_metadata_df only has video_path column, take data without calling video_path column
    video_path_list = video_metadata_df["video_path"].tolist()
    video_path_list = [os.path.basename(video_path) for video_path in video_path_list]

    if not (args.saved_path.endswith(".csv") or args.saved_path.endswith(".jsonl")):
        raise ValueError("The saved_path must end with .csv or .jsonl.")

    if os.path.exists(args.saved_path):
        if args.saved_path.endswith(".csv"):
            saved_metadata_df = pd.read_csv(args.saved_path)
        elif args.saved_path.endswith(".jsonl"):
            saved_metadata_df = pd.read_json(args.saved_path, lines=True)
        saved_video_path_list = saved_metadata_df["video_path"].tolist()
        video_path_list = list(
            set(video_path_list).difference(set(saved_video_path_list))
        )
        logger.info(
            f"Resume from {args.saved_path}: {len(saved_video_path_list)} processed and {len(video_path_list)} to be processed."
        )

    video_path_list = filter(
        video_path_list,
        basic_metadata_path=args.basic_metadata_path,
        min_resolution=args.min_resolution,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        asethetic_score_metadata_path=args.asethetic_score_metadata_path,
        min_asethetic_score=args.min_asethetic_score,
        asethetic_score_siglip_metadata_path=args.asethetic_score_siglip_metadata_path,
        min_asethetic_score_siglip=args.min_asethetic_score_siglip,
        text_score_metadata_path=args.text_score_metadata_path,
        min_text_score=args.min_text_score,
        motion_score_metadata_path=args.motion_score_metadata_path,
        min_motion_score=args.min_motion_score,
    )

    video_path_list = [
        os.path.join(args.video_folder, video_path) for video_path in video_path_list
    ]
    video_path_list = natsorted(video_path_list)

    messages_obj = []
    batch_size = 6
    logger.info(f"Enable XLA SPMD: {enable_xla_spmd()} with batch size {batch_size}.")
    logger.info(
        f"______Total: {len(video_path_list)} "
    )
    for video_path in video_path_list:
        input_prompt = args.input_prompt
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://./{video_path}",
                        "fps": 1.0,
                    },
                    {"type": "text", "text": input_prompt},
                ],
            }
        ]
        messages_obj.append((message, video_path))
    
    batches_messages_obj = []
    for i in range(0, len(messages_obj), batch_size):
        # [message_obj for message_obj in messages_obj[i:i + batch_size]]
        message_batch = [message_obj for message_obj, _ in messages_obj[i:i + batch_size]]
        video_batch = [video_path for _, video_path in messages_obj[i:i + batch_size]]

        batches_messages_obj.append((message_batch, video_batch))

    logger.info(f"Total batches: {len(batches_messages_obj)}")

    result_dict = {"video_path": [], "caption": []}

    model, processor = get_qwen_prep()

    # loop through tqdm and batch
    for i, (message_batch, video_batch) in zip(tqdm(range(len(batches_messages_obj))), batches_messages_obj):
        output_texts = get_prompt_xla(model, processor, message_batch, batch_size=batch_size)
        for video_path, output_text in zip(video_batch, output_texts):
            video_name = os.path.basename(video_path)
            result_dict["video_path"].append(video_name)
            result_dict["caption"].append(output_text)

            if len(result_dict["video_path"]) % args.saved_freq == 0:
                logger.info(f"Saving intermediate results at iteration {i}")
                result_df = pd.DataFrame(result_dict)
                if args.saved_path.endswith(".csv"):
                    header = not os.path.exists(args.saved_path)
                    result_df.to_csv(
                        args.saved_path, header=header, index=False, mode="a"
                    )
                elif args.saved_path.endswith(".jsonl"):
                    result_df.to_json(
                        args.saved_path,
                        orient="records",
                        lines=True,
                        mode="a",
                        force_ascii=False,
                    )
                for k in result_dict.keys():
                    result_dict[k] = []
 
    # Final save
    logger.info(f"Saving final results to {args.saved_path}.")
    result_df = pd.DataFrame(result_dict)
    if args.saved_path.endswith(".csv"):
        header = not os.path.exists(args.saved_path)
        result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
    elif args.saved_path.endswith(".jsonl"):
        result_df.to_json(
            args.saved_path, orient="records", lines=True, mode="a", force_ascii=False
        )
    logger.info(f"Finished processing. Results saved to {args.saved_path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
