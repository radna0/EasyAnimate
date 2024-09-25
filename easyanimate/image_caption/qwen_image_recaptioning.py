import argparse
import torch
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import json

from utils.logger import logger


# set env TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD
os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "99999999999999999999"

# Initialize helpers for model management
helpers = {}


def perform_inference(message_obj):
    try:
        all_messages, idx, _ = message_obj
        outputs = []
        all_messages = [all_messages]

        if idx in helpers:
            model, processor, active = helpers[idx]
            if active:
                model, processor, _ = helpers[(idx + 1) % 2]
                helpers[(idx + 1) % 2] = (model, processor, True)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map="auto",
            )
            model.eval()

            max_pixels = 1280 * 28 * 28
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", max_pixels=max_pixels
            )

            helpers[idx] = (model, processor, True)

        for messages in all_messages:
            texts = [
                processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Inference
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.append(output_text[0])

        helpers[idx] = (model, processor, False)
        return outputs
    except Exception as e:
        logger.warning(f"Failed to process {message_obj}: {e}")
        return None


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
        default="Describe this image. In detail.",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        required=True,
        help="The save path to the output results (csv/jsonl).",
    )
    parser.add_argument(
        "--saved_freq",
        type=int,
        default=100,
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
    video_path_list = video_metadata_df["file_path"].tolist()
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

    video_path_list = [
        os.path.join(args.video_folder, video_path) for video_path in video_path_list
    ]
    video_path_list = natsorted(video_path_list)

    messages_obj = []
    logger.info(
        f"______Total: {len(video_path_list)} | Processing {video_path_list[0]}..."
    )
    for video_path in video_path_list:
        try:
            input_prompt = args.input_prompt
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://./{video_path}",
                        },
                        {"type": "text", "text": input_prompt},
                    ],
                }
            ]
            messages_obj.append(([message], len(messages_obj) % 2, video_path))
        except Exception as e:
            logger.warning(f"Failed to process {video_path}: {e}")

    result_dict = {"video_path": [], "caption": []}

    with ProcessPoolExecutor(max_workers=2) as executor, tqdm(
        total=len(messages_obj)
    ) as pbar:
        futures = {executor.submit(perform_inference, msg): msg for msg in messages_obj}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()  # Get result from the completed future
                message_obj = futures[future]
                messages, idx, video_path = message_obj
                logger.info(f"Finished processing {message_obj}. | result: {result}")
                video_name = os.path.basename(video_path)
                output_text = result[0]  # Assuming a single output per task
                result_dict["video_path"].append(video_name)
                result_dict["caption"].append(output_text)

                # Save intermediate results periodically
                if i != 0 and i % args.saved_freq == 0:
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
            except Exception as e:
                logger.warning(f"Failed to process {message_obj}: {e}")
            pbar.update(1)

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
