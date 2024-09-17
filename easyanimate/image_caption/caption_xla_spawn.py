from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import torch
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs

from torch.distributed._tensor import DeviceMesh, distribute_module
from torch_xla.distributed.spmd import auto_policy

from torch_xla import runtime as xr
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    _prepare_spmd_partition_spec,
    SpmdFullyShardedDataParallel as FSDPv2,
)

import time

start = time.time()

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
)

print(model.device)


# default processer

max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", max_pixels=max_pixels
)


message = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://p11-sign.douyinpic.com/tos-cn-i-0813c001/oIt9A0rEnoIa2AumAWAQ9eTFJfG6LhACADwCgP~tplv-dy-lqen-new:1438:1910:q82.jpeg?x-expires=1728374400&x-signature=KmnL29qpx1r7S3wWCE0m3rnpF8w%3D&from=327834062&s=PackSourceEnum_AWEME_DETAIL&se=false&sc=image&biz_tag=aweme_images&l=202409081622473A9A9B405BA9E5EAEB7B",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

all_messages = [[message for _ in range(1)]]
for messages in all_messages:

    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
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
    print(inputs.items())
    for key, value in inputs.items():
        print(key, value.shape)

    # Inference: Generation of the output
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
    for i, text in enumerate(output_text):
        print(f"Output {i}: {text}")


print(f"Time taken: {time.time() - start}")
