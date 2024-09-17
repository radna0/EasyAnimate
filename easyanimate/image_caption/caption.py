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

xla.experimental.eager_mode(True)
xr.use_spmd()

num_devices = xr.global_runtime_device_count()
mesh_shape = (num_devices, 1)
device_ids = np.array(range(num_devices))
# To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
xs.set_global_mesh(mesh)


device = xla.device()
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
).to(device)

print(model.device)

model = FSDPv2(model)
model = torch.compile(model, backend="openxla")

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
                "image": "https://cdn.discordapp.com/attachments/1247496748281233428/1282227211067199549/tcaal.png?ex=66de96a5&is=66dd4525&hm=0eca1b4abf67d68cab061f4cb7c86aaac3e6e6e19e7510bb894819f973b74ac1&",
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
    inputs = inputs.to(device)
    print(inputs.items())

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
