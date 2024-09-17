from concurrent.futures import ProcessPoolExecutor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import torch

helpers = {}


def perform_inference(message_obj):
    all_messages, idx = message_obj
    outputs = []
    all_messages = [all_messages]

    if idx in helpers:
        model, processor, active = helpers[idx]
        print(f"model device: {idx} {active} {model.device}")
        if active:
            model, processor, _ = helpers[idx + 1 % 2]
            helpers[idx + 1 % 2] = (model, processor, True)
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

    print(f"model device: {idx} {model.device}")

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

        outputs.append(output_text)

    helpers[idx] = (model, processor, False)
    return outputs


if __name__ == "__main__":
    start = time.time()

    # Define the message(s) to process
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "file://../video_caption/datasets/qbit-downloads_batch/data_11/data/[sam] Mahouka Koukou no Rettousei - 04 [BD 1080p FLAC] [632E4D5A]_12_1.mp4",
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video. In detail."},
            ],
        }
    ]

    messages_obj = [
        ([message], idx % 2)
        for idx in range(2)
        #   ([message for _ in range(4)], idx % 2) for idx in range(2)
    ]  # Create a list of 2 identical messages

    # Use ProcessPoolExecutor to process messages concurrently with 2 workers
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(perform_inference, messages_obj))

    # Display the results
    for i, output_text in enumerate(results):
        for j, text in enumerate(output_text):
            print(f"\n\nOutput {i}-{j}: {text}\n\n")

    print(f"Total time taken: {time.time() - start}")
