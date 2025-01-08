import torch
import transformers
import json
import sys
import os
from tqdm import tqdm
from PIL import Image

from transformers.generation.streamers import TextStreamer
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.llava_gpt2 import LlavaGpt2ForCausalLM
from llava.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llava.train.dataset import tokenizer_image_token

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')


if __name__ == "__main__":
    # Input files
    questions_file = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/questions_ja.jsonl"
    image_dir = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/images"

    # output file
    output_dir = "/work/gn53/k75057/projects/LLaVA-JP/results/laioncc-to-gpt"
    output_file = os.path.join(output_dir, "heron_bench.jsonl")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # vlmのモデルを読み込む
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_path = '/work/gn53/k75057/projects/LLaVA-JP/output_llava/checkpoints/finetune-llava-jp-1.3b-v1.1-laioncc-to-gpt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device=="cuda" else torch.float32

    model = LlavaGpt2ForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=1532,
        padding_side="right",
        use_fast=False,
    )
    model.eval()

    # 質問データを読み込む
    data_list = []
    with open(questions_file, "r") as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    # 質問に対して応答を生成する
    with open(output_file, "w") as f:
        for question_data in tqdm(data_list):
            question_id = question_data["question_id"]
            image_category = question_data["image_category"]
            image_name = question_data["image"]
            question = question_data["text"]

            # image pre-process
            image_filepath = os.path.join(image_dir, image_name)
            image = Image.open(image_filepath).convert("RGB")
            image_size = model.get_model().vision_tower.image_processor.size["height"]
            if model.get_model().vision_tower.scales is not None:
                image_size = model.get_model().vision_tower.image_processor.size["height"] * len(model.get_model().vision_tower.scales)
            
            if device == "cuda":
                image_tensor = model.get_model().vision_tower.image_processor(
                    image, 
                    return_tensors='pt', 
                    size={"height": image_size, "width": image_size}
                )['pixel_values'].half().cuda().to(torch_dtype)
            else:
                image_tensor = model.get_model().vision_tower.image_processor(
                    image, 
                    return_tensors='pt', 
                    size={"height": image_size, "width": image_size}
                )['pixel_values'].to(torch_dtype)

            # create prompt
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, 
                tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0)

            if device == "cuda":
                input_ids = input_ids.to(device)

            input_ids = input_ids[:, :-1] # </sep>がinputの最後に入るので削除する
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            streamer = TextStreamer(tokenizer, skip_prompt=True, timeout=20.0)

            # predict
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.1,
                    top_p=1.0,
                    max_new_tokens=256,
                    streamer=streamer,
                    use_cache=True,
                )

            output_ids = [token_id for token_id in output_ids.tolist()[0][input_ids.size(1):] if token_id != IMAGE_TOKEN_INDEX]
            output = tokenizer.decode(output_ids, skip_special_tokens=True)

            output_data = {
                "question_id": question_id,
                "images": image_name,
                "image_category": image_category,
                "prompt": prompt,
                "answer_id": "",
                "model_id": "laioncc-to-gpt",
                "metadata": {},
                "text": output,
            }

            print(output_data["text"])
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
