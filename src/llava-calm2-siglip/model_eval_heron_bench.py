import torch
import json
import sys
import os
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')


if __name__ == "__main__":
    # Input files
    questions_file = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/questions_ja.jsonl"
    image_dir = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/images"

    # output file
    output_file = "/work/gn53/k75057/projects/LLaVA-JP/results/llava-calm2-siglip/heron_bench.jsonl"

    # vlmのモデルを読み込む
    model = LlavaForConditionalGeneration.from_pretrained("cyberagent/llava-calm2-siglip", torch_dtype=torch.bfloat16).to(0)
    processor = AutoProcessor.from_pretrained("cyberagent/llava-calm2-siglip")

    # 推論モードに設定
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

            # create prompt
            prompt = "USER: <image>\n"
            prompt += "{}\n".format(question)
            prompt += "ASSISTANT: "

            input_ids = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.bfloat16)

            # predict
            output_ids = model.generate(
                **input_ids,
                max_length=500,
                do_sample=True,
                temperature=0.2,
            )
            output = processor.tokenizer.decode(output_ids[0][input_ids['input_ids'].size(1):-1], clean_up_tokenization_spaces=False)

            output_data = {
                "question_id": question_id,
                "images": image_name,
                "image_category": image_category,
                "prompt": prompt,
                "answer_id": "",
                "model_id": "llava-calm2-siglip",
                "metadata": {},
                "text": output,
            }

            print(output_data["text"])
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
