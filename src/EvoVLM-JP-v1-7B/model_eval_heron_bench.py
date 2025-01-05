import torch
import json
import sys
import os
from tqdm import tqdm
from PIL import Image

from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')


if __name__ == "__main__":
    # Input files
    questions_file = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/questions_ja.jsonl"
    image_dir = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/images"

    # output file
    output_file = "/work/gn53/k75057/projects/LLaVA-JP/results/EvoVLM-JP-v1-7B/heron_bench.jsonl"

    # vlmのモデルを読み込む
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "SakanaAI/EvoVLM-JP-v1-7B"
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    model.to(device)

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
            prompt = "<image>\n{}".format(question)
            messages = [
                {"role": "system", "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、質問に答えてください。"},
                {"role": "user", "content": prompt},
            ]
            inputs = processor.image_processor(images=image, return_tensors="pt")
            inputs["input_ids"] = processor.tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            )
            # generate
            output_ids = model.generate(**inputs.to(device))
            output_ids = output_ids[:, inputs.input_ids.shape[1] :]
            output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            output_data = {
                "question_id": question_id,
                "images": image_name,
                "image_category": image_category,
                "prompt": prompt,
                "answer_id": "",
                "model_id": "EvoVLM-JP-v1-7B",
                "metadata": {},
                "text": output,
            }

            print(output_data["text"])
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
