import torch
import json
import sys
import os
from tqdm import tqdm
from PIL import Image

from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')

def build_prompt(prompt="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    user_query = "与えられた画像について、詳細に述べてください。"
    msgs = [": \n" + user_query, ": "]
    if prompt:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + prompt)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

if __name__ == "__main__":
    # Input files
    questions_file = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/questions_ja.jsonl"
    image_dir = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/images"

    # output file
    output_file = "/work/gn53/k75057/projects/LLaVA-JP/results/japanese-instructblip-alpha/heron_bench.jsonl"

    # vlmのモデルを読み込む
    model = AutoModelForVision2Seq.from_pretrained("stabilityai/japanese-instructblip-alpha", trust_remote_code=True)
    processor = BlipImageProcessor.from_pretrained("stabilityai/japanese-instructblip-alpha")
    tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            prompt = question
            prompt = build_prompt(prompt)
            
            input_ids = processor(images=image, return_tensors="pt")
            text_encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
            text_encoding["qformer_attention_mask"] = text_encoding["attention_mask"].clone()
            input_ids.update(text_encoding)

            output_ids = model.generate(
                **input_ids.to(device, dtype=model.dtype),
                num_beams=5,
                max_new_tokens=32,
                min_length=1,
            )
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            output_data = {
                "question_id": question_id,
                "images": image_name,
                "image_category": image_category,
                "prompt": prompt,
                "answer_id": "",
                "model_id": "japanese-instructblip-alpha",
                "metadata": {},
                "text": output,
            }

            print(output_data["text"])
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
