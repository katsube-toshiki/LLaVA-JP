import torch
import json
import sys
import os
from tqdm import tqdm
from PIL import Image

from heron.models.git_llm.git_japanese_stablelm_alpha import GitJapaneseStableLMAlphaForCausalLM
from transformers import AutoProcessor, LlamaTokenizer

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')


if __name__ == "__main__":
    # Input files
    questions_file = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/questions_ja.jsonl"
    image_dir = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/images"

    # output file
    output_file = "/work/gn53/k75057/projects/LLaVA-JP/results/heron-chat-git-ja-stablelm-base-7b-v1/heron_bench.jsonl"

    # vlmのモデルを読み込む
    device_id = 0
    device = f"cuda:{device_id}"

    MODEL_NAME = "turing-motors/heron-chat-git-ja-stablelm-base-7b-v1"
        
    model = GitJapaneseStableLMAlphaForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, ignore_mismatched_sizes=True
    )
    model.eval()
    model.to(device)
    
    # prepare a processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = LlamaTokenizer.from_pretrained(
        "novelai/nerdstash-tokenizer-v1",
        padding_side="right",
        additional_special_tokens=["▁▁"],
    )
    processor.tokenizer = tokenizer

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
            prompt = f"##human: {question}\n##gpt: "

            # do preprocessing
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                truncation=True,
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # do inference
            with torch.no_grad():
                out = model.generate(**inputs, max_length=256, do_sample=False, temperature=0., no_repeat_ngram_size=2)

            output_ids = out[:, inputs['input_ids'].shape[1] :]
            output = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            output_data = {
                "question_id": question_id,
                "images": image_name,
                "image_category": image_category,
                "prompt": prompt,
                "answer_id": "",
                "model_id": "heron-chat-git-ja-stablelm-base-7b-v1",
                "metadata": {},
                "text": output,
            }

            print(output_data["text"])
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
