import torch
import json
import sys
import os
from tqdm import tqdm
from PIL import Image

from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')

# helper function to format input prompts
TASK2INSTRUCTION = {
    "caption": "画像を詳細に述べてください。",
    "tag": "与えられた単語を使って、画像を詳細に述べてください。",
    "vqa": "与えられた画像を下に、質問に答えてください。",
}

def build_prompt(task="caption", input=None, sep="\n\n### "):
    assert (
        task in TASK2INSTRUCTION
    ), f"Please choose from {list(TASK2INSTRUCTION.keys())}"
    if task in ["tag", "vqa"]:
        assert input is not None, "Please fill in `input`!"
        if task == "tag" and isinstance(input, list):
            input = "、".join(input)
    else:
        assert input is None, f"`{task}` mode doesn't support to input questions"
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    instruction = TASK2INSTRUCTION[task]
    msgs = [": \n" + instruction, ": \n"]
    if input:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + input)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

if __name__ == "__main__":
    # Input files
    questions_file = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/questions_ja.jsonl"
    image_dir = "/work/gn53/k75057/projects/heron/playground/data/japanese-heron-bench/images"

    # output file
    output_file = "/work/gn53/k75057/projects/LLaVA-JP/results/japanese-stable-vlm/heron_bench.jsonl"

    # vlmのモデルを読み込む
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForVision2Seq.from_pretrained("stabilityai/japanese-stable-vlm", trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained("stabilityai/japanese-stable-vlm")
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/japanese-stable-vlm")
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
            prompt = build_prompt(task="vqa", input=question)

            input_ids = processor(images=image, return_tensors="pt")
            text_encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            input_ids.update(text_encoding)

            # predict
            output_ids = model.generate(
                **input_ids.to(device, dtype=model.dtype),
                do_sample=False,
                num_beams=5,
                max_new_tokens=128,
                min_length=1,
                repetition_penalty=1.5,
            )
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            output_data = {
                "question_id": question_id,
                "images": image_name,
                "image_category": image_category,
                "prompt": prompt,
                "answer_id": "",
                "model_id": "japanese-stable-vlm",
                "metadata": {},
                "text": output,
            }

            print(output_data["text"])
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
