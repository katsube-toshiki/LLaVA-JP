import torch
import json
import sys
import os
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')
from metrics import compute_score

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

    # vlmのモデルを読み込む
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForVision2Seq.from_pretrained("stabilityai/japanese-stable-vlm", trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained("stabilityai/japanese-stable-vlm")
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/japanese-stable-vlm")
    model.to(device)

    # 推論モードに設定
    model.eval()

    # JA-VLM-Bench-In-the-Wildを読み込む
    dataset = load_dataset("SakanaAI/JA-VLM-Bench-In-the-Wild", split="test")

    # 保存用リスト
    questions = []
    answers = []
    predictions = []

    for data in tqdm(dataset):
        # image pre-process
        image = data['image']

        # それぞれの質問に対して応答を生成する
        question = data['question']
        answer = data['answer']

        questions.append(question)
        answers.append(answer)

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

        predictions.append(output)
                
    results = {
        "question": questions,
        "answer": answers,
        "prediction": predictions
    }

    # 評価
    metrics = compute_score(results)

    json_data = {
        "metrics": metrics,
        "results": results,
    }

    save_dir = "/work/gn53/k75057/projects/LLaVA-JP/results/llava-calm2-siglip"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "ja_vlm_bench_in_the_wild.json"), 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

