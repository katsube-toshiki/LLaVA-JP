import torch
import json
import sys
import os
from datasets import load_dataset
from tqdm import tqdm

from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')
from metrics import compute_score

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

    # vlmのモデルを読み込む
    model = AutoModelForVision2Seq.from_pretrained("stabilityai/japanese-instructblip-alpha", trust_remote_code=True)
    processor = BlipImageProcessor.from_pretrained("stabilityai/japanese-instructblip-alpha")
    tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    save_dir = "/work/gn53/k75057/projects/LLaVA-JP/results/japanese-instructblip-alpha"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "ja_vlm_bench_in_the_wild.json"), 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print("save results to", save_dir)