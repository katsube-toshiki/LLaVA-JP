import torch
import json
import sys
import os
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')
from metrics import compute_score


if __name__ == "__main__":

    # vlmのモデルを読み込む
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "SakanaAI/EvoVLM-JP-v1-7B"
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_id)
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
        prompt = "<image>\n{}".format(question)
        messages = [
            {"role": "system", "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、質問に答えてください。"},
            {"role": "user", "content": prompt},
        ]
        inputs = processor.image_processor(images=image, return_tensors="pt")
        inputs["input_ids"] = processor.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        )
        # 3. generate
        output_ids = model.generate(**inputs.to(device))
        output_ids = output_ids[:, inputs.input_ids.shape[1] :]
        output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

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

