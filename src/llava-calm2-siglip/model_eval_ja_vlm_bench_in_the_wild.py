import torch
import json
import sys
import os
from datasets import load_dataset

from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')
from metrics import compute_score


if __name__ == "__main__":

    # vlmのモデルを読み込む
    model = LlavaForConditionalGeneration.from_pretrained("cyberagent/llava-calm2-siglip", torch_dtype=torch.bfloat16).to(0)
    processor = AutoProcessor.from_pretrained("cyberagent/llava-calm2-siglip")

    # 推論モードに設定
    model.eval()

    # JA-VLM-Bench-In-the-Wildを読み込む
    dataset = load_dataset("SakanaAI/JA-VLM-Bench-In-the-Wild", split="test")

    # 保存用リスト
    questions = []
    answers = []
    predictions = []

    for data in dataset:
        # image pre-process
        image = data['image']

        # それぞれの質問に対して応答を生成する
        question = data['question']
        answer = data['answer']

        questions.append(question)
        answers.append(answer)

        # create prompt
        prompt = "USER: <image>\n"
        prompt += "この画像を説明してください。\n"
        prompt += "ASSISTANT: "

        input_ids = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.bfloat16)

        # predict
        with torch.inference_mode():
            output_ids = model.generate(
                **input_ids,
                max_length=500,
                do_sample=True,
                temperature=0.2,
                return_full_text = False,
            )
            output = processor.tokenizer.decode(output_ids[0][:-1], clean_up_tokenization_spaces=False)

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

