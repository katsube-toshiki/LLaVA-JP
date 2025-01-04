import torch
import json
import sys
import os
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')
from metrics import compute_score


if __name__ == "__main__":

    # vlmのモデルを読み込む
    model = LlavaForConditionalGeneration.from_pretrained("cyberagent/llava-calm2-siglip", torch_dtype=torch.bfloat16).to(0)
    processor = AutoProcessor.from_pretrained("cyberagent/llava-calm2-siglip")

    # 推論モードに設定
    model.eval()

    # JA-VG-VQA-500 datasetを読み込む
    dataset = load_dataset("SakanaAI/JA-VG-VQA-500", split="test")

    # 保存用リスト
    questions = []
    answers = []
    predictions = []

    for data in tqdm(dataset):
        # image pre-process
        image = data['image']

        # それぞれの質問に対して応答を生成する
        for qa in data['qas']:
            question = qa['question']
            answer = qa['answer']

            questions.append(question)
            answers.append(answer)

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

    with open(os.path.join(save_dir, "ja_vg_vqa_500.json"), 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

