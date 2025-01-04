import torch
import json
import sys
import os
from datasets import load_dataset
from tqdm import tqdm

from heron.models.git_llm.git_japanese_stablelm_alpha import GitJapaneseStableLMAlphaForCausalLM
from transformers import AutoProcessor, LlamaTokenizer

sys.path.append('/work/gn53/k75057/projects/LLaVA-JP/src')
from metrics import compute_score


if __name__ == "__main__":

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
        text = f"##human: {question}\n##gpt: "

        # do preprocessing
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
            truncation=True,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # do inference
        with torch.no_grad():
            out = model.generate(**inputs, max_length=256, do_sample=False, temperature=0., no_repeat_ngram_size=2)

        # print result
        output = processor.tokenizer.batch_decode(out)

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

    save_dir = "/work/gn53/k75057/projects/LLaVA-JP/results/heron-chat-git-ja-stablelm-base-7b-v1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "ja_vlm_bench_in_the_wild.json"), 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print("save results to", save_dir)

