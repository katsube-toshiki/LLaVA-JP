import torch
import transformers
import json

from transformers.generation.streamers import TextStreamer
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.llava_gpt2 import LlavaGpt2ForCausalLM
from llava.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llava.train.dataset import tokenizer_image_token

from ..utils import compute_score

from datasets import load_dataset

if __name__ == "__main__":

    # vlmのモデルを読み込む
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_path = '/work/gn53/k75057/projects/LLaVA-JP/output_llava/checkpoints/finetune-llava-jp-1.3b-v1.1-laioncc-w-1022k-to-gpt-w-1022k'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device=="cuda" else torch.float32

    model = LlavaGpt2ForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=1532,
        padding_side="right",
        use_fast=False,
    )
    model.eval()

    # JA-VG-VQA-500 datasetを読み込む
    dataset = load_dataset("SakanaAI/JA-VG-VQA-500", split="test")

    # 保存用リスト
    questions = []
    answers = []
    predictions = []

    for data in dataset:
        # image pre-process
        image = data['image']
        image_size = model.get_model().vision_tower.image_processor.size["height"]
        if model.get_model().vision_tower.scales is not None:
            image_size = model.get_model().vision_tower.image_processor.size["height"] * len(model.get_model().vision_tower.scales)
        
        if device == "cuda":
            image_tensor = model.get_model().vision_tower.image_processor(
                image, 
                return_tensors='pt', 
                size={"height": image_size, "width": image_size}
            )['pixel_values'].half().cuda().to(torch_dtype)
        else:
            image_tensor = model.get_model().vision_tower.image_processor(
                image, 
                return_tensors='pt', 
                size={"height": image_size, "width": image_size}
            )['pixel_values'].to(torch_dtype)

        # それぞれの質問に対して応答を生成する
        for qa in data['qas']:
            question = qa['question']
            answer = qa['answer']

            questions.append(question)
            answers.append(answer)

            # create prompt
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, 
                tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0)

            if device == "cuda":
                input_ids = input_ids.to(device)

            input_ids = input_ids[:, :-1] # </sep>がinputの最後に入るので削除する
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            streamer = TextStreamer(tokenizer, skip_prompt=True, timeout=20.0)

            # predict
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.1,
                    top_p=1.0,
                    max_new_tokens=256,
                    streamer=streamer,
                    use_cache=True,
                )
                output_ids = [token_id for token_id in output_ids.tolist()[0][input_ids.size(1):] if token_id != IMAGE_TOKEN_INDEX]
                output = tokenizer.decode(output_ids, skip_special_tokens=True)

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

    with open("/work/gn53/k75057/projects/LLaVA-JP/results/laioncc-w-1022k-to-gpt-w-1022k/ja_vg_vqa_500.json", 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

