import json
import collections

with open("./dataset/llava_pretrain_blip_laion_cc_sbu_558k_ja.json") as f:
    data = json.load(f)
    prompts = []
    for row in data:
        prompts.append(row['conversations'][0]['value'])

    c = collections.Counter(prompts)
    print(data[0])
