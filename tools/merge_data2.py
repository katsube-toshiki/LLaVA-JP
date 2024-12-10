import os
import shutil
import json
from pathlib import Path

if __name__ == '__main__':
    base_path = Path("/work/gn53/k75057/projects")
    new_base_path = Path("/work/gn53/k75057/projects/LLaVA-JP/dataset")

    marge_json_path1 = Path(base_path, "commoncrawl/data/output_json", 'llava_pretrain_1022k.json')
    marge_data_path1 = Path(base_path, "commoncrawl/data/images")

    marge_json_path2 = Path(base_path, "LLaVA-JP/dataset", 'llava_v1_5_instruct_620k_ja_v2.json')
    marge_data_path2 = Path(base_path, "LLaVA-JP/dataset")

    with marge_json_path1.open('r', encoding='utf-8') as f:
        caption_data = f.read()
        caption_data_json1 = json.loads(caption_data)
        for i in range(len(caption_data_json1)):
            caption_data_json1[i]['image'] = os.path.relpath(Path(marge_data_path1, caption_data_json1[i]["image"]), new_base_path)

    with marge_json_path2.open('r', encoding='utf-8') as f:
        caption_data = f.read()
        caption_data_json2 = json.loads(caption_data)
        for i in range(len(caption_data_json2)):
            caption_data_json2[i]['image'] = os.path.relpath(Path(marge_data_path2, caption_data_json2[i]["image"]), new_base_path)

    caption_data_json = caption_data_json1 + caption_data_json2

    chat_ja_path = Path(new_base_path, 'gpt_w_1022k.json')
    with open(chat_ja_path, mode="w") as f:
        json.dump(caption_data_json, f, indent=2, ensure_ascii=False)
    