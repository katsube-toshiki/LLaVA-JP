import os
import shutil
import json
from pathlib import Path

if __name__ == '__main__':
    base_path = Path("/work/gn53/k75057/projects")

    marge_json_path1 = Path(base_path, "commoncrawl/data/output_json", 'llava_pretrain_1022k.json')
    marge_data_path1 = Path(base_path, "commoncrawl/data/images")

    marge_json_path2 = Path(base_path, "LLaVA-JP/dataset", 'llava_pretrain_blip_laion_cc_sbu_558k_ja.json')
    marge_data_path2 = Path(base_path, "LLaVA-JP/dataset/llava-pretrain")

    with marge_json_path1.open('r', encoding='utf-8') as f:
        caption_data = f.read()
        caption_data_json = json.loads(caption_data)
        print(caption_data_json)

    with marge_json_path2.open('r', encoding='utf-8') as f:
        caption_data = f.read()
        caption_data_json.extend(json.loads(caption_data))

    chat_ja_path = Path('llava_pretrain_stair.json')
    with open(chat_ja_path, mode="w") as f:
        json.dump(caption_data_json, f, indent=2, ensure_ascii=False)
    