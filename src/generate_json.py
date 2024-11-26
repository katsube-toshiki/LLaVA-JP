import json
import os
import pandas as pd

prompts = ['写真の概要を簡潔に説明してください。\n<image>',
           '写真の簡潔な要約を書いてください。\n<image>',
           'これは何ですか？\n<image>',
           '写真には何が写っていますか？\n<image>',
           'イメージを簡潔に説明してください。\n<image>',
           '提供された画像の簡潔な解釈を共有してください。\n<image>',
           '画像の簡単な説明をしてください。\n<image>',
           '写真の主な特徴をコンパクトに説明してください。\n<image>',
           '与えられた画像について簡単な説明をしてください。\n<image>',
           '画像の内容を要約してください。\n<image>',
           'その後に続く画像について、簡潔でわかりやすい説明をしてください。\n<image>'
           ]

csv_dir = "/work/gn53/k75057/projects/commoncrawl/data/output_csv"
json_dir = "/work/gn53/k75057/projects/commoncrawl/data/output_json"

if not os.path.exists(json_dir):
    os.makedirs(json_dir)

json_data = []

for csv_file in os.listdir(csv_dir):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)
        print(df)
        
        # for index, row in df.iterrows():
        #     for prompt in prompts:
        #         json_data.append({
        #             'prompt': prompt,
        #             'image_path': row['image_path'],
        #             'caption': row['caption']
        #         })

print(json_data)
        
        # json_file = os.path.join(json_dir, os.path.splitext(csv_file)[0] + '.json')
        # with open(json_file, 'w', encoding='utf-8') as f:
        #     json.dump(json_data, f, ensure_ascii=False, indent=4)
