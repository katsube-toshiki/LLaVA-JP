import json
import os
import csv

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

for i, csv_file in enumerate(os.listdir(csv_dir)):
    csv_path = os.path.join(csv_dir, csv_file)
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        j = 0
        for row in reader:
            json_data.append({
                "id": "{}{}".forrmat(str(i).zfill(8), row[0]),
                "image": row[1].replace("/work/gn53/k75057/projects/commoncrawl/data/images/", ""),
                "conversations": [
                    {
                        "from": "ユーザー",
                        "value": prompts[j % len(prompts)],
                    },
                    {
                        "from": "システム",
                        "value": row[2],
                    }
                ]
            })
            j += 1
    
print(json_data)