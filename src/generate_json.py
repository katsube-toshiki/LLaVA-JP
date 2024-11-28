import json
import os
import csv

prompts = ['与えられた画像を表す簡潔な文を作成してください。\n<image>',
           '画像について簡単に説明してください。\n<image>',
           '与えられた画像について簡単に説明してください。\n<image>',
           '入力された写真について簡単に説明してください。\n<image>',
           '画像の内容を教えてください。\n<image>',
           '次の画像を短く分かりやすく説明してください。\n<image>',
           '与えられた画像について教えてください。\n<image>',
           '写真の特徴を手短に教えてください。\n<image>',
           'この画像について短い言葉で説明してください。\n<image>',
           '写真の概要を簡潔かつ分かりやすく伝えてください。\n<image>',
           '写真について簡単に説明してください。\n<image>',
           ]

csv_dir = "/work/gn53/k75057/projects/commoncrawl/data/output_csv"
json_dir = "/work/gn53/k75057/projects/commoncrawl/data/output_json"

json_data = []

for i, csv_file in enumerate(os.listdir(csv_dir)):
    csv_path = os.path.join(csv_dir, csv_file)
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        j = 0
        for row in reader:
            json_data.append({
                "id": "{}{}".format(str(i).zfill(5), row[0]),
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
    
json_filename = "llava_pretrain_{}k.json".format(int(len(json_data)/1000))
json_path = os.path.join(json_dir, json_filename)
with open(json_path, 'w') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)
    print("Generated JSON file: {}".format(json_path))

print("datasize: {}".format(len(json_data)))