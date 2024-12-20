import json
from pathlib import Path

import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_data_paths = [
        {
            'path': '../output_llava/checkpoints/finetune-llava-jp-1.3b-v1.1-laioncc-to-gpt-w-1022k',
            'label': 'laioncc-to-gpt-w-1022k',
        },
        {
            'path': '../output_llava/checkpoints/finetune-llava-jp-1.3b-v1.1-laioncc-to-gpt',
            'label': 'laioncc-to-gpt',
        },
    ]

    train_losses = []
    for data_path in train_data_paths:
        with open(Path(data_path['path'], 'trainer_state.json')) as f:
            log_data = f.read()
            log_data_json = json.loads(log_data)

        train_losses.append([history['loss'] for history in log_data_json['log_history'][:-1]])

    for i, train_loss in enumerate(train_losses):
        plt.plot(train_loss, label=train_data_paths[i]['label'], linewidth=0.5)
    
    plt.title('Finetune Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig("finetune_laioncc.png")
