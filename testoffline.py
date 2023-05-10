from datasets import load_dataset
import torch

dataset = load_dataset("chavinlo/test1", streaming=False, revision="testing-3")

for split in dataset:
    print(split)
    for entry in dataset[split]:
        print(entry)
        txt_embed = torch.load(open(entry['prompt_binary'], 'rb'))
        print("Text:", txt_embed)
        vid_embed = torch.load(open(entry['frames_binary'], 'rb'))
        print("Video:", type(vid_embed))