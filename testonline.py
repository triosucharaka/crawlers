from datasets import load_dataset
import torch

dataset = load_dataset("chavinlo/tempofunk", streaming=False, revision="testing-3")

for split in dataset:
    print(split)
    for e in dataset[split]:
        entry = e['embeddings']
        # list entry keys
        print(entry.keys())
        print(entry)
        txt_embed = torch.load(open(entry['prompt'], 'rb')).cpu().detach().numpy()
        print("Text:", txt_embed)
        print("Text:", txt_embed.shape)
        print("Text:", txt_embed.dtype)
        vid_embed = torch.load(open(entry['frames'], 'rb'))
        single_frame = vid_embed[0]['mean']
        print("Video:", single_frame)
        print("Video:", single_frame.shape)
        print("Video:", single_frame.dtype)

        exit()