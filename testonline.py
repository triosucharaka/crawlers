from datasets import load_dataset
import numpy as np

dataset = load_dataset("chavinlo/tempofunk-s", streaming=True)

for split in dataset:
    #print(split)
    for e in dataset[split]:
        entry = e
        # list entry keys
        #print(entry.keys())
        #print(entry)
        print("desc: ", entry['description'])
        print("url: ", entry['videourl'])
        txt_embed = entry['prompt']
        print(type(txt_embed))
        txt_embed = np.array(txt_embed)
        #print("Text:", txt_embed)
        print("Text:", txt_embed.shape)
        print("Text:", txt_embed.dtype)
        vid_embed = entry['video']
        vid_embed = np.array(vid_embed)
        #print(vid_embed)
        single_frame = vid_embed[0]
        #print("Video:", single_frame)
        print("Video:", single_frame.shape)
        print("Video:", single_frame.dtype)
        print("#################")
        #exit()