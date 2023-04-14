from datasets import load_dataset
from datasets import load_dataset_builder

dataset = load_dataset("chavinlo/tempofunk", streaming=True)

sample = dataset['00000'].take(2)

for entry in sample:
    print(entry['tokenized_prompt'])
