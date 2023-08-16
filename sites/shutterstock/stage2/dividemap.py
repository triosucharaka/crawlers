import json
import argparse

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="The JSON file to split")
parser.add_argument("-d", "--divs", help="The number of divisions to split the file into")
args = parser.parse_args()

json_path = args.file
divs = int(args.divs)

# Load the JSON file
with open(json_path, "r") as json_file:
    data = json.load(json_file)

# Calculate the size of each chunk
chunk_size = len(data) // divs
if len(data) % divs != 0:
    chunk_size += 1
    
# Split the data and write to different files
for i in range(divs):
    start = i*chunk_size
    end = (i+1)*chunk_size if (i+1)*chunk_size < len(data) else len(data)
    
    chunk = data[start:end]
    with open(f"output{i+1}.json", "w") as json_file:
        json.dump(chunk, json_file)
