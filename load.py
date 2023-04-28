import datasets
import json
import numpy

_FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "prompt": datasets.Array3D(shape=(1, 77, 768), dtype="float32"),
        "video": datasets.Sequence(feature=datasets.Array3D(shape=(4, 64, 64), dtype="float64")),
        "description": datasets.Value("string"),
        "videourl": datasets.Value("string"),
        "categories": datasets.Value("string"),
        "duration": datasets.Value("float"),
        "full_metadata": datasets.Value("string"),
    }
)

class FunkLoaderStream(datasets.GeneratorBasedBuilder):
    """TempoFunk Dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description="TempoFunk Dataset",
            features=_FEATURES,
            homepage="None",
            citation="None",
            license="None"
        )

    def _split_generators(self, dl_manager):

        print("id_list available at:", dl_manager.download("data/id_list.json"))

        _ID_LIST = json.loads(open(dl_manager.download("data/id_list.json"), 'r').read())
        
        _SHARD_LENGTH = 20

        _SPLITS = [_ID_LIST[i:i + _SHARD_LENGTH] for i in range(0, len(_ID_LIST), _SHARD_LENGTH)]

        print("avail splits: ", _SPLITS)
        

        l=[]

        _split_count = 0

        for split in _SPLITS:

            _list = []

            for video_id in split:
                _list.append({
                    "frames": dl_manager.download(f"data/videos/{video_id}.npy"),
                    "prompt": dl_manager.download(f"data/prompts/{video_id}.npy"),
                    "metadata": dl_manager.download(f"data/metadata/{video_id}.json"),
                })

            l.append(
                datasets.SplitGenerator(
                    name=f"split_{_split_count}",
                    gen_kwargs={
                        "chunk_container": _list,
                    },)
            )

            _split_count = _split_count + 1

        print("Total Splits: ", _split_count)
        
        return l
    
    def _generate_examples(self, chunk_container):
        """Generate images and labels for splits."""
        for video_entry in chunk_container:
            frames_binary = video_entry['frames']
            prompt_binary = video_entry['prompt']
            metadata = json.loads(open(video_entry['metadata'], 'r').read())

            txt_embed = numpy.load(prompt_binary)
            vid_embed = numpy.load(frames_binary)

            print(vid_embed.shape)

            yield metadata['id'], {
                "id": metadata['id'],
                "description": metadata['description'],
                "prompt": txt_embed,
                "video": vid_embed,
                "videourl": metadata['videourl'],
                "categories": metadata['categories'],
                "duration": metadata['duration'],
                "full_metadata": metadata
            }