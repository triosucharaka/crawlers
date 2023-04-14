import datasets
from io import BytesIO
import numpy as np

_TAR_FILES=[
    "data/00000.tar",
    "data/00001.tar",
    "data/00002.tar",
    "data/00003.tar",
    "data/00004.tar",
    "data/00005.tar",
    "data/00006.tar",
    "data/00007.tar",
    "data/00008.tar",
    "data/00009.tar",
    ]

_TAR_FILES_DICT={
    "00000": "data/00000.tar",
    "00001": "data/00001.tar",
    "00002": "data/00002.tar",
    "00003": "data/00003.tar",
    "00004": "data/00004.tar",
    "00005": "data/00005.tar",
    "00006": "data/00006.tar",
    "00007": "data/00007.tar",
    "00008": "data/00008.tar",
    "00009": "data/00009.tar",
    }

class FunkLoader(datasets.GeneratorBasedBuilder):
    """TempoFunk Dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description="TempoFunk Dataset",
            homepage="None",
            citation="Terry A. Davis",
            license="PlusNi-"
        )

    def _split_generators(self, dl_manager):
        
        l=[]

        for k in _TAR_FILES_DICT.keys():
            archive_path = dl_manager.download(_TAR_FILES_DICT[k])
            l.append(
                datasets.SplitGenerator(
                name=k,
                gen_kwargs={
                    "npy_files": dl_manager.iter_archive(archive_path),
                },)
            )
            
        return l

    def _generate_examples(self, npy_files):
        """Generate images and labels for splits."""
        for file_path, file_obj in npy_files:
            # NOTE: File object is (ALREADY) opened in binary mode.
            numpy_bytes = file_obj.read()
            numpy_dict = np.load(BytesIO(numpy_bytes), allow_pickle=True)

            reconverted_dict = {
                "frames": numpy_dict.item().get("frames"),
                "prompt": numpy_dict.item().get("prompt")
            }

            yield file_path, {
                "tokenized_prompt": reconverted_dict['prompt'],
                "video": reconverted_dict['frames']
                }