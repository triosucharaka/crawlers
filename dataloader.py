import datasets
import json
import numpy
import tarfile
import io

_FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "prompt": datasets.Array3D(shape=(1, 77, 768), dtype="float32"),
        "video": datasets.Sequence(feature=datasets.Array3D(shape=(4, 64, 64), dtype="float32")),
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
        # Load the chunk list.
        _CHUNK_LIST = json.loads(open(dl_manager.download("lists/chunk_list.json"), 'r').read())

        # Create a list to hold the downloaded chunks.
        _list = []

        # Download each chunk file.
        for chunk in _CHUNK_LIST:
           _list.append(dl_manager.download(f"data/{chunk}.tar"))

        # Return the list of downloaded chunks.
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "chunks": _list,
                },
            ),
        ]
    
    def _generate_examples(self, chunks):
        """Generate images and labels for splits."""
        for chunk in chunks:
            tar_data = open(chunk, 'rb')
            tar_bytes = tar_data.read()
            tar_bytes_io = io.BytesIO(tar_bytes)

            response_dict = {}

            with tarfile.open(fileobj=tar_bytes_io, mode='r') as tar:
                for file_info in tar:
                    if file_info.isfile():
                        file_name = file_info.name
                        #filename format is typ_id.ext
                        file_type = file_name.split('_')[0]
                        file_id = file_name.split('_')[1].split('.')[0]
                        file_ext = file_name.split('_')[1].split('.')[1]
                        file_contents = tar.extractfile(file_info).read()

                        if file_type == 'txt' or file_type == 'vid':
                            response_dict[file_id][file_type] = numpy.load(file_contents)
                        elif file_type == 'jso':
                            response_dict[file_id][file_type] = json.loads(file_contents)
            
            for key, value in response_dict.items():
                yield key, {
                    "id": key,
                    "description": value['jso']['description'],
                    "prompt": value['txt'],
                    "video": value['vid'],
                    "videourl": value['jso']['videourl'],
                    "categories": value['jso']['categories'],
                    "duration": value['jso']['duration'],
                    "full_metadata": value['jso']
                }