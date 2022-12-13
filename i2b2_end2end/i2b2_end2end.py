import os
import datasets


_DIR = "data"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "valid.txt"
_TEST_FILE = "test.txt"


class I2B2End2EndConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for I2B2End2End.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(I2B2End2EndConfig, self).__init__(**kwargs)


class I2B2End2End(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        I2B2End2EndConfig(name="i2b2_end2end", version=datasets.Version("1.0.0"), description="i2b2_end2end dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "sentence_id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dir_path = _DIR
        data_files = {
            "train": os.path.join(dir_path, _TRAINING_FILE),
            "dev": os.path.join(dir_path, _DEV_FILE),
            "test": os.path.join(dir_path, _TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        yield guid, {
                            "sentence_id": str(guid),
                            "words": words,
                            "labels": labels,
                        }
                        guid += 1
                        words = []
                        labels = []
                else:
                    # conll2003 words are space separated
                    splits = line.split("\t")
                    words.append(splits[0])
                    labels.append(splits[-1].rstrip())
            # last example
            yield guid, {
                "sentence_id": str(guid),
                "words": words,
                "labels": labels,
            }


