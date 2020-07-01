import collections
import tensorflow as tf
from glearn.datasets.sequence import Vocabulary, SequenceDataset
from glearn.utils.download import ensure_download


TRANSLATIONS = {
    # "en/sp": "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    "en-de": {
        # "url": "https://drive.google.com/open?id=0B_bZck-ksdkpM25jRUN2X2UxMm8",
        "url": "/Users/glennpowell/data/datasets/nmt/wmt16_en_de.tar.gz",
        "vocab.source": "vocab.bpe.32000",
        "vocab.source": "vocab.bpe.32000",
        "train.source": "train.tok.clean.bpe.32000.en",
        "train.target": "train.tok.clean.bpe.32000.de",
        "test.source": "newstest2013.tok.bpe.32000.en",
        "test.target": "newstest2013.tok.bpe.32000.de",
    },
}


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    return Vocabulary(words)


def _file_to_word_ids(filename, vocabulary):
    data = _read_words(filename)
    return vocabulary.encipher(data)


def _load_data(languages):
    language_map = "-".join(languages)
    data_path = SequenceDataset.get_data_path(f"nmt/{language_map}")
    translation = TRANSLATIONS[language_map]

    zip_data = ensure_download(translation["url"], data_path, extract=True)
    # TODO
    return None


def nmt_dataset(config, languages=["en", "de"]):
    _load_data(languages)
    # data = {
    #     "train": train_data,
    #     "validate": valid_data,
    #     "test": test_data,
    # }

    # batch_size = config.batch_size
    # timesteps = config.get("timesteps", 35)

    # return SequenceDataset("NMT", data, batch_size, vocabulary, timesteps)
    return None
