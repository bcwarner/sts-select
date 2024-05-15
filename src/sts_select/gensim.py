# Wrappers for baseline models so they fit the SBERT API
import os.path

from gensim.models import FastText, Word2Vec
from tqdm import tqdm


class SkipgramModel:
    """
    Wrapper class for a Skipgram model that can be used with `~sts_select.SentenceTransformerScorer`.
    """
    def __init__(self, model_path=None):
        """
        :param model_path: Path to a saved model. If None, a new model will be created.
        """
        self.model = None
        if model_path is not None:
            self.model = Word2Vec.load(os.path.join(model_path, "model.bin"))

    def fit(self, train_objectives=None, epochs=None, warmup_steps=None, worker_count=-1):
        """
        Fits the model to the given sentence pair/similarity dataset.

        :param train_objectives: List of tuples. Each tuple contains a dataloader and a loss function.
        :param epochs: Number of epochs to train the model.
        :param warmup_steps: Number of warmup steps.
        """
        dataloader = train_objectives[0][0]
        self.model = Word2Vec(min_count=1, workers=-1, window=3, sg=1)
        sentences = []
        for sentence in dataloader:
            sentences.append(sentence)

        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=len(sentences), epochs=epochs)

    def encode(self, sentences, batch_size=1, show_progress_bar=False):
        """
        Prepares sentence bpairs for the model and returns the embeddings.

        :param sentences: List of sentences to encode.
        """
        output = []
        iterator = (
            sentences if not show_progress_bar else tqdm(sentences, desc="Batches")
        )
        # Batch size not relevant for this model
        for sentence in iterator:
            output.append(self.model.wv[sentence])
        return output

    def save(self, path):
        """
        Saves the model to the specified path.

        :param path: Path to save the model to.
        """
        self.model.save(os.path.join(path, "model.bin"))


class FastTextModel:
    """
    Wrapper class for a FastText model that can be used with `~sts_select.SentenceTransformerScorer`.
    """
    def __init__(self, model_path=None):
        """
        :param model_path: Path to a saved model. If None, a new model will be created.
        """
        self.model = None
        if model_path is not None:
            self.model = FastText.load(os.path.join(model_path, "model.bin"))

    def fit(self, train_objectives=None, epochs=None, warmup_steps=None):
        """
        Fits the model to the given sentence pair/similarity dataset.

        :param train_objectives: List of tuples. Each tuple contains a dataloader and a loss function.
        :param epochs: Number of epochs to train the model.
        :param warmup_steps: Number of warmup steps.
        """
        dataloader = train_objectives[0][0]
        self.model = FastText()
        # Iterate through dataset and build a vocab.
        sentences = []
        for sentence in dataloader:
            sentences.append(sentence)

        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=len(sentences), epochs=epochs)

    def encode(self, sentences, batch_size=1, show_progress_bar=False):
        """
        Prepares sentence bpairs for the model and returns the embeddings.

        :param sentences: List of sentences to encode.
        """
        output = []

        iterator = (
            sentences if not show_progress_bar else tqdm(sentences, desc="Batches")
        )
        # Batch size not relevant for this model
        for sentence in iterator:
            output.append(self.model.wv[sentence])
        return output

    def save(self, path):
        """
        Saves the model to the specified path.

        :param path: Path to save the model to.
        """
        self.model.save(os.path.join(path, "model.bin"))
