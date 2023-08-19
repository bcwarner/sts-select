# Wrappers for baseline models so they fit the SBERT API
import os.path

from gensim.models import Word2Vec, FastText
from tqdm import tqdm


class SkipgramModel:
    def __init__(self, model_path=None):
        self.model = None
        if model_path is not None:
            self.model = Word2Vec.load(os.path.join(model_path, "model.bin"))
    def fit(self, train_objectives=None, epochs=None, warmup_steps=None):
        # Called by trainer to fit the model.
        dataloader = train_objectives[0][0]
        self.model = Word2Vec(min_count=1, workers=8, window=3, sg=1)
        sentences = []
        for sentence in dataloader:
            sentences.append(sentence)

        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=len(sentences), epochs=epochs)

    def encode(self, sentences, batch_size=1, show_progress_bar=False):
        output = []
        iterator = sentences if not show_progress_bar else tqdm(sentences, desc="Batches")
        # Batch size not relevant for this model
        for sentence in iterator:
            output.append(self.model.wv[sentence])
        return output

    def save(self, path):
        self.model.save(os.path.join(path, "model.bin"))

class FastTextModel:
    def __init__(self, model_path=None):
        self.model = None
        if model_path is not None:
            self.model = FastText.load(os.path.join(model_path, "model.bin"))

    def fit(self, train_objectives=None, epochs=None, warmup_steps=None):
        dataloader = train_objectives[0][0]
        self.model = FastText()
        # Iterate through dataset and build a vocab.
        sentences = []
        for sentence in dataloader:
            sentences.append(sentence)

        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=len(sentences), epochs=epochs)

    def encode(self, sentences, batch_size=1, show_progress_bar=False):
        output = []

        iterator = sentences if not show_progress_bar else tqdm(sentences, desc="Batches")
        # Batch size not relevant for this model
        for sentence in iterator:
            output.append(self.model.wv[sentence])
        return output

    def save(self, path):
        self.model.save(os.path.join(path, "model.bin"))

