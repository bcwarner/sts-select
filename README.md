# `sts-select`
[![PyPI - Version](https://img.shields.io/pypi/v/sts-select)](https://pypi.org/project/sts-select/)
[![Hugging Face](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/papers/2308.09892)
[![arXiv](https://img.shields.io/badge/arXiv-2308.09892-b31b1b.svg)](https://arxiv.org/abs/2308.09892)
[![PyPI - License](https://img.shields.io/pypi/l/sts-select)](https://pypi.org/project/sts-select/)
## Overview
Small datasets often require feature selection to ensure generalizability, but such feature selection methods rely on the information provided in the dataset, which may not be enough to make good selections. An underutilized source of information is the similarity between the feature and target names, and we find that utilizing feature names in the selection process can sometimes improve the performance of feature selection methods. This is probably due to the fact that statistical measures of feature and target relationships are often noisy/incomplete, and the feature/target names can provide a more consistent measure of any relationship.

This package provides some Python tools to implement STS-based feature selection using fine-tuned models that you've trained either with [`sentence_transformers`](https://www.sbert.net/) or [Gensim](https://radimrehurek.com/gensim/). You can install it with `pip install sts-select`.

## Usage
There are several steps to using this package. The first is to fine-tune a language model to produce semantic textual similarity (STS) scores. After that, you can use these scores to select features using either one of the selection methods provided or your own. From there you can apply this feature selection method to a dataset.

### Fine-Tuning a Model
The first step is to fine-tune a model to produce STS scores. An example of how to obtain fine-tuning datasets for STS scores is can be found in `examples/ppsp/redcap_sts_scorers/data.py`. An example of how to train these models can be found in `examples/ppsp/redcap_sts_scorers/train.py`. We fine-tune our models using the `sentence_transformers` package and Gensim, but any other framework can be easily adapted to this library.

### Scoring

Once you have a fine-tuned model, you can use it to score features, either by itself, or in combination with other scoring measures or models.

Here's a brief snippet of what this looks like in brief. A comprehensive example can be found in `examples/ppsp/redcap/train.py`.

```
pipe = Pipeline([
    ("selector", MRMRBase(
        SentenceTransformerScorer(
            X,
            y,
            X_names=X_names,
            y_names=y_names,
            model_path="bcwarner/PubMedBERT-base-uncased-sts-combined", # Example model. 
            cache="your/cache/path", # Location to save STS scores for repeated runs.
        ), 
        n_features=20
    )),
    ("classifier", MLPClassifier(activation="relu", alpha=1))
])

pipe.fit(X, y)
```

### Example

We provide a refactored version of our code used in our findings in the `examples/ppsp` directory for a comprehensive overview of how to use our package, along with a README that explains how to use it if you wish to start from there.

## Citing

If you use this code in your research, please cite the following preprint:

```
@misc{warner2023utilizing,
      title={Utilizing Semantic Textual Similarity for Clinical Survey Data Feature Selection}, 
      author={Benjamin C. Warner and Ziqi Xu and Simon Haroutounian and Thomas Kannampallil and Chenyang Lu},
      year={2023},
      eprint={2308.09892},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
