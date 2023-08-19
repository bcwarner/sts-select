# Trains a generic STS scorer on the STS benchmark in the dataset.
# Also saves the result for later calling.
import argparse
import itertools
import os
import pickle
# Add the current directory to the path
import sys
from collections import defaultdict
from typing import Dict, Tuple

import datasets as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sentence_transformers
import tikzplotlib
import torch
import transformers as tr
from sentence_transformers import (LoggingHandler, SentencesDataset,
                                   SentenceTransformer, losses, models, util)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from data import CLINICAL_VOCAB_NAME, COMBINED_VOCAB_NAME, GEN_VOCAB_NAME
from gensim import FastTextModel, SkipgramModel

RANDOM_STATE = 424989

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

stage_format = "==== {} ===="

old_env_vars = {}


def set_env_vars():
    # Set $TORCH_HOME to a torch subdirectory, and $TRANSFORMERS_CACHE to a transformers subdirectory
    global old_env_vars
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "torch")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "torch"))
    old_env_vars["TORCH_HOME"] = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "torch")

    if not os.environ.get("TRANSFORMERS_CACHE"):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "transformers")):
            os.mkdir(os.path.join(os.path.dirname(__file__), "transformers"))
    old_env_vars["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(
        os.path.dirname(__file__), "transformers"
    )


def reset_env_vars():
    global old_env_vars
    for k, v in old_env_vars.items():
        if v is None:
            del os.environ[k]
        os.environ[k] = v


def model_path(dset_name, model_name):
    return os.path.join(os.path.dirname(__file__), f"models/{model_name}/{dset_name}/")


dset_options = [CLINICAL_VOCAB_NAME, GEN_VOCAB_NAME, COMBINED_VOCAB_NAME]
model_options = [
    "bert-base-uncased",  # General LLMs
    # "allenai/scibert_scivocab_uncased",
    "sentence-transformers/all-MiniLM-L12-v2",
    # "sentence-transformers/all-mpnet-base-v2",
    # "sentence-transformers/all-distilroberta-v1",
    # Clinical LLMs
    "emilyalsentzer/Bio_ClinicalBERT",
    "allenai/biomed_roberta_base",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    # "microsoft/biogpt", Unable to reserve a card big enough for this
    # "AshtonIsNotHere/GatorTron-OG",
    "other/Skipgram",
    "other/FastText",
]

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--opt_params", type=str, default="")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", type=bool, default=False)
    args = parser.parse_args()

    # Set environment variables
    set_env_vars()

    losses_results = []

    for dset_name, model_name in itertools.product(dset_options, model_options):
        # Does this model exist? Yes => Skip if desired
        save_location = model_path(dset_name, model_name)
        if os.path.exists(save_location) and not args.overwrite:
            print(
                f"Model {model_name} already exists for dataset {dset_name}. Skipping."
            )
            continue

        # Load the dataset
        print(stage_format.format(f"Loading Dataset {dset_name}"))

        train_data = ds.load_from_disk(
            os.path.join(os.path.dirname(__file__), dset_name)
        )

        # Prepare the model
        print(stage_format.format("Preparing Model"))
        if "other" not in model_name:
            model = SentenceTransformer(model_name, device=args.device)
            # Convert data to InputExamples
            print(stage_format.format("Converting Data to InputExamples"))
            mapper = lambda x: InputExample(
                texts=[x["sent1"], x["sent2"]], label=x["score"]
            )
            train_data_mapped = [mapper(x) for x in train_data]
            train_data_mapped = SentencesDataset(train_data_mapped, model=model)
            train_dataloader = DataLoader(
                train_data_mapped, shuffle=True, batch_size=args.batch_size
            )
            train_loss = losses.CosineSimilarityLoss(model)
        else:
            if dset_name != GEN_VOCAB_NAME:
                # Only do the general vocab. Full sentences do not appear in the clinical set and loss is questionable.
                continue
            model = eval(model_name.split("/")[1] + "Model")()
            # For the two baselines, they are unsupervised, and so we just feed them in as-is
            train_dataloader = [x["sent1"] for x in train_data] + [
                x["sent2"] for x in train_data
            ]
            train_loss = None  # Not used
            train_data_mapped = None  # Not used

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=100,
        )

        if "other" not in model_name:
            # Evaluate the final loss on the training set
            print(stage_format.format("Evaluating Model"))
            # Evaluator
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                train_data_mapped, batch_size=args.batch_size, write_csv=False
            )
            # Evaluate
            eval_loss = evaluator(model, train_dataloader)
        else:
            eval_loss = model.model.get_latest_training_loss()

        print(f"Loss for {dset_name}/{model_name}: {eval_loss}")
        losses_results.append(
            [{"Dataset": dset_name, "Model": model_name, "Loss": eval_loss}]
        )

        # Save the model
        print(stage_format.format("Saving Model"))
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        model.save(save_location)

    # Print the losses in a table
    print(stage_format.format("Losses"))
    print(tabulate(losses_results, headers="keys"))
    print(tabulate(losses_results, headers="keys", tablefmt="latex"))

    # Reset environment variables
    reset_env_vars()

    print("Done")
