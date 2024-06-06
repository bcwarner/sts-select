# Scores the similarity of all entries in the All of Us Codebook
import re

import sts_select
import pandas as pd
import os
import rootutils
import hydra
import numpy as np
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf

from sts_select.scoring import SentenceTransformerScorer


def get_codebook_columns(config: DictConfig) -> list:
    # Load the DF
    codebook = pd.read_excel(os.path.join(os.path.dirname(__file__), config["data"]["codebook"]), sheet_name=None)
    columns = []
    for page_name, page in codebook.items():
        if any([ep in page_name for ep in config["data"]["exclude_pages"]]):
            continue

        print(f"Unique field types: {page[config['data']['field_type']].unique()}")

        for _, field in page.iterrows():
            # Skip fields that are "text" without any validation
            # Skip descriptive fields 
            if field[config["data"]["field_type"]] == "descriptive" or field[config["data"]["field_type"]] == "text":
                continue
            choices = None
            if isinstance(field[config["data"]["choices"]], str) and "|" in field[config["data"]["choices"]]:
                if field[config["data"]["field_type"]] == "slider":
                    choices = None # Just a number
                elif field[config["data"]["field_type"]] == "checkbox" or field[config["data"]["field_type"]] == "radio" or field[config["data"]["field_type"]] == "dropdown":
                    choices = [x.split("_")[-1] for x in field[config["data"]["choices"]].split("|")]
                    if len(choices) == 2:
                        choices = None # Binary
                    elif "old" in field[config["data"]["field_label"]].lower() or "new" in field[config["data"]["field_label"]].lower():
                        choices = None # Age groups, ordinal
                    elif len(choices) > config["data"]["max_categories"]:
                        continue
                else:
                    choices = None
            f_name = field[config["data"]["field_label"]]
            if choices is not None:
                columns.extend([f"{f_name}_{choice}" for choice in choices])
            else:
                columns.append(f_name)
    # Filter out nans
    columns = [c for c in columns if isinstance(c, str)]
    return columns

@hydra.main(config_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf"),
            config_name="config",
            version_base="1.1")
def main(config: DictConfig) -> None:
    columns = get_codebook_columns(config)
    breakpoint()

    # Get all of the models in the fine-tuned directory
    models_path = config["path"]["sts_models_path"]
    # Search recursively for all models (i.e. .bins)
    models = []
    for dirname, _, files in os.walk(models_path):
        for file in files:
            if file.endswith(".bin"):
                models.append(dirname)
    
    y_names = config["data"]["y_labels"]
    for idx in range(config["start_model"], len(models), config["step_model"]):
        model_path = models[idx]
        print(f"Scoring model {model_path}")
        model_name = "_".join(model_path.split("/")[-3:])
        cache_path = os.path.join(os.path.dirname(__file__), "cache", model_name + ".pkl")
        print(f"Cache path: {cache_path}")

        if os.path.exists(cache_path):
            continue

        scorer = SentenceTransformerScorer(
            np.zeros((len(columns), len(columns))),
            np.zeros(len(y_names)),
            X_names=columns,
            y_names=y_names,
            model_path=model_path,
            cache=cache_path,
            verbose=1,
        )
        scorer.save_cache()
    # Scoring should go by itself.

if __name__ == "__main__":
    main()
