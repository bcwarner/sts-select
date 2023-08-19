# Copy


# Loads datasets from HuggingFace and caches them locally.
# Builds datasets not available on HuggingFace and also caches them locally.
# Also normalizes the scores to be between 0 and 1.

import datasets as ds
import numpy as np

CLINICAL_VOCAB_NAME = "clinical_vocab"
GEN_VOCAB_NAME = "gen_vocab"
COMBINED_VOCAB_NAME = "combined_vocab"


def load_huggingface(dset_name, sent1_key, sent2_key, score_key, score_max):
    """
    Loads a dataset from HuggingFace.
    :param dset_name:
    :return:
    """
    # Train set alone should be good
    if ";" in dset_name:
        dset_name, opt = dset_name.split(";")
        dset = ds.load_dataset(dset_name, opt, split="train")
    else:
        dset = ds.load_dataset(dset_name, split="train")
    mapper = lambda x: {
        "sent1": x[sent1_key],
        "sent2": x[sent2_key],
        "score": np.float32(x[score_key] / score_max),
    }
    dset = dset.map(mapper, remove_columns=dset.column_names)
    # Drop all columns except for sent1, sent2, and score
    return dset


if __name__ == "__main__":
    # Load all Hugging Face datasets
    huggingface_clin_datasets = {
        "bigbio/bio_simlex": ("text_1", "text_2", "score", 10),
        "bigbio/bio_sim_verb": ("text_1", "text_2", "label", 10),
        "bigbio/mayosrs": ("text_1", "text_2", "label", 10),
    }
    huggingface_gen_datasets = {
        "metaeval/sts-companion": ("sentence1", "sentence2", "label", 5),
        "stsb_multi_mt;en": ("sentence1", "sentence2", "similarity_score", 5),
    }

    huggingface_loaded_gen_data = {}
    huggingface_loaded_clin_data = {}

    for k, v in huggingface_clin_datasets.items():
        print(f"Loading {k}")
        huggingface_loaded_clin_data[k] = load_huggingface(k, *v)

    for k, v in huggingface_gen_datasets.items():
        print(f"Loading {k}")
        huggingface_loaded_gen_data[k] = load_huggingface(k, *v)

    # Load all custom datasets
    custom_clin_datasets = {}
    custom_gen_datasets = {}

    print("==== Concatenating Datasets ====")
    # Merge all datasets into three variants:
    # - General vocab
    print("General Vocab")
    gen_vocab = ds.concatenate_datasets(
        [v for k, v in huggingface_loaded_gen_data.items()]
        + [v for k, v in custom_gen_datasets.items()]
    )
    # - Clinical vocab
    print("Clinical Vocab")
    clin_vocab = ds.concatenate_datasets(
        [v for k, v in huggingface_loaded_clin_data.items()]
        + [v for k, v in custom_clin_datasets.items()]
    )
    # - General + clinical vocab
    print("General + Clinical Vocab")
    gen_clin_vocab = ds.concatenate_datasets([gen_vocab, clin_vocab])
    # Save
    gen_vocab.save_to_disk(GEN_VOCAB_NAME)
    clin_vocab.save_to_disk(CLINICAL_VOCAB_NAME)
    gen_clin_vocab.save_to_disk(COMBINED_VOCAB_NAME)
    print("Done")
