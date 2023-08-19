This is a refactored version of the experimental code used in the paper "Utilizing Semantic Textual Similarity for Clinical Survey Data Feature Selection." The data for this project is not publicly available because it involves patient data, but if another set of data with detailed feature names is available, our code can be easily adapted to that dataset.

# Structure & Usage
The full requirements are in `requirements.txt`. There are three directories: `redcap`, `mrmr_sts`, and `sts_scorers`. `redcap` contains the code for the feature selection and classification pioppeline. `mrmr_sts` contains the code for the top N, std. dev., and mRMR feature selection. `sts_scorers` contains the code for the STS scoring and training. 

# Setup
We set up our project around a root directory controlled by the argument `root_dir` when running `redcap/train.py`. It contains three directories `redcap`, `mrmr_sts`, and `sts_scorers`. The base directory should have the actual data and be named `REDCap Survey Data Annotated 2-6-2023.xlsx`. There should be another file in `redcap` called `redcap_features.xlsx` with one column listing features names and another named `impoprtant ` where features with values of `1` are selected. Alternatively modify line 555-564 in `redcap/train.py` to point to the correct data. 

This data will assume that some target questions exist among the data, if so, lines 129-132 in `redcap/train.py` should be modified to point to the correct target questions, and then lines 116-120 will need to be modified if we want to test a different labeling scheme.

Caching is used to speed up the process. `mi_sim.pk` is the cache for the mutual information scores,  `sts_sim.pk` is the cache for the STS scores for Bio_ClinicalBERT + ClinicalSTS. The remainder of the cached scoring files will be generated as needed depending on what scoring model/FT dataset combinations have been trained. These files will be created if they do not exist. 

# Training
The steps to train a model are as follows:

1. Download and combine the fine-tuning datasets by running `python data.py` while in `sts_scorers`.
2. Then run `python train.py` in `sts_scorers`. This will train the LM models listed from lines 67-80.
3. Change directories to `redcap` and then run `python train.py -r [base_dir]`. All of the feature selection scoring will be done before training and cached.
   - Add `-n -1` to run multithreaded on all cores, and `-v 1` or higher to get verbose output.
   - Use `-l` to get a list of all the possible scoring models.
   - Use `--sel_fs "fs1,fs2"` to select the feature selectors to use, and `--sel_class "[XGBoost, LinearSVM, ...]` to select the classifiers to train.


# Plotting Results
After training is done, it will print out a LaTeX table and should save as an xlsx. If it fails to save, the LaTeX output can be pasted into a `.txt` file and then converted using `tabtex.py`. Run `tabtex.py` to get summaries of the results and generate plots based on the individual configurations. Run `plot.py` to get the scoring heatmaps.