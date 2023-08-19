# Combine tablulated results from training into a summarized table.
import argparse
import os
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redcap_sts_scorers.train import dset_options, model_options
import tikzplotlib

if __name__ == "__main__":
    # Parse arguments
    # - xlsx files to include
    # - xlsx fname to be cached as for future use
    # - what kind of summarization we want to do
    # - what kind of column we should group by
    parser = argparse.ArgumentParser(prog="summarize", description="Summarize the results.")
    parser.add_argument(
        "-f",
        "--files",
        action="store",
        default=None,
        help="Files to include in the summary.",
        nargs="+",
    )
    parser.add_argument(
        "-c",
        "--cache",
        action="store",
        default=None,
        help="Cache the results to this file.",
    )
    args = parser.parse_args()

    # Set matplotlib font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    data = None
    # Load the data
    if len(args.files) == 1:
        if args.files[0].endswith(".txt"):
            # For when only a table is printed and no Excel file is saved.
            # This follows the LaTeX style since the grid style is too hard to parse.
            cols = None
            data = []
            sub_columns = ["feature_selector", "scorer", "model", "dset"]
            for line in open(args.files[0]):
                if "&" not in line:
                    continue
                line = line.replace("\\\\", "")
                if cols is None:
                    cols = [x.strip() for x in line.replace(r"\_", "_").split(" & ")]
                    cols = sub_columns + cols[1:]
                else:
                    row = [x.strip() for x in line.split(" & ")]
                    # Replace occurrences of the dset name and model name with a placholder
                    # since they may have multiple "\_" in them.
                    for idx, el in enumerate(model_options + dset_options):
                        row[0] = row[0].replace(el.replace("_", r"\_"), str(idx))
                    # Split the first column into sub_columns:
                    sub_vals = row[0].strip().split(r"\_")
                    for idx, el in enumerate(model_options + dset_options):
                        for i, x in enumerate(sub_vals):
                            if x == str(idx):
                                sub_vals[i] = el
                    # Append Nones for the missing sub_columns
                    sub_vals += [None] * (len(sub_columns) - len(sub_vals))
                    for i, x in enumerate(row):
                        try:
                            row[i] = float(x)
                        except ValueError:
                            pass
                    row = sub_vals + row[1:]
                    data.append(dict(zip(cols, row)))
            data = pd.DataFrame(data)
            # Save the data to an Excel file
            data.to_excel(args.files[0].replace(".txt", ".xlsx"))
        elif args.files[0].endswith(".xlsx"):
            data = pd.read_excel(args.files[0])
    else:
        data = []
        for fname in args.files:
            data.append(pd.read_excel(fname))

        data = pd.concat(data)

        if args.cache:
            data.to_excel(args.cache)

    # Summarization preprocessing:
    # Add a column indicating test-train difference for each score
    for col in data.columns:
        if col.endswith("_test"):
            data[col.replace("_test", "_diff")] = data[col] - data[col.replace("_test", "_train")]

    plots_dir = os.path.join(os.path.dirname(args.files[0]), "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    def nice_name_map(x, latex=False):
        map = {
            "feature_selector": "Feature Selector",
            "scorer": "Scorer",
            "model": "Scoring Model",
            "dset": "Fine-Tuning Dataset" if not latex else "FT Dataset",
            "roc_auc_score_test": "Test AUROC" if not latex else ("AUROC", "Test"),
            "roc_auc_score_train": "Train AUROC" if not latex else ("AUROC", "Train"),
            "roc_auc_score_diff": "Test - Train AUROC" if not latex else ("AUROC", r"$\Delta$"),
            "average_precision_score_test": "Test AUPRC" if not latex else ("AUPRC", "Test"),
            "average_precision_score_train": "Train AUPRC" if not latex else ("AUPRC", "Train"),
            "average_precision_score_diff": "Test - Train AUPRC" if not latex else ("AUPRC", r"$\Delta$"),
            "accuracy_score_test": "Test Accuracy" if not latex else ("Accuracy", "Test"),
            "accuracy_score_train": "Train Accuracy" if not latex else ("Accuracy", "Train"),
            "accuracy_score_diff": "Test - Train Accuracy" if not latex else ("Accuracy", r"$\Delta$"),
            "clinical_vocab": "Clinical Pairs",
            "gen_vocab": "General Pairs",
            "ext": "ClinicalSTS", # We combine this with the clinical vocab since it captures the same idea.
            "combined_vocab": "Combined Pairs",
            "classifier": "Classifier",
            "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "microsoft/PubMedBERT",
            "SelectFromModel-XGBoost": "SFM-XGBoost",
            "SelectFromModel-LinearSVM": "SFM-LinearSVM",
            "MRMRBase": "mRMR",
            "TopNSelector": "Top N" if not latex else "Top $N$",
            "StdDevSelector": "Std. Dev." if not latex else "Std. Dev.",
            "MIScorer": "MI" if not latex else "MI",
            "STSScorer": "STS" if not latex else "STS",
            "LinearScorer": "MI & STS" if not latex else r"MI \& STS",
        }
        if x in map and isinstance(map[x], tuple):
            return map[x]
        elif x in map and latex and isinstance(map[x], str):
            return ("Selection Hyperparameters", map[x])
        elif x in map:
            return map[x]
        return x

    # Summarizations:
    # - Group by feature_selector, scorer, model, and dset individually
    def group_by(hyperparam_col, score_col, score_col2=None):
        # Copy the data and rename the entries in the hyperparam_col to be nicer
        data_ = data.copy()
        data_[hyperparam_col] = data_[hyperparam_col].apply(nice_name_map)

        if score_col2 is not None:
            plt.subplot(1, 2, 1)
        # Plot a boxplot of the score_col grouped by hyperparam_col
        opts = data_[hyperparam_col].dropna().unique()
        opts = sorted(opts, key=lambda x: np.max(data_[data_[hyperparam_col] == x][score_col]))
        plt.gcf().set_size_inches(3.5, 2.75)
        for idx, opt in enumerate(opts):
            plt.violinplot(
                data_[data_[hyperparam_col] == opt][score_col],
                positions=[idx],
                vert=False,
            )
        plt.xlabel(nice_name_map(score_col))
        plt.ylabel(nice_name_map(hyperparam_col))
        # If there's a / in the opts, delete the part before the /
        opts = [x.split("/")[-1] if "/" in x else x for x in opts]
        plt.yticks(ticks=np.arange(len(opts)), labels=opts)

        if score_col2 is not None:
            plt.subplot(1, 2, 2)
            plt.gcf().set_size_inches(7, 2.75)
            for idx, opt in enumerate(opts):
                plt.violinplot(
                    data_[data_[hyperparam_col] == opt][score_col2],
                    positions=[idx],
                    vert=False,
                )
            plt.xlabel(nice_name_map(score_col2))
            plt.yticks(ticks=np.arange(len(opts)), labels=["" for _ in opts])

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{hyperparam_col}_{score_col}.pdf"))
        tikzplotlib.save(os.path.join(plots_dir, f"{hyperparam_col}_{score_col}.tex"))
        plt.clf()

        # Histogram of the score_col grouped by hyperparam_col


    group_by("feature_selector", "roc_auc_score_test", "roc_auc_score_diff")
    group_by("scorer", "roc_auc_score_test", "roc_auc_score_diff")
    group_by("model", "roc_auc_score_test")
    group_by("dset", "roc_auc_score_test")
    # Do a groupby for scorer + feature_selector

    # - Table with all baseline feature selection + best non-traditional feature selectors.
    def table_by_classifier(classifier, drop_cols=[]):
        # Filter out so the classifier is the only one in the frame
        df = data[data["classifier"] == classifier]
        if "Unnamed: 0" in data.columns:
            df = df.drop("Unnamed: 0", axis=1)
        # Results: All columns without a scorer.
        results: DataFrame = df[pd.isna(df["scorer"])]
        # + All MI scorers
        results = pd.concat([results, df[df["scorer"] == "MIScorer"]], axis=0)
        # + whichever ft-model is best
        m_ = df[df["scorer"] != "MIScorer"].groupby("model").max()["roc_auc_score_test"]
        best_model = m_.keys()[m_.argmax()]
        results = pd.concat([results, df[df["model"] == best_model]], axis=0)
        # Drop the model column
        results = results.drop(columns=["model"])

        # Drop the classifier column
        results = results.drop(columns=["classifier"])

        # Replace SelectFromModel with SFM
        results["feature_selector"] = results["feature_selector"].apply(lambda x: x.replace("SelectFromModel", "SFM") if not pd.isna(x) else x)

        # Reorder the _diff columns to be next to their _train counterparts
        cols = list(results.columns)
        for col in cols:
            if col.endswith("_diff"):
                cols.remove(col)
                cols.insert(cols.index(col.replace("_diff", "_train")) + 1, col)
        results = results[cols]

        # For the dset colum, delete "_vocab"
        results["dset"] = results["dset"].apply(lambda x: x.replace("_vocab", "") if not pd.isna(x) else x)

        # Iterate through the data and replace _ with \_
        results = results.apply(lambda x: x.str.replace("_", r"\_") if x.dtype == "object" else x)

        # Select the best result among MRMRBase, TopNSelector, and StdDevSelector where dset is not empty
        # Keep the other results as well
        best_results = OrderedDict()
        min_results = OrderedDict()
        main_results = {}
        ours = ["MRMRBase", "TopNSelector", "StdDevSelector"]
        fs_max = results[~pd.isna(results["dset"])].groupby("feature_selector").max()["roc_auc_score_test"]
        fs_min = results[~pd.isna(results["dset"])].groupby("feature_selector").apply(lambda x: abs(x["roc_auc_score_diff"]).min())
        for idx, row in results.iterrows():
            in_ours = row["feature_selector"] in ours
            if in_ours and not pd.isna(row["dset"]):
                if row["roc_auc_score_test"] == fs_max[row["feature_selector"]]:
                    # This is the best result
                    best_results[idx] = row
                # Is the abs value of the overfitting the smallest?
                if abs(row["roc_auc_score_diff"]) == fs_min[row["feature_selector"]]:
                    # This is the best result
                    min_results[idx] = row
            else:
                # Keep this result
                main_results[idx] = row

        # Combine the best and min results
        comb_results = {**best_results, **min_results}
        # Sort comb_results by the the auroc score
        comb_results = OrderedDict(x for x in sorted(comb_results.items(), key=lambda x: x[1]["roc_auc_score_test"], reverse=True))

        # Add a star if it was in best, dagger if in min.
        for key in comb_results.keys():
            if key in best_results.keys():
                comb_results[key]["feature_selector"] += r"$^*$"
            if key in min_results.keys():
                comb_results[key]["feature_selector"] += r"$^\dagger$"

        # Sort the main results by the auroc score
        main_results = OrderedDict(x for x in sorted(main_results.items(), key=lambda x: x[1]["roc_auc_score_test"], reverse=True))

        results_all = results.copy()
        results = pd.concat([DataFrame.from_dict(comb_results, orient="index"), DataFrame.from_dict(main_results, orient="index")])
        # Sort by whether the dset is empty or not
        results = results.sort_values(by="dset", ascending=False)

        # For each numerical column, bold the best value and round to 3 decimal places
        for col in results.columns:
            if is_numeric_dtype(results[col]):
                def test(x, col):
                    fmt = f"{x:.3f}"
                    if x == results_all[col].max():
                        return r"\textbf{" + fmt + "}"
                    elif "_diff" in col and abs(x) == abs(results[col]).min():
                        # There's a few perfectly random results, so we need to use results
                        return r"\textbf{" + fmt + "}"
                    else:
                        return fmt
                results[col] = results[col].apply(lambda x: test(x, col))
        # Replace the non-numeric columns with their nice names
        for col in results.columns:
            if not is_numeric_dtype(results[col]):
                def nice_name_ext(x):
                    if pd.isna(x):
                        return x
                    ds_index = x.find("$")
                    if ds_index != -1:
                        suffix = x[ds_index:]
                        x = x[:ds_index]
                    z = nice_name_map(x, latex=False)
                    if ds_index != -1:
                        z = z + suffix
                    if z is not None:
                        return z.replace("&", r"\&")
                results[col] = results[col].apply(nice_name_ext)

        # Drop the columns that are not needed

        # Convert to MultiIndex
        index = pd.MultiIndex.from_tuples(
            [nice_name_map(x, latex=True) for x in results.columns],
        )
        results.columns = index
        # Drop the accuracy multiindex, not enough space
        results = results.drop(columns=[nice_name_map("accuracy_score_test", latex=True),
                                        nice_name_map("accuracy_score_train", latex=True),
                                        nice_name_map("accuracy_score_diff", latex=True)])
        # Remove row index
        #results = results.set_index([[""] * results.shape[0]])
        out = results.to_latex(index=False,
                               escape=False,
                               bold_rows=False,
                               label=f"tab:{classifier}",
                               na_rep="-",
                               multicolumn=True,
                               multicolumn_format="|l|",
                               column_format="|" + "|".join(["l"] * results.shape[1]) + "|",
                               caption=f"Results for {classifier} with baseline feature selection methods "
                                        f"and using {best_model} to score features.",
                               )
        out = out.replace("{} &", "")

        # Print table
        print(f"======= {classifier} =======")
        print(out)
        print("==============================")


    table_by_classifier("MLP")
    # - Table with best performing overall vs. baselines

    def table_all():
        # Get all options for classifiers
        classifiers = data["classifier"].unique()

        def table_by_classifier(classifier):
            # Filter out so the classifier is the only one in the frame
            results = data[data["classifier"] == classifier]
            if "Unnamed: 0" in data.columns:
                results = results.drop("Unnamed: 0", axis=1)

            # Drop the classifier column
            results = results.drop(columns=["classifier"])

            # Replace SelectFromModel with SFM
            results["feature_selector"] = results["feature_selector"].apply(
                lambda x: x.replace("SelectFromModel", "SFM") if not pd.isna(x) else x)

            # Reorder the _diff columns to be next to their _train counterparts
            cols = list(results.columns)
            for col in cols:
                if col.endswith("_diff"):
                    cols.remove(col)
                    cols.insert(cols.index(col.replace("_diff", "_train")) + 1, col)
            results = results[cols]

            # For the dset colum, delete "_vocab"
            results["dset"] = results["dset"].apply(lambda x: x.replace("_vocab", "") if not pd.isna(x) else x)

            # Iterate through the data and replace _ with \_
            #results = results.apply(lambda x: x.str.replace("_", r"\_") if x.dtype == "object" else x)

            # For each numerical column, bold the best value and round to 3 decimal places
            for col in results.columns:
                if is_numeric_dtype(results[col]):
                    def test(x, col):
                        fmt = f"{x:.3f}"
                        # Test if the value is the best to 3 decimal places
                        if np.round(x, 3) == np.round(results[col].max(), 3):
                            return r"\textbf{" + fmt + "}"
                        elif "_diff" in col and abs(np.round(x, 3)) == abs(np.round(results[col], 3)).min():
                            # There's a few perfectly random results, so we need to use results
                            return r"\textbf{" + fmt + "}"
                        else:
                            return fmt

                    results[col] = results[col].apply(lambda x: test(x, col))
            # Replace the non-numeric columns with their nice names
            for col in results.columns:
                if not is_numeric_dtype(results[col]):
                    def nice_name_ext(x):
                        if pd.isna(x):
                            return x
                        ds_index = x.find("$")
                        if ds_index != -1:
                            suffix = x[ds_index:]
                            x = x[:ds_index]
                        z = nice_name_map(x, latex=False)
                        if ds_index != -1:
                            z = z + suffix
                        if z is not None:
                            z = z.replace("_", r"\_")
                            z = z.replace("&", r"\&")
                            if "/" in z:
                                z = z.split("/")[1]
                            return z

                    results[col] = results[col].apply(nice_name_ext)

            # Sort
            results = results.sort_values(by=["feature_selector", "scorer", "model", "dset"])

            #  # Convert to MultiIndex
            index = pd.MultiIndex.from_tuples(
                [nice_name_map(x, latex=True) for x in results.columns],
            )
            results.columns = index

            # Drop the accuracy multiindex, not enough space
            results = results.drop(columns=[nice_name_map("accuracy_score_test", latex=True),
                                            nice_name_map("accuracy_score_train", latex=True),
                                            nice_name_map("accuracy_score_diff", latex=True)])

            out = results.to_latex(index=False,
                                   escape=False,
                                   bold_rows=False,
                                   label=f"tab:{classifier}",
                                   na_rep="-",
                                   multicolumn=True,
                                   multicolumn_format="|l|",
                                   column_format="|" + "|".join(["l"] * results.shape[1]) + "|",
                                   caption=f"All results for {classifier}.",
                                   longtable=True,
                                   )
            out = out.replace("{} &", "")
            out = out.replace(r"\\", r"\\ \hline")
            out = out.replace(r"\toprule", r"\hline").replace(r"\midrule", r"\hline").replace(r"\bottomrule", r"\hline")
            out = out.replace(r"\begin{table}", r"\begin{table}[H]")
            print(out)


        for classifier in classifiers:
            print(r"\subsection{" + classifier + "}")
            table_by_classifier(classifier)

    table_all()
