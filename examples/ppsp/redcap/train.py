import argparse
import itertools
import os
import pickle
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from operator import itemgetter

import numpy as np
import pandas as pd
import scipy
import torch
from joblib import Parallel, delayed
from pandas.api.types import (is_datetime64_any_dtype, is_numeric_dtype,
                              is_string_dtype)
from pandas.errors import ParserError
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   Normalizer, OneHotEncoder)
from sklearn.svm import LinearSVC
from tabulate import tabulate
from xgboost import XGBClassifier

# Add the current working directory's parent to the path above to import the neighboring packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from redcap_sts_scorers.train import (GEN_VOCAB_NAME, dset_options,
                                      model_options, model_path)
from skrebate import ReliefF

from sts_select.gensim import FastTextModel, SkipgramModel
from sts_select.mrmr import MRMRBase
from sts_select.scoring import (BaseSTSScorer, GensimScorer, LinearScorer,
                                MIScorer, SentenceTransformerScorer)
from sts_select.target_sel import StdDevSelector, TopNSelector

pd.set_option("display.max_columns", None)
redcap = None
redcap_features = None

TARGET_LABEL = "persistent_pain"
THREE_MO_COMPLETION = 282
SIX_MO_COMPLETION = 344
RANDOM_STATE = 278797835

TARGET_QUESTION_1 = "In the past week, did you have any pain in your surgical incision or in the area related to your surgery? "
TARGET_QUESTION_2 = "For pain in the area related to your surgery, did the pain start or worsen after the surgery?"
TARGET_QUESTION_3 = "On a scale of zero to ten, with zero being no pain and ten being the worst pain, please fill in your average pain level during the past week, while you were at rest."
TARGET_QUESTION_4 = "On a scale of zero to ten, with zero being no pain and ten being the worst pain, please fill in your average pain level during the past week, when you were active or moving."

STS_CACHE_FNAME = "sem_sim.pk"
MI_CACHE_FNAME = "mi_sim.pk"
FIGURE_DIR = "figures"
RESULTS_DIR = "results"

args = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


set_seed()

# Get survey data.
if not os.path.exists(os.path.join(os.path.curdir, FIGURE_DIR)):
    os.makedirs(os.path.join(os.path.curdir, FIGURE_DIR))

if not os.path.exists(os.path.join(os.path.curdir, RESULTS_DIR)):
    os.makedirs(os.path.join(os.path.curdir, RESULTS_DIR))


def dump_result(data, param_dict, fname):
    with open(os.path.join(os.path.curdir, RESULTS_DIR, fname), "wb") as of:
        pickle.dump((data, param_dict), of)


def select_outcome(df, completion):
    mask = ~df.iloc[
        :, completion
    ].isnull()  # status.loc[:,'Complete?.10'] == 'Complete'
    id_list = df[mask]["Study ID:"]
    return df.loc[redcap["Study ID:"].isin(id_list)]


def get_result(x):
    bool1 = x[TARGET_QUESTION_1] == "Yes" and x[TARGET_QUESTION_2] == "Yes"
    bool2 = x[TARGET_QUESTION_3] >= 3 or x[TARGET_QUESTION_4] >= 3

    return int(bool1 and bool2)


def extract_outcome(redcap: pd.DataFrame):
    outcome = redcap.copy()
    outcome["STUDY_ALIAS"] = redcap["Study ID:"]
    outcome[TARGET_LABEL] = redcap.apply(lambda x: get_result(x), axis=1)
    outcome = outcome.drop(
        columns=[
            TARGET_QUESTION_1,
            TARGET_QUESTION_2,
            TARGET_QUESTION_3,
            TARGET_QUESTION_4,
        ]
    )
    # Merge all rows with the same STUDY_ALIAS, picking the last value by Repeat Instance, except when values are NaN
    outcome = outcome.groupby("STUDY_ALIAS").last()
    outcome = outcome.reset_index()

    out_table = {
        "Individuals with Complete Mark": outcome.STUDY_ALIAS.nunique(),
        "PPSP (+)": outcome[outcome.persistent_pain == 1].STUDY_ALIAS.nunique(),
        "PPSP (-)": outcome[outcome.persistent_pain == 0].STUDY_ALIAS.nunique(),
    }

    race_key_format = "Please specify your race: (choice={})"
    race_keys = [
        "Caucasian",
        "American Indian / Alaskan Native",
        "Asian",
        "Black / African Heritage",
        "Hawaiian Native / Other Pacific Islander",
        "Other",
        "Prefer not to answer",
    ]
    for race in race_keys:
        out_table[race] = outcome[
            outcome[race_key_format.format(race)] == "Checked"
        ].STUDY_ALIAS.nunique()
    # Do this for sex
    sex_key = "What is your sex (assigned at birth):"
    for sex in outcome[sex_key].unique():
        out_table[sex] = outcome[outcome[sex_key] == sex].STUDY_ALIAS.nunique()
    # Check for NAs
    out_table["(No Answer)"] = outcome[pd.isna(outcome[sex_key])].STUDY_ALIAS.nunique()

    # Do this for age
    age_key = "Age:"
    out_table["Age (min)"] = outcome[age_key].min()
    out_table["Age (mean)"] = outcome[age_key].mean()
    out_table["Age (std. dev.)"] = outcome[age_key].std()
    out_table["Age (max)"] = outcome[age_key].max()

    # Map keys and vals to separate cols
    tab_dict = {"Name": out_table.keys(), "Value": out_table.values()}

    # Use tabular to print out a table in the style specified with argpase.

    print(tabulate(tab_dict, headers="keys", tablefmt=args.table_fmt))
    # Now do it with a LaTeX table.
    print(tabulate(tab_dict, headers="keys", tablefmt="latex"))
    return outcome


# ==== TEMPORARY ====
# Select only the things that are class labels or continuous.
def filter_data(df: pd.DataFrame, max_categories: int = 5):
    # Get list of acceptable features.

    acceptable_features = set()

    if args.feature_filter:
        for idx, data in redcap_features.iterrows():
            if data["important "] == 1:
                acceptable_features.add(data["Name"])

    # Attempt to categorize columns as one of: continuous, binary, one-hot, time, external data (pngs)
    image_ext_filter = re.compile("(jpg|jpeg|tiff|png|gif|svg)", re.IGNORECASE)

    kept_cols = []
    transformers = defaultdict(list)  # Map transformers => columns for later.

    # Try and repair some of the bad columns.
    bad_dtypes = df.select_dtypes(include=["O"])

    bad_res = defaultdict(list)  # For debugging inspection.
    for col_name in bad_dtypes:
        col = df[col_name]

        # Try numeric casting.
        try:
            df[col_name] = pd.to_numeric(col)
            bad_res["numeric"].append(col_name)
            continue
        except (ParserError, ValueError, TypeError):
            pass

        # Try string casting.
        try:
            df[col_name] = col.astype(pd.StringDtype())
            bad_res["str"].append(col_name)
            continue
        except (ParserError, ValueError, TypeError):
            pass

        # NA out for deletion later.
        df.loc[:, col_name] = pd.NA
        bad_res["fail"].append(col_name)

    for col_name in df:
        col = df[col_name]
        col_dtype = col.dtype

        if col_name == TARGET_LABEL:
            kept_cols.append(col)
            continue

        if args.feature_filter:
            if col_name not in acceptable_features:
                continue

        if all(pd.isna(col)):
            continue

        if is_string_dtype(col_dtype):
            # Contain extensions? Sample until we find something that isn't null.
            keep = False
            for v in col:
                if pd.isna(v):  # Skip NAs
                    continue

                if not isinstance(v, str):
                    breakpoint()
                keep = image_ext_filter.match(v) is None
                break

            if not keep:  # Skip this column.
                continue

            # Are there a small enough number of categories?
            # Yes => Convert to categorical
            # No => Drop for now.

            if len(col.unique()) <= max_categories:
                col_dum = pd.get_dummies(col, prefix=col_name)
                for col_sub in col_dum:
                    kept_cols.append(col_dum[col_sub])
                    transformers[col_sub].append("SimpleImputerS")

        elif is_datetime64_any_dtype(col_dtype):
            # Convert to numerical
            kept_cols.append(col)
            transformers[col_name].append(MinMaxScaler)
            transformers[col_name].append("SimpleImputerT")

        elif is_numeric_dtype(col_dtype):
            kept_cols.append(col)
            transformers[col_name].append(Normalizer)
            transformers[col_name].append(SimpleImputer)

    # Get rid of images, text with large data.
    df_filtered = pd.DataFrame(kept_cols)  # DO NOT TRANSPOSE
    # Make categorical text-columns one-hot.

    # Normalize continuous numerical data.
    if args.print_del_cols:
        print("--- Deleted columns ---")
        print(
            "\n".join(
                set([col_name for col_name in df]) - set([x.name for x in kept_cols])
            )
        )
        print("-----------------------")

    print(f"Total features (including one-hot): {len(kept_cols)}")

    return df_filtered, transformers


def build_preprocessor(transformer_dict):
    trans_instantiators = {
        "IdentityTransformer": lambda: FunctionTransformer(),  # Defaults to the identity
        SimpleImputer: lambda: SimpleImputer(missing_values=np.nan, strategy="mean"),
        "SimpleImputerT": lambda: SimpleImputer(
            missing_values=pd.NaT, strategy="median"
        ),
        "SimpleImputerS": lambda: SimpleImputer(
            missing_values=np.nan, strategy="most_frequent"
        ),
        Normalizer: lambda: Normalizer(),
        OneHotEncoder: lambda: OneHotEncoder(),
        MinMaxScaler: lambda: MinMaxScaler(),
    }
    sort_index = list(trans_instantiators.keys()).index

    pipelines_cols = defaultdict(list)
    # Steps:
    # Map frozensets of pipeline stages => columns.
    # Convert frozensets to pipelines, map to columns.
    for target, tfs in transformer_dict.items():
        pipelines_cols[frozenset(tfs)].append(target)

    trans = []
    for pipe_items, pipe_targets in pipelines_cols.items():
        # Make pipeline
        pipe_items = sorted(pipe_items, key=sort_index)
        pipe = make_pipeline(*[trans_instantiators[p]() for p in pipe_items])
        name = "-".join([t if type(t) == str else t.__name__ for t in pipe_items])
        trans.append((name, pipe, pipe_targets))

    return ColumnTransformer(trans, remainder="passthrough", verbose=False)


def build_pipeline(**kwargs):
    # Build the preprocessor pipeline.
    pipe = Pipeline([(k, v) for k, v in kwargs.items()])

    return pipe


def train_eval(df_filtered, pipe, X_questions=None, test_size=0.2, parameters={}):
    X = df_filtered.drop(TARGET_LABEL, axis=0).T
    y = df_filtered.loc[TARGET_LABEL]

    # Split
    k_folds = args.k_folds if args.k_folds != -1 else 5
    inner_stratcv = StratifiedKFold(
        n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE
    )
    outer_stratcv = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=RANDOM_STATE
    )

    scores = ["accuracy", "roc_auc", "average_precision"]
    if type(pipe[1]) != ReliefF:
        cv = GridSearchCV(
            pipe,
            parameters,
            scoring=scores,
            refit="roc_auc",
            verbose=args.verbose,
            cv=inner_stratcv,
            n_jobs=args.n_jobs if type(pipe[-1]) != XGBClassifier else 1,
        )  # 5-fold CV
        # Train
        warnings.filterwarnings("ignore")
    else:
        cv = pipe

    if args.nested:
        score_res = cross_validate(
            cv,
            X,
            y,
            scoring=scores,
            cv=outer_stratcv,
            n_jobs=1,
            verbose=args.verbose,
            return_estimator=True,
        )
        # Eval
        out = {k: np.mean([x for x in score_res[f"test_{k}"]]) for k in scores}

        # Just pick an estimator for now:
        cv = score_res["estimator"][0]
        cv_est = cv  # Untested and not used
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        cv.fit(X_train, y_train)

        out = {}
        cv_est = cv.best_estimator_ if type(cv) == GridSearchCV else cv
        for score in [roc_auc_score, average_precision_score, accuracy_score]:
            prob_decision = (
                cv_est.decision_function
                if hasattr(cv_est, "decision_function")
                else (lambda z: cv_est.predict_proba(z)[:, 1])
            )
            if score == roc_auc_score or score == average_precision_score:
                out[f"{score.__name__}_test"] = score(y_test, prob_decision(X_test))
                out[f"{score.__name__}_train"] = score(y_train, prob_decision(X_train))
            else:
                out[f"{score.__name__}_test"] = score(y_test, cv_est.predict(X_test))
                out[f"{score.__name__}_train"] = score(y_train, cv_est.predict(X_train))

    best_feats = None
    if isinstance(cv_est.steps[1][1], MRMRBase):
        best_feats = [X_questions[y] for y in cv_est.steps[1][1].sel_features]

    feat_importance = None
    if args.feature_importance:
        import shap

        print("Getting SHAP values")

        def f(x_):
            return cv.predict_proba(pd.DataFrame(x_, columns=X.columns))

        explainer = shap.KernelExplainer(f, X_train, seed=RANDOM_STATE)
        shap_values = explainer(X_test)
        shap_abs_values = np.mean(np.abs(shap_values.values), axis=0)
        # Test on entire set since it's small.

        # Get mapping from questions to order picked.
        if type(cv_est.named_steps["feature_selector"]) in [
            MRMRBase,
            StdDevSelector,
            TopNSelector,
        ]:
            fs = cv_est.named_steps["feature_selector"]
            # Map X_question_idx to occurrence in sel_features
            X_question_order, X_question_sel = zip(
                *enumerate([X_questions[x] for x in fs.sel_features], start=1)
            )
            shap_fil_values = [shap_abs_values[x] for x in fs.sel_features]
            feat_importance = {
                "Feature": X_question_sel,
                "Order": X_question_order,
            }
            if type(fs) == MRMRBase:
                feat_importance["mRMR Value"] = fs.mrmr
                feat_importance["Relevancy"] = fs.relevancies
                feat_importance["Redundancy"] = fs.redundancies
            else:
                feat_importance["Score"] = fs.scores
            feat_importance["SHAP Value"] = shap_fil_values
        else:
            q_sort, s1, s2, s3 = zip(
                *sorted(
                    zip(X_questions, shap_abs_values),
                    key=itemgetter(1),
                )
            )

            feat_importance = {
                "Feature": q_sort,
                f"{scores[0]} Mean": s1,
                f"{scores[1]} Mean": s2,
                f"{scores[2]} Mean": s3,
            }

    return (
        out,
        (cv.best_params_ if type(cv_est) == GridSearchCV else {}),
        best_feats,
        feat_importance,
    )


def combine_elements(preprocessor, feature_selector, classifier):
    pipe = build_pipeline(
        preprocessor=preprocessor,
        feature_selector=feature_selector[0],
        classifier=classifier[0],
    )

    parameters = {**feature_selector[1], **classifier[1]}

    return pipe, parameters


class ClinicalBERTSTSScorer(BaseSTSScorer):
    def __init__(self, X, y, X_names, y_names, cache=None, verbose=0):
        super().__init__(
            X,
            y,
            X_names,
            y_names,
            cache=os.path.join(args.root_dir, STS_CACHE_FNAME),
            sts_function=lambda a, b: None,
            verbose=args.verbose,
        )
        # Empty function here because BERT will be loaded only if needed.

    def score(self, X, y):
        if (
            self.sts_function("", "") is None
        ):  # Hack to determine if it is an uninitalized fn.
            from semantic_text_similarity.models import ClinicalBertSimilarity

            self.bert = ClinicalBertSimilarity(device="cuda", batch_size=10)
            self.sts_function = lambda a, b: self.bert.predict(
                [(a, b)], return_predictions=True
            )[0]
        return super().score(X, y)


def compute_similarities(
    X_questions, y_questions, X, y_truth, preprocessor, clear_cache=False
):
    # Generate all possible scorers for X and X_y
    # - MI Scorer
    # - STS Scorer
    # - LinearScorer from both
    X = preprocessor.fit_transform(X)
    scorers = {}

    scorers["MIScorer"] = MIScorer(
        X,
        y_truth,
        random_state=RANDOM_STATE,
        cache=os.path.join(args.root_dir, MI_CACHE_FNAME),
        verbose=args.verbose,
    )

    scorers["STSScorer_ClinicalBERT_ext"] = ClinicalBERTSTSScorer(
        X,
        y_truth,
        X_questions,
        y_questions,
        cache=os.path.join(args.root_dir, STS_CACHE_FNAME),
        verbose=args.verbose,
    )

    scorers["LinearScorer_ClinicalBERT_ext"] = LinearScorer(
        X,
        y_truth,
        scorers=[scorers["MIScorer"], scorers["STSScorer_ClinicalBERT_ext"]],
        alpha=[1, 1],
        verbose=args.verbose,
    )

    # Product of models and datasets for each possible BaseSTS and LinearScorer.
    for dset, model in itertools.product(dset_options, model_options):
        print(f"Loading {model} for {dset}")
        sts_name = f"STSScorer_{model}_{dset}"
        sts_cache_name = f"STS_{model}_{dset}.pkl".replace("/", "-")
        if "other" in model:
            if dset != GEN_VOCAB_NAME:
                continue
            sts_scorer = GensimScorer(
                X,
                y_truth,
                X_questions,
                y_questions,
                model_path=model_path(dset, model),
                cache=os.path.join(args.root_dir, sts_cache_name),
                model_type=SkipgramModel if "Skipgram" in model else FastTextModel,
                # Note: this will break if we add more models.
                verbose=args.verbose,
            )
        else:
            sts_scorer = SentenceTransformerScorer(
                X,
                y_truth,
                X_questions,
                y_questions,
                model_path=model_path(dset, model),
                device="cuda",
                cache=os.path.join(args.root_dir, sts_cache_name),
                verbose=args.verbose,
            )

        scorers[sts_name] = sts_scorer
        linear_name = f"LinearScorer_{model}_{dset}"
        linear_scorer = LinearScorer(
            X,
            y_truth,
            scorers=[scorers["MIScorer"], sts_scorer],
            alpha=[1, 1],
            verbose=args.verbose,
        )
        scorers[linear_name] = linear_scorer

    # Add the

    return scorers


def prepare_source_data():
    global redcap, redcap_features
    source_options = [
        r"REDCAP_FUL_REPORT_AUG_3_2022.csv",
        r"REDCap Survey Data Annotated 2-6-2023.xlsx",
    ]
    redcap = pd.read_excel(
        os.path.join(os.path.dirname(args.root_dir), source_options[1]),
        # on_bad_lines="warn",
        # encoding="latin1",
    )
    redcap_features = pd.read_excel(
        os.path.join(args.root_dir, "redcap", r"redcap_features.xlsx")
    )
    # Important preprocessing filtration.
    df_filtered = select_outcome(redcap, SIX_MO_COMPLETION)
    df_filtered = extract_outcome(df_filtered)
    df_filtered, transformers = filter_data(df_filtered)
    preprocessor = build_preprocessor(transformers)
    # Combine pipelines and build
    y_questions = [
        TARGET_LABEL,
        TARGET_QUESTION_1,
        TARGET_QUESTION_2,
        TARGET_QUESTION_3,
        TARGET_QUESTION_4,
    ]
    X_questions = df_filtered.drop(TARGET_LABEL, axis=0).axes[0].tolist()

    X = df_filtered.drop(TARGET_LABEL, axis=0).T
    y_truth = df_filtered.loc[TARGET_LABEL]
    scorers = compute_similarities(
        X_questions, y_questions, X, y_truth, preprocessor, clear_cache=args.clear_cache
    )

    return (
        df_filtered,
        preprocessor,
        scorers,
        X_questions,
        y_questions,
    )


# ======= Note =========
# The following functions were designed for an earlier version of the code, and do not work with the current version.
# They also evaluate some experiments that we've left out for brevity.


def eval_auroc_hyperparms(
    df_filtered=None,
    pipe=None,
    y_questions=None,
    X_questions=None,
    MI_X_similarities=None,
    MI_X_y_similarities=None,
    STS_X_similarities=None,
    STS_X_y_similarities=None,
    parameters=None,
):
    """
    Generate a grid of the AUROC and AUPRC as a function of feature and alpha count.
    :return:
    """

    feature_selector = pipe.named_steps["feature_selector"]
    classifier = pipe.named_steps["classifier"]
    model_name = type(classifier).__name__
    if not isinstance(feature_selector, MRMRBase):
        print("=== Needs an mRMR pipeline! ====")
        sys.exit(0)

    # Anymore than quarter is probably overkill.
    N_rng = [x for x in range(0, len(X_questions) // 4)]
    a_rng = [1] if type(feature_selector) != MRMRCombined else np.linspace(0, 2, 50)

    auroc_res_test = {}
    auroc_res_train = {}
    auprc_res_test = {}
    auprc_res_train = {}

    def eval_params(N, alpha):
        params = parameters if parameters is not None else {}
        params["feature_selector__n_features"] = [N]
        if type(feature_selector) == MRMRCombined:
            params["feature_selector__alpha"] = [alpha]
        te_results, _, _, _ = train_eval(
            df_filtered, pipe, X_questions=X_questions, parameters=params
        )
        return te_results

    space = [x for x in itertools.product(N_rng, a_rng)]
    res = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
        delayed(eval_params)(N, alpha) for N, alpha in space
    )

    for pms, r in zip(space, res):
        auroc_res_test[pms] = r["roc_auc_score_test"]
        auroc_res_train[pms] = r["roc_auc_score_train"]
        auprc_res_test[pms] = r["average_precision_score_test"]
        auprc_res_train[pms] = r["average_precision_score_train"]

    # One param => curve
    if type(feature_selector) != MRMRCombined:
        auroc_line = [
            (auroc_res_test[N, a_rng[0]], auroc_res_train[N, a_rng[0]]) for N in N_rng
        ]
        auprc_line = [
            (auprc_res_test[N, a_rng[0]], auprc_res_train[N, a_rng[0]]) for N in N_rng
        ]

        params = {
            "x": N_rng,
            "x_name": "N",
            "type": "au_n_line",
            "model_name": model_name,
            "feature_selector": type(feature_selector).__name__,
        }

        dump_result(
            auroc_line,
            {**params, **{"au_name": "AUROC"}},
            f"auroc-{model_name}-{type(feature_selector).__name__}.dat",
        )
        dump_result(
            auprc_line,
            {**params, **{"au_name": "AUPRC"}},
            f"auprc-{model_name}-{type(feature_selector).__name__}.dat",
        )
    else:
        auroc_res_test_np = np.array(
            [auroc_res_test[N, a] for N, a in itertools.product(N_rng, a_rng)]
        ).reshape((len(N_rng), len(a_rng)))
        auroc_res_train_np = np.array(
            [auroc_res_train[N, a] for N, a in itertools.product(N_rng, a_rng)]
        ).reshape((len(N_rng), len(a_rng)))
        auroc_grid = (auroc_res_test_np, auroc_res_train_np)

        auprc_res_test_np = np.array(
            [auprc_res_test[N, a] for N, a in itertools.product(N_rng, a_rng)]
        ).reshape((len(N_rng), len(a_rng)))
        auprc_res_train_np = np.array(
            [auprc_res_train[N, a] for N, a in itertools.product(N_rng, a_rng)]
        ).reshape((len(N_rng), len(a_rng)))
        auprc_grid = (auprc_res_test_np, auprc_res_train_np)

        params = {
            "x": a_rng,
            "y": N_rng,
            "x_name": "\\alpha",
            "y_name": "N",
            "type": "au_map",
            "model_name": model_name,
            "feature_selector": type(feature_selector).__name__,
        }

        dump_result(
            auroc_grid,
            {**params, **{"au_name": "AUROC"}},
            f"auroc-{model_name}-{type(feature_selector).__name__}.dat",
        )
        dump_result(
            auprc_grid,
            {**params, **{"au_name": "AUPRC"}},
            f"auprc-{model_name}-{type(feature_selector).__name__}.dat",
        )

    # Two params => Heatmap and curve

    print("Done!")
    if type(feature_selector) == MRMRCombined:
        sys.exit(0)


def eval_mi_embeddings_correlation(
    df_filtered=None,
    pipe=None,
    y_questions=None,
    X_questions=None,
    X_similarities=None,
    X_y_similarities=None,
    parameters=None,
):
    """
    Evaluate the correlation between mutual information and cosine similarity scores.
    :return:
    """
    preprocessor = pipe.named_steps["preprocessor"]
    X = df_filtered.drop(TARGET_LABEL, axis=0).T
    y_truth = df_filtered.loc[TARGET_LABEL]
    X = preprocessor.fit_transform(X, y_truth)

    feature_selector = pipe.named_steps["feature_selector"]
    if not isinstance(feature_selector, MRMRBase):
        print("=== Needs an mRMR pipeline! ====")
        sys.exit(0)
    feature_selector.fit(X, y_truth)

    raise NotImplementedError("TODO: Implement this")

    # Get the STS scores.

    STS_X_pairings, STS_X_y_pairings = X_similarities, X_y_similarities

    def color_val(k1, k2, sel_features):
        if k1 in sel_features:
            return sel_features.index(k1)
        elif k2 in sel_features:
            return sel_features.index(k2)
        return len(sel_features)  # Not inclusive, should work

    # Zip the X-X and X-y pairings together.
    MI_STS_X_pairings = [
        [
            MI_X_pairings[k],
            STS_X_pairings[k],
            color_val(k[0], k[1], feature_selector.sel_features),
        ]
        for k in MI_X_pairings.keys()
    ]
    MI_STS_X_y_pairings = [
        [
            MI_X_y_pairings[k],
            STS_X_y_pairings[k],
            color_val(k[0], None, feature_selector.sel_features),
        ]
        for k in MI_X_y_pairings.keys()
    ]

    # Plot each pairing.
    def scatt_pair(pair, pair_name, pair_x_name, pair_y_name):
        pair = sorted(pair, key=itemgetter(2))
        x, y, colors = zip(*pair)
        # breakpoint()
        x = np.array(x).reshape(-1)
        y = np.array(y).reshape(-1)
        colors = list(colors)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        x_rng = np.array([min(x), max(x)])
        y_tr = slope * x_rng + intercept

        plt.clf()
        # Filter
        # breakpoint()
        m_fts = colors.index(len(feature_selector.sel_features))
        plt.scatter(x[m_fts:], y[m_fts:], c="k")
        plt.scatter(x[:m_fts], y[:m_fts], c=colors[:m_fts], cmap="cool")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("Order Selected", rotation=270)
        # plt.plot(x_rng, y_tr, label=f"Trend (r^2 = {r_value**2})", color="b")

        # plt.legend()
        plt.title(f"{pair_y_name} vs. {pair_x_name}")
        plt.xlabel(f"{pair_x_name}")
        plt.ylabel(f"{pair_y_name}")
        plt.savefig(
            os.path.join(
                args.root_dir,
                FIGURE_DIR,
                f"{type(feature_selector).__name__}-{pair_name}.png",
            ),
            dpi=300,
        )

    scatt_pair(
        MI_STS_X_pairings,
        "Feature-Feature",
        "Mutual Information",
        "Semantic Textual Similarity",
    )
    scatt_pair(
        MI_STS_X_y_pairings,
        "Feature-Target",
        "Mutual Information",
        "Semantic Textual Similarity",
    )
    # Save.
    print("Done!")
    if type(feature_selector) == MRMRCombined:
        sys.exit(0)


def eval_base_train_time(
    df_filtered=None,
    preprocessor=None,
    y_questions=None,
    X_questions=None,
    X_similarities=None,
    X_y_similarities=None,
):
    """
    Evaluate the training time of MRMRBase
    :return:
    """
    pass


# =============================

if __name__ == "__main__":
    # Process arguments.
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=set_seed)

    parser = argparse.ArgumentParser(
        prog="train",
        description="Evaluator for different feature selection schemes on REDCap data.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store",
        default=0,
        help="Verbosity number for CV (0-3)",
        type=int,
    )
    parser.add_argument(
        "-k",
        "--k-folds",
        default=5,
        action="store",
        help="Number of folds for CV, default 5",
        type=int,
    )
    parser.add_argument(
        "-f",
        "--feature_importance",
        default=False,
        action="store",
        help="Display feature importance",
        type=bool,
    )
    parser.add_argument(
        "-b",
        "--best-params",
        default=True,
        action="store",
        help="Display best parameters",
        type=bool,
    )
    parser.add_argument(
        "--best-feats",
        default=True,
        action="store",
        help="Display best features",
        type=bool,
    )
    parser.add_argument(
        "-p",
        "--feature-filter",
        default=True,
        action="store",
        help="Filter out columns from a given list",
        type=bool,
    )
    parser.add_argument(
        "-t",
        "--table-fmt",
        default="grid",
        choices=["grid", "latex"],
        action="store",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--clear-cache",
        default=False,
        action="store",
        help="Clear the similarity score cache.",
        type=bool,
    )
    parser.add_argument(
        "--nested",
        default=False,
        action="store",
        help="Use nested cross-validation.",
        type=bool,
    )
    parser.add_argument(
        "--list-scorers",
        "-l",
        default=False,
        action="store_true",
        help="List available scorers.",
    )
    parser.add_argument(
        "--feature_num",
        default=20,
        action="store",
        help="Number of features to select.",
        type=int,
    )
    parser.add_argument("-n", "--n-jobs", default=1, action="store", type=int)
    # Default root-dir should correspond to the root of the git repo (i.e. contains feature_selectors, redcap, sts_scorers)
    parser.add_argument(
        "-r",
        "--root-dir",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        action="store",
        type=str,
    )
    parser.add_argument("--print_del_cols", default=False, action="store", type=bool)
    parser.add_argument("--sel_fs", default=None, action="store", type=str)
    parser.add_argument("--sel_class", default=None, action="store", type=str)
    parser.add_argument("--prog_name", default="train_eval", type=str)
    args = parser.parse_args()

    (
        df_filtered,
        preprocessor,
        scorers,
        X_questions,
        y_questions,
    ) = prepare_source_data()

    feature_selectors = {
        "Identity": (FunctionTransformer(), {}),
        "SelectFromModel-XGBoost": (
            SelectFromModel(
                XGBClassifier(
                    objective="binary:hinge",
                    eval_metric=roc_auc_score,
                    random_state=RANDOM_STATE,
                ),
                max_features=args.feature_num,
            ),
            {},
        ),
        "SelectFromModel-LinearSVM": (
            SelectFromModel(
                LinearSVC(C=1, dual=False, random_state=RANDOM_STATE),
                max_features=args.feature_num,
            ),
            {"feature_selector__estimator__C": np.logspace(-2, 0, 10)},
        ),
        "RFE": (
            RFE(
                LinearSVC(C=1, dual=False, random_state=RANDOM_STATE),
                n_features_to_select=args.feature_num,
            ),
            {"feature_selector__estimator__C": np.logspace(-2, 0, 10)},
        ),
        "ReliefF": (
            ReliefF(n_features_to_select=args.feature_num, n_jobs=1, n_neighbors=10),
            {},
        ),
    }
    for scorer in scorers.keys():
        hyperparam_dict = {
            "feature_selector__n_features": [args.feature_num],
        }
        if "LinearScorer" in scorer:
            hyperparam_dict["feature_selector__scorer__alpha[1]"] = np.logspace(
                -2, 2, 30
            )
        feature_selectors[f"MRMRBase_{scorer}"] = (
            MRMRBase(scorers[scorer]),
            hyperparam_dict,
        )
        feature_selectors[f"TopNSelector_{scorer}"] = (
            TopNSelector(scorers[scorer]),
            hyperparam_dict,
        )
        feature_selectors[f"StdDevSelector_{scorer}"] = (
            StdDevSelector(scorers[scorer]),
            {
                "feature_selector__std_dev": [
                    0.3 if ("Skipgram" in scorer or "FastText" in scorer) else 1
                ]
            },
        )

    # Print list of valid FS
    if args.list_scorers:
        print("Valid feature selectors:")
        for fs in feature_selectors.keys():
            print(f"\t{fs}")
        sys.exit(0)

    classifiers = {
        "XGBoost": (
            XGBClassifier(
                objective="binary:hinge",
                eval_metric=roc_auc_score,
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                verbosity=max(args.verbose - 1, 0),
            ),
            {
                "classifier__n_estimators": [2],
                "classifier__max_leaves": [2],
            },
        ),
        "LinearSVM": (
            LinearSVC(random_state=RANDOM_STATE, dual=True),
            {
                "classifier__C": np.logspace(-2, 0, 10),
            },
        ),
        "MLP": (
            MLPClassifier(random_state=RANDOM_STATE),
            {
                "classifier__activation": ["tanh", "relu"],
                "classifier__alpha": np.logspace(-4, 0, 10),
            },
        ),
        "GaussianNB": (GaussianNB(), {}),
        "KNearest": (KNeighborsClassifier(), {"classifier__n_neighbors": [3, 5, 7]}),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

    pipes = {}

    # Go over every possible combination of feature-selection schemes and classifiers.

    sel_fs = (
        args.sel_fs.split(",") if args.sel_fs is not None else feature_selectors.keys()
    )
    sel_class = (
        args.sel_class.split(",") if args.sel_class is not None else classifiers.keys()
    )

    # Iterate through sel_fs and find all feature selectors that match
    # the given string
    matching_fs = set()
    for fs in sel_fs:
        matching_fs.update([x for x in feature_selectors.keys() if fs in x])

    sel_fs = list(matching_fs)

    print(f"Selected feature selectors: {sel_fs}")

    for cf, fs in itertools.product(sel_class, sel_fs):
        pipes[(fs, cf)] = combine_elements(
            preprocessor, feature_selectors[fs], classifiers[cf]
        )

    results = []

    for name, p in pipes.items():
        pipe, params = p

        if args.prog_name == "train_eval":
            print(f"Fitting {name}")
            te_results, best_params, best_feats, feat_imp = train_eval(
                df_filtered, pipe, X_questions=X_questions, parameters=params
            )
            if args.best_params:
                print(f"Best Params for {name}")
                bp = [[k, v] for k, v in best_params.items()]
                print(tabulate(bp, tablefmt="latex"))
                print(tabulate(bp, tablefmt=args.table_fmt))

            if args.best_feats and best_feats is not None:
                print(f"--- Best Features for {name} ---")
                print(best_feats)
                print(f"--------------------------------")

            if args.feature_importance and feat_imp is not None:
                print(f"--- Feature Importance for {name} ---")
                # For each of the non-whole number keys in feat_imp, print the key and the value
                for k, v in feat_imp.items():
                    if k in ["mRMR Value", "Relevancy", "Redundancy"]:
                        # Format to 4 decimal places
                        feat_imp[k] = [f"{x:.4f}" for x in v]

                print(tabulate(feat_imp, "keys", tablefmt="latex"))
                print(tabulate(feat_imp, "keys", tablefmt=args.table_fmt))
                feat_imp_df = pd.DataFrame(feat_imp)
                feat_imp_df.to_excel(
                    os.path.join(
                        args.root_dir,
                        "results",
                        f"{name[0].replace('/', '-')}-{name[1].replace('/', '-')}-feat_imp.xlsx",
                    ),
                    sheet_name="Feature Importance",
                    index=False,
                )
                print(f"--------------------------------")

            results.append(
                {**{"feature_selector": name[0], "classifier": name[1]}, **te_results}
            )

        else:
            print(f"Running {args.prog_name}")
            locals()[args.prog_name](
                df_filtered,
                pipe,
                scorers,
                parameters=params,
            )

    if args.prog_name == "train_eval":
        print(tabulate(results, "keys", tablefmt="latex"))
        print(tabulate(results, "keys", tablefmt=args.table_fmt))
        # Split up the feature_selector names into separate columns
        sub_columns = ["feature_selector", "scorer", "model", "dset"]
        for result in results:
            fs_names = result["feature_selector"].split("_")
            for idx, sc in enumerate(sub_columns):
                if idx < len(fs_names):
                    result[sc] = fs_names[idx]
                else:
                    break

        # Resort columns so that the feature_selector, scorer, model, and dset columns are first
        results = [
            {
                **{s: ("" if s not in result else result[s]) for s in sub_columns},
                **result,
            }
            for result in results
        ]
        results_df = pd.DataFrame(results)

        # Include the sel_fs and sel_class in the filename
        sel_fs_fmt = "-".join(sel_fs) if args.sel_fs is not None else "all"
        sel_fs_fmt = sel_fs_fmt.replace("/", "-")
        sel_class_fmt = "-".join(sel_class) if args.sel_class is not None else "all"
        sel_class_fmt = sel_class_fmt.replace("/", "-")
        results_df.to_excel(
            f"results/{sel_fs_fmt}-{sel_class_fmt}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.xlsx",
            sheet_name="Results",
            index=False,
        )
