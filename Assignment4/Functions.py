import numpy as np
import pandas as pd
import time
import sklearn
from sklearn.tree import DecisionTreeClassifier

# COLUMN FILTER
def create_column_filter(df):
    df_copy = df.copy()
    col_keep = ["ACTIVE", "INDEX"]

    for col in df_copy.columns:
        if col not in col_keep:
            if df_copy[col].dropna().nunique() <= 1:
                df_copy = df_copy.drop(columns=[col])

    return df_copy, df_copy.columns.tolist()


def apply_column_filter(df, column_filter):
    df_copy = df.copy()
    df_copy = df_copy[column_filter]
    return df_copy


# ==============================================================================================


# NORMALIZATION MAPING
def create_normalization(df, normalizationtype):
    df_copy = df.copy()
    normalization = {}
    col_keep = ["ACTIVE", "INDEX"]

    numeric_cols = df_copy.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [c for c in numeric_cols if c not in col_keep]

    if normalizationtype == "minmax":
        min_value = df_copy[numeric_cols].min()
        max_value = df_copy[numeric_cols].max()
        df_copy[numeric_cols] = (df_copy[numeric_cols] - min_value) / (
            max_value - min_value
        )
        for c in numeric_cols:
            normalization[c] = ("minmax", min_value[c], max_value[c])
    elif normalizationtype == "zscore":
        mean_value = df_copy[numeric_cols].mean()
        std_value = df_copy[numeric_cols].std()
        df_copy[numeric_cols] = (df_copy[numeric_cols] - mean_value) / std_value
        for c in numeric_cols:
            normalization[c] = ("zscore", mean_value[c], std_value[c])
    else:
        print("Use minmax or zscore")
        normalization = None

    return df_copy, normalization


def apply_normalization(df, normalization):
    df_copy = df.copy()
    for col in normalization:
        normalization_type, value_1, value_2 = normalization[col]
        if normalization_type == "minmax":
            df_copy[col] = (df_copy[col] - value_1) / (value_2 - value_1)
            df_copy[col] = df_copy[col].clip(0, 1)

        elif normalization_type == "zscore":
            df_copy[col] = (df_copy[col] - value_1) / value_2

    return df_copy


# ==============================================================================================

# INPUTATION


def create_imputation(df):
    df_copy = df.copy()
    col_keep = ["ACTIVE", "INDEX"]
    imputation = {}

    cols = df_copy.select_dtypes(
        include=["int64", "float64", "category", "object"]
    ).columns
    cols = [n for n in cols if n not in col_keep]
    for col in cols:
        if df_copy[col].dtype in ["int64", "float64"]:
            if pd.isna(df_copy[col]).all():
                df_copy[col] = 0
                imputation[col] = 0
            else:
                mean_val = df_copy[col].mean()
                df_copy[col] = df_copy[col].fillna(mean_val)
                imputation[col] = mean_val

        elif df_copy[col].dtype == "O":
            if pd.isna(df_copy[col]).all():
                df_copy[col] = ""
                imputation[col] = ""
            else:
                mode_val = df_copy[col].mode().iloc[0]
                df_copy[col] = df_copy[col].fillna(mode_val)
                imputation[col] = mode_val

        elif df_copy[col].dtype == "category":
            if pd.isna(df_copy[col]).all():
                cat_val = df_copy[col].cat.categories[0]
                df_copy[col] = df_copy[col].fillna(cat_val)
                imputation[col] = cat_val
            else:
                mode_val = df_copy[col].mode().iloc[0]
                df_copy[col] = df_copy[col].fillna(mode_val)
                imputation[col] = mode_val

    return df_copy, imputation


def apply_imputation(df, imputation):
    df_copy = df.copy()
    for col in imputation:
        fill_value = imputation[col]  # directly use the value
        df_copy[col] = df_copy[col].fillna(fill_value)
    return df_copy


# ==============================================================================================


# ONE-HOT MAPPING


def create_one_hot(df):
    df_copy = df.copy()
    one_hot = {}
    category_cols = df_copy.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    category_cols_no_ids = [c for c in category_cols if c not in ["INDEX", "ACTIVE"]]
    for column in category_cols_no_ids:
        all_cat = set(cat for cat in df_copy[column])
        one_hot[column] = {cat: "" for cat in all_cat}
        for cat in all_cat:
            df_copy[f"{column} {cat}"] = df_copy[column].apply(
                lambda x: 1.0 if cat in x else 0.0
            )
            one_hot[column][cat] = f"{column} {cat}"
        df_copy = df_copy.drop(column, axis=1)

    return df_copy, one_hot


def apply_one_hot(df, one_hot):
    df_copy = df.copy()
    category_cols = df_copy.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    category_cols_no_ids = [c for c in category_cols if c not in ["INDEX", "ACTIVE"]]
    for kay, value in one_hot.items():
        for cat, new_name in value.items():
            df_copy[new_name] = df_copy[kay].apply(lambda x: 1.0 if cat in x else 0.0)
        df_copy = df_copy.drop(kay, axis=1)

    return df_copy


# ==============================================================================================


# DISCRETIZATION MAPING
def create_bins(df, nobins = 10, bintype = "equal-size"):
    df_copy = df.copy()
    binning = {}
    numeric_cols = df_copy.select_dtypes(include=['float', 'int']).columns.tolist()
    numeric_cols_no_ids = [c for c in numeric_cols if c not in ["INDEX", "ACTIVE"]]
    cols_to_discretize = []
    for col in numeric_cols:
        unique_vals = df[col].nunique()
        if unique_vals > 2:  # Only discretize if more than 2 unique values
            cols_to_discretize.append(col)
    
    print(cols_to_discretize)
    for column in cols_to_discretize:
        try:
            if bintype == "equal-width":
                binned_data, bins = pd.cut(df_copy[column], bins=nobins, labels=False, retbins=True)
            elif bintype == "equal-size":
                binned_data, bins = pd.qcut(df_copy[column], q=nobins, labels=False, retbins=True, duplicates='drop')
        except ValueError as e:
            print(f"Skipping column {column} due to error during binning: {e}")
            continue
    
        bins[0] = -np.inf
        bins[-1] = np.inf
        binning[column] = bins

        df_copy[column] = binned_data
    
        actual_nobins = len(np.unique(binned_data[binned_data.notna()]))
        if actual_nobins > 0:
            categories = pd.RangeIndex(start=0, stop=actual_nobins, step=1)
            df_copy[column] = pd.Categorical(df_copy[column], categories=categories, ordered=True)
        else:
            df_copy[column]= df_copy[column].astype('category')
    return df_copy, binning

def apply_bins(df, binning):
    df_copy = df.copy()

    for key, value in binning.items():
        new_binned_data, _ = pd.cut(df_copy[key], bins=value, labels=False, retbins=True)
        df_copy[key] = new_binned_data

        actual_nobins = len(np.unique(new_binned_data[new_binned_data.notna()]))
        if actual_nobins > 0:
            categories = pd.RangeIndex(start=0, stop=actual_nobins, step=1)
            df_copy[key] = pd.Categorical(df_copy[key], categories=categories, ordered=True)
        else:
            df_copy[key]= df_copy[key].astype('category')
    return df_copy




# ==============================================================================================


def accuracy(df, correctlabels):
    df_copy = df.copy()
    predictions = df_copy.idxmax(axis=1).values
    sum = 0
    for i in range(len(predictions)):
        if predictions[i] == correctlabels[i]:
            sum += 1

    return sum / len(correctlabels)


# ==============================================================================================


def brier_score(df, correctlabels):
    scores = []
    for i, label in enumerate(correctlabels):
        correct_vector = np.zeros(len(df.columns))
        index = np.where(df.columns == label)[0][0]
        correct_vector[index] = 1
        row_score = (df.iloc[i].values - correct_vector) ** 2
        scores.append(row_score)
    return np.sum(scores) / len(df)


# ==============================================================================================


def auc(df, correctlabels):
    classes = df.columns
    total_auc = 0
    for c in classes:
        scores = {}
        for index, score in enumerate(df[c].values):
            if score not in scores:
                scores[score] = [0, 0]

            if correctlabels[index] == c:
                scores[score][0] += 1
            else:
                scores[score][1] += 1
        sorted_scores = [scores[s] for s in sorted(scores.keys(), reverse=True)]
        positives = [s[0] for s in sorted_scores]
        negatives = [s[1] for s in sorted_scores]
        tot_pos = sum(positives)
        tot_neg = sum(negatives)
        cum_pos = 0
        auc_class = 0
        for i in range(len(positives)):
            if negatives[i] == 0:
                cum_pos += positives[i]
            elif positives[i] == 0:
                auc_class += (cum_pos / tot_pos) * (negatives[i] / tot_neg)
            else:
                auc_class += (cum_pos / tot_pos) * (negatives[i] / tot_neg)
                auc_class += (positives[i] / tot_pos) * (negatives[i] / tot_neg) / 2
                cum_pos += positives[i]

        class_weight = sum(1 for label in correctlabels if label == c) / len(df)
        total_auc += class_weight * auc_class

    return total_auc