# This project is based on the assignments completed for the Coursera specialization:
# Machine Learning by Andrew Ng (Stanford)

# Title: Decision trees and boosting
# By: Qidu(Quentin) Fu


# Decision tree ---------------------------------------------------------------
# --------------------------------------------------------------------------------

"""
Problem statement:
Suppose you are starting a company that grows and sells wild mushrooms.

Since not all mushrooms are edible, you'd like to be able to tell 
whether a given mushroom is edible or poisonous based on it's physical attributes
You have some existing data that you can use for this task.
Can you use the data to help you identify which mushrooms can be sold safely?

Note: The dataset used is for illustrative purposes only. 
It is not meant to be a guide on identifying edible mushrooms.
"""

# import libraries

import numpy as np
import matplotlib.pyplot as plt

# load the data
X_train = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]
)
y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])

# view the variables
print("First few elements of X_train: ", X_train[:3])
print("First few elements of y_train: ", y_train[:3])
print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)
# check the dimensions of the data
print("The shape of X_train is: ", X_train.shape)
print("The shape of y_train is: ", y_train.shape)
print("The number of training examples is: ", X_train.shape[0])
print("The number of features is: ", X_train.shape[1])


# calculate the entropy of the data
def compute_entropy(y):
    # initialize entropy
    entropy = 0
    if len(y) != 0:
        p1 = len(y[y == 1]) / len(y)
        if p1 not in (0, 1):
            entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        else:
            entropy = 0
    return entropy


def split_dataset(X, node_indices, feature):
    # initialize the left and right datasets
    left_indices = []
    right_indices = []
    # loop over the node indices
    for i in node_indices:
        # if the feature value is 0, add the index to the left dataset
        if X[i, feature] == 0:
            left_indices.append(i)
        # if the feature value is 1, add the index to the right dataset
        else:
            right_indices.append(i)
    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):
    # split the dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    # calculate the entropy of the node, left, and right datasets
    entropy_node = compute_entropy(y[node_indices])
    entropy_left = compute_entropy(y[left_indices])
    entropy_right = compute_entropy(y[right_indices])
    # calculate the weights
    weight_left = len(left_indices) / len(node_indices)
    weight_right = len(right_indices) / len(node_indices)
    # calculate the information gain
    information_gain = entropy_node - (
        weight_left * entropy_left + weight_right * entropy_right
    )
    return information_gain


def get_best_split(X, y, node_indices):
    num_features = X.shape[1]
    max_info_gain = 0
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
    return best_feature, max_info_gain


def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    return tree_building_helper(
        X, y, node_indices, branch_name, max_depth, current_depth, []
    )


def tree_building_helper(
    X, y, node_indices, branch_name, max_depth, current_depth, tree
):
    # Max depth reached
    if current_depth == max_depth:
        return tree
    # Otherwise, build the tree recursively
    best_feature, max_info_gain = get_best_split(X, y, node_indices)
    tree.append((current_depth, branch_name, best_feature, node_indices))

    # split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    # build the left tree
    tree_building_helper(X, y, left_indices, "left", max_depth, current_depth + 1, tree)
    # build the right tree
    tree_building_helper(
        X, y, right_indices, "right", max_depth, current_depth + 1, tree
    )
    return tree


# Tree ensemble with sklearn ---------------------------------------------------------------
# --------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from scipy import stats
import matplotlib.pyplot as plt

# Introduction
"""
#### Datatset
- This dataset is obtained from Kaggle: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

#### Context
- Cardiovascular disease (CVDs) is the number one cause of death globally, 
    taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. 
    Four out of five CVD deaths are due to heart attacks and strokes, 
    and one-third of these deaths occur prematurely in people under 70 years of age. 
    Heart failure is a common event caused by CVDs.
- People with cardiovascular disease or who are at high cardiovascular risk 
    (due to the presence of one or more risk factors such as hypertension, 
    diabetes, hyperlipidaemia or already established disease) need early detection and management.  
- This dataset contains 11 features that can be used to predict possible heart disease.
- Let's train a machine learning model to assist with diagnosing this disease.

#### Attribute Information
- Age: age of the patient [years]
- Sex: sex of the patient [M: Male, F: Female]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, 
    NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure [mm Hg]
- Cholesterol: serum cholesterol [mm/dl]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, 
    ST: having ST-T wave abnormality (T wave inversions 
    and/or ST elevation or depression of > 0.05 mV), 
    LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment 
    [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease: output class [1: heart disease, 0: Normal] 
"""

# Let's now load the dataset. As we can see above, the variables:

# - Sex
# - ChestPainType
# - RestingECG
# - ExerciseAngina
# - ST_Slope

# Are *categorical*, so we must one-hot encode them.

# Load the data
df = pd.read_csv("heart_failure.csv")
df.head()


# Build the ML pipeline with sklearn
class DecisionTreePipeline:
    def __init__(self, df):
        self.df = df
        self.pipeline = self._create_pipeline()

    def _remove_outliers_zscore(self, data, threshold=3):
        z_scores = np.abs(stats.zscore(data))
        data_no_outliers = data[(z_scores < threshold).all(axis=1)]
        return data_no_outliers

    def _split_numeric_categorical(self):
        if self.df is None:
            raise ValueError("No data to split")
        # Split the data into features and target variable
        num_features = []
        cat_features = []
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                cat_features.append(col)
            else:
                num_features.append(col)
        return num_features, cat_features

    def _create_pipeline(self):
        # Define the numeric and categorical transformers with imputation and scaling
        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("outlier_remover", FunctionTransformer(self._remove_outliers_zscore)),
                ("scaler", StandardScaler()),
            ]
        )
        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Define the features as numeric and categorical
        num_features, cat_features = self._split_numeric_categorical()

        # Create a ColumnTransformer to apply the transformations to the correct columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_features),
                ("cat", cat_transformer, cat_features),
            ],
            remainder="passthrough",  # Include other columns not specified above
        )

        # Create the full pipeline with preprocessing and modeling steps
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeClassifier()),
            ]
        )
        return pipeline

    def run_pipeline(self, test_size=0.2, random_state=0):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.df.drop("HeartDisease", axis=1),
            self.df["HeartDisease"],
            test_size=test_size,
            random_state=random_state,
        )
        # Fit the pipeline to the training data
        self.pipeline.fit(X_train, y_train)
        # Make predictions on the test set
        y_pred = self.pipeline.predict(X_test)
        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy


# Boosting with XGBoost ---------------------------------------------------------------
# --------------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    df.drop(["HeartDisease"], axis=1),
    df["HeartDisease"],
    train_size=0.8,
    random_state=0,
)

n = int(len(X_train) * 0.8)  ## Let's use 80% to train and 20% to eval
X_train_fit, X_train_eval, y_train_fit, y_train_eval = (
    X_train[:n],
    X_train[n:],
    y_train[:n],
    y_train[n:],
)
xgb_model = XGBClassifier(
    n_estimators=500, learning_rate=0.1, verbosity=1, random_state=0
)
xgb_model.fit(
    X_train_fit,
    y_train_fit,
    eval_set=[(X_train_eval, y_train_eval)],
    early_stopping_rounds=10,
)
print(xgb_model.best_iteration)
print(
    f"Metrics train:\n\tAccuracy score: {accuracy_score(y_train, xgb_model.predict(X_train),):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_val),y_val):.4f}"
)
