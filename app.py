

!pip install pandas numpy scikit-learn matplotlib seaborn gradio --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, r2_score
import gradio as gr


def chatbot_pipeline(file):
    df = pd.read_csv(file.name)

    # Cleaning filler values
    df.replace(['?', '-', '', 'NA', 'N/A'], np.nan, inplace=True)
    df.dropna(axis=1, thresh=0.6*len(df), inplace=True)

    # Impute missing values
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    df = pd.get_dummies(df, drop_first=True)

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    task_type = "classification" if y.nunique() <= 10 else "regression"

    # Handle imbalance (for classification only)
    if task_type == "classification":
        majority = df[df[y.name] == y.value_counts().idxmax()]
        minority = df[df[y.name] != y.value_counts().idxmax()]
        minority_upsampled = resample(minority, replace=True,
                                      n_samples=len(majority), random_state=42)
        df = pd.concat([majority, minority_upsampled])
        X = df.drop(y.name, axis=1)
        y = df[y.name]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = ""
    models = {}

    if task_type == "classification":
        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier()
        }
        for name, model in classifiers.items():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            results += f"{name} Accuracy: {acc:.4f}
"
            models[name] = model

    elif task_type == "regression":
        regressors = {
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor()
        }
        for name, model in regressors.items():
            model.fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            results += f"{name} R2 Score: {r2:.4f}
"
            models[name] = model

    else:
        model = KMeans(n_clusters=3)
        model.fit(X)
        results += f"KMeans Cluster Centers:
{model.cluster_centers_}
"
        models["KMeans"] = model

    # Save all models
    os.makedirs("models", exist_ok=True)
    for name, model in models.items():
        with open(f"models/{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    return results

gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.File(label="Upload Your Dataset (CSV Only)"),
    outputs="text",
    title="🧠 AI Data Science Chatbot",
    description="Upload a CSV file and get model analysis, training, and results!"
).launch()

