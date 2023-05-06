
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
from scipy import stats
import wandb
import pandas as pd
import itertools


def plot_confusion_matrix(preds, labels):
# Create confusion matrix
    num_classes = len(config.idx2label)-1
    confusion_matrix = np.zeros((num_classes, num_classes))
    for pred, label in zip(preds, labels):
        for p, l in zip(pred, label):
            confusion_matrix[l][p] += 1
    
    # Normalize confusion matrix
    row_sums = confusion_matrix.sum(axis=1)
    confusion_matrix_norm = np.nan_to_num(confusion_matrix / row_sums[:, np.newaxis])

    # remove the last row and column
    # confusion_matrix_norm = confusion_matrix_norm[:-1, :-1]
    
    # Create labels for x-axis and y-axis
    x_axis_labels = [config.idx2label[i] for i in range(num_classes)]
    y_axis_labels = [config.idx2label[i] for i in range(num_classes)]
    
    # Plot confusion matrix with percentage values
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(confusion_matrix_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False, xticklabels=x_axis_labels, yticklabels=y_axis_labels, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.savefig("hw1/stud/img/confusion_matrix.png")


def plot_optimizers_comparison():
    api = wandb.Api()
    runs = api.runs("nlp_stats")
    data = []
    for run in runs:
        data.append({
            'optimizer': run.config['optimizer'],
            'learning_rate': run.config['learning_rate'],
            'val_f1_score': run.summary['val_f1_score']
        })

    df = pd.DataFrame(data)

    sns.set(style="whitegrid")
    sns.violinplot(x='optimizer', y='val_f1_score', data=df, inner="quartile")
    plt.title("Optimizers Comparison")
    plt.show()


def plot_hiddensize_comparison():
    results = {512:[], 1024: [], 2048: [], 4096: []}
    api = wandb.Api()
    runs = api.runs("nlp_stats")
    for run in runs:
        config = run.config
        results[config["hidden_layer"]].append(run.summary["val_f1_score"])

    lstm_units = sorted(list(results.keys()))
    medians = [np.median(results[unit]) for unit in lstm_units]
    conf_intervals = [stats.t.interval(0.95, len(results[unit])-1, loc=np.mean(results[unit]), scale=stats.sem(results[unit])) for unit in lstm_units]

    # Polynomial Regression
    coeffs = np.polyfit(lstm_units, medians, deg=2)
    poly_func = np.poly1d(coeffs)
    x = np.linspace(min(lstm_units), max(lstm_units), 100)
    y = poly_func(x)

    plt.plot(x, y, label="Polynomial Regression", color="mediumseagreen", linestyle="--")

    for unit, median, interval in zip(lstm_units, medians, conf_intervals):
        plt.plot([unit, unit], interval, color="dodgerblue", linewidth=2.5)
        plt.scatter(unit, median, color="dodgerblue")

    plt.xlabel("LSTM Units")
    plt.ylabel("F1-score")
    plt.legend()
    plt.show()

def plot_classifiers_comparison():
    api = wandb.Api()
    runs = api.runs("nlp_stats")
    data = []

    for run in runs:
        data.append({
            'classifier': run.config['classifier'],
            'learning_rate': run.config['learning_rate'],
            'val_f1_score': run.summary['val_f1_score']
        })

    df = pd.DataFrame(data)

    sns.set(style="whitegrid")
    sns.violinplot(x='classifier', y='val_f1_score', data=df, inner="quartile", split=True)
    plt.title("Classifiers Comparison")
    plt.show()

