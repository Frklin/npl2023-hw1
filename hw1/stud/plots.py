
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
from scipy import stats
import wandb
import pandas as pd

def plot_confusion_matrix(pred, labels):
    """
    This function plots the confusion matrix.
    """
    # pred is a list of lists of predictions associated to each token in the respective position.
    # Ex: Ex: [ ["O", "O", "O", "O", "O"], ["O", "O", "O", "O", "O", "B-ACTION", "O", "O", "O", "O"] ]
    # labels is a list of lists of labels associated to each token in the respective position.
    # Ex: Ex: [ ["O", "O", "O", "O", "O"], ["O", "O", "O", "O", "O", "B-ACTION", "O", "O", "O", "O"] ]
    # The two lists have the same length.
    # The inner lists have the same length.
    # plot the confusion matrix
    confusion_matrix = [[0 for _ in range(len(config.LABELS))] for _ in range(len(config.LABELS))]
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            confusion_matrix[config.LABELS.index(labels[i][j])][config.LABELS.index(pred[i][j])] += 1
    confusion_matrix = np.array(confusion_matrix)
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', xticklabels=config.LABELS, yticklabels=config.LABELS)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("hw1/stud/img/confusion_matrix.png")

def plot_optimizers_comparison():
    api = wandb.Api()
    runs = api.runs("nlp_stats")
    data = []
    print(runs[0].name)
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

