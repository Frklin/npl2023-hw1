
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hw1.config as config

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
    plt.show()
