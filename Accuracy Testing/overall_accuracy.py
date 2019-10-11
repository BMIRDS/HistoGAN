import os
from os import listdir
from os.path import join, isfile, isdir
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import operator
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import roc_auc_score
from utils import get_classes, get_dataset, get_image_paths
from filter import filter_by_confidence, filter_by_confidence_binary
random.seed(0)


def roc(test_y, y_pred, ax, label_, color_):

    test_y_matrix = np.zeros((len(test_y), 2))
    pred_y_matrix = np.zeros((len(test_y), 2))

    for i in range(len(test_y_matrix)):
        test_y_matrix[i, test_y[i]] = 1
        pred_y_matrix[i, 0] = 1 - y_pred[i]
        pred_y_matrix[i, 1] = y_pred[i]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(test_y_matrix[:, i], pred_y_matrix[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.rcParams.update({'font.size': 16})
    lw = 2
    ax.plot(fpr[1], tpr[1], color=color_, lw=lw, label=label_ + ' (AUC = %0.2f)' % roc_auc[0], linestyle='-')


# Returns accuracy for multiple folders
def calculate_overall_accuracy(input_folders, input_classes, model_path, binary,
                               class_num_direc, positive_class, negative_class):
    correct_counter, total_counter = 0, 0

    tp, fp, tn, fn = 0, 0, 0, 0
    binary_labels = []
    predicted_labels = []
    probabilities = []

    # Loop through each folder
    for i in range(len(input_folders)):
        if binary is True:
            # Model binary values
            class_tp, class_fp, class_tn, class_fn, class_binary, class_predicted, class_probabilities =\
             filter_by_confidence_binary(
                input_folders[i],
                model_path,
                input_classes[i],
                class_num_direc,
                positive_class,
                negative_class)

            # Add to overall counter
            tp += class_tp
            fp += class_fp
            tn += class_tn
            fn += class_fn

            # Add in the predictions
            binary_labels = binary_labels + class_binary
            predicted_labels = predicted_labels + class_predicted
            probabilities = probabilities + class_probabilities
        else:
            # Get model's accuracy on current folder
            class_correct, class_total = filter_by_confidence(
                input_folders[i], model_path, 0, input_folders[i],
                input_classes[i], False, class_num_direc)
            # Add to overall counter
            correct_counter += class_correct
            total_counter += class_total

    if binary is False:
        print("----------------------------------------------------")
        print("Model: " + model_path)
        print("Accuracy: " + str(round(1.0*correct_counter/total_counter, 3)))
        print("----------------------------------------------------")
    elif binary is True:
        print("----------------------------------------------------")
        print("Model: " + model_path)
        print(tp, fp, tn, fn)
        print("Sensitivity: " + str(round(1.0*tp/(tp+fn), 3)))
        print("Specificity: " + str(round(1.0*tn/(tn+fp), 3)))
        print("AUC: " + str(round(roc_auc_score(binary_labels, probabilities), 3)))
        print("----------------------------------------------------")
        # roc(binary_labels, probabilities)
        return binary_labels, probabilities


if __name__ == "__main__":
    input_folders = []  # Should be a list of folders where each folder contains images of a separate class
    input_classes = []  # Classes' indexes should correspond with the folder they're with
    model_paths = []  # Direct paths to models
    model_labels = []  # What to call each model
    colors = []  # Colors for plotting
    binary = False
    do_roc = False
    class_num_direc = ""  # Directory to relate each class number o string
    positive_class = ""  # Which class should be referred to as positive
    negative_class = ""  # Which class should be referred to as negative
    if do_roc is True:
        plt.figure()
        ax = plt.subplot()
        ax.tick_params(labelsize=14)
    for i in range(len(model_paths)):
        model, label = model_paths[i], model_labels[i]
        binary_labels, probabilities = calculate_overall_accuracy(
            input_folders, input_classes, model, binary, positive_class, negative_class)
        if do_roc is True:
            roc(binary_labels, probabilities, ax, label, colors[i])

    if do_roc is True:
        plt.rcParams.update({'font.size': 16})
        box = ax.get_position()
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.rcParams.update({'font.size': 16})
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
        ax.legend(loc='lower right', fancybox=True, shadow=True, prop={'size': 12}).get_frame().set_edgecolor('black')
        plt.savefig('AUROC.png', dpi=1500, format='png', bbox_inches='tight')
