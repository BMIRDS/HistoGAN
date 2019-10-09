import torch.nn as nn
import os
from os import listdir
from os.path import join, isfile, isdir, basename
import torch
import operator
import argparse
import cv2
from torchvision import datasets, models, transforms
from PIL import ImageFile
from utils import get_classes, get_dataset, get_image_paths
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Takes in a model and a folder of generated images
# Outputs the n most confident images
def filter_by_confidence(synthetic_folder, model, n, output_folder, _class, misclassified, class_num_direc):
    # Set device for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_folder, exist_ok=True)
    if misclassified:
        os.makedirs("misclassified_images", exist_ok=True)

    # Load in the model
    active_model = torch.load(model)
    active_model.train(False)
    print("Loaded the model")

    # Results dictionary - key is image path, value is confidence
    path_results = {}

    # load the image dataset
    image_dataset = get_dataset(synthetic_folder)
    # synthetic folder should be in a folder of same name (e.g. syn_tu/syn_tu/)

    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=16, shuffle=False, num_workers=4)

    window_names = get_image_paths(join(synthetic_folder, synthetic_folder))
    class_num_to_class = {i: get_classes(class_num_direc)[i] for i in range(len(get_classes(class_num_direc)))}
    batch_num = 0
    correct_counter, total_counter = 0.0, 0.0

    for test_inputs, test_labels in dataloader:
        # Model predictions
        batch_window_names = window_names[batch_num*16:batch_num*16+16]
        test_inputs = test_inputs.to(device)
        test_outputs = active_model(test_inputs)
        softmax_test_outputs = nn.Softmax()(test_outputs)
        confidences, test_preds = torch.max(softmax_test_outputs, 1)

        for i in range(test_preds.shape[0]):
            if i >= len(batch_window_names):
                continue
            image_name = batch_window_names[i]
            confidence = confidences[i].data.item()
            predicted_class = class_num_to_class[test_preds[i].data.item()]

            # Check prediction
            if predicted_class is _class:
                path_results[image_name] = confidence
                correct_counter += 1
            elif predicted_class is not _class and misclassified is True:
                output_path = join(
                    "misclassified_images",
                    "{}_{}".format(predicted_class, basename(image_name)))
                os.system("cp -r {} {}".format(image_name, output_path))
            total_counter += 1
        batch_num += 1
    sorted_results = sorted(path_results.items(), key=operator.itemgetter(1), reverse=True)

    for i in range(n):
        original_path = sorted_results[i][0]
        confidence = "{:.3}".format(sorted_results[i][1])
        output_path = join(output_folder, "{}_{}".format(confidence, basename(original_path)))
        os.system("cp -r {} {}".format(original_path, output_path))

    print("---------------------------------------")
    print("{:.3}".format(correct_counter/total_counter))
    return correct_counter, total_counter


# Takes in a model and a folder of generated images
def filter_by_confidence_binary(synthetic_folder, model, _class, class_num_direc,
                                positive_class, negative_class):
    # Set device for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load in the model
    active_model = torch.load(model)
    active_model.train(False)
    print("Loaded the model")
    # load the image dataset
    # synthetic folder should be in a folder of same name (e.g. syn_tu/syn_tu/)
    image_dataset = get_dataset(synthetic_folder)
    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=16, shuffle=False, num_workers=4)
    num_test_images = len(dataloader)*16

    window_names = get_image_paths(join(synthetic_folder, synthetic_folder))
    class_num_to_class = {i: get_classes(class_num_direc)[i] for i in range(len(get_classes(class_num_direc)))}
    batch_num = 0

    tp, fp, tn, fn = 0, 0, 0, 0
    binary_labels = []
    predicted_labels = []
    probabilities = []

    for test_inputs, test_labels in dataloader:

        # Model predictions
        batch_window_names = window_names[batch_num*16:batch_num*16+16]
        test_inputs = test_inputs.to(device)
        test_outputs = active_model(test_inputs)
        softmax_test_outputs = nn.Softmax()(test_outputs)
        confidences, test_preds = torch.max(softmax_test_outputs, 1)

        for i in range(test_preds.shape[0]):
            image_name = batch_window_names[i]
            confidence = confidences[i].data.item()
            predicted_class = class_num_to_class[test_preds[i].data.item()]

            if predicted_class is positive_class and _class is positive_class:
                tp += 1
                binary_labels.append(1)
                predicted_labels.append(1)
                probabilities.append(confidence)
            elif predicted_class is positive_class and _class is negative_class:
                fp += 1
                binary_labels.append(0)
                predicted_labels.append(1)
                probabilities.append(confidence)
            elif predicted_class is negative_class and _class is negative_class:
                tn += 1
                binary_labels.append(0)
                predicted_labels.append(0)
                probabilities.append(1.0-confidence)
            elif predicted_class is negative_class and _class is positive_class:
                fn += 1
                binary_labels.append(1)
                predicted_labels.append(0)
                probabilities.append(1.0-confidence)
        batch_num += 1
    return tp, fp, tn, fn, binary_labels, predicted_labels, probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,
                        help="input path to folder to evaluate")
    parser.add_argument("--top_image", type=int,
                        help="number of image to save")
    parser.add_argument("--class_to_use", type=str,
                        help="class to test for")
    parser.add_argument("--output_folder", type=str,
                        help="where to save top images")
    parser.add_argument("--model_path", type=str,
                        help="path to model that should be used")
    parser.add_argument("--save_misclassified", action="store_true", default=False,
                        help="whether to save the images that are misclassified")
    args = parser.parse_args()

    # Class Number to class string directory
    class_num_direc = ""

    filter_by_confidence(
        args.input_folder,
        args.model_path,
        args.top_image,
        args.output_folder,
        args.class_to_use,
        args.save_misclassified,
        class_num_direc)
