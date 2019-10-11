import csv
import os
import sys
from os.path import basename, join
EPS = sys.float_info.epsilon


def calculate_conf_matrix(master_file, real_answers, fake_answers):
    """
    Args:
        master_file (<TYPE>): <EXPLAIN THIS PARAM HERE>
        real_answers (<TYPE>): <EXPLAIN THIS PARAM HERE>
        fake_answers (<TYPE>): <EXPLAIN THIS PARAM HERE>
    """
    real = []
    predictions = []
    names = []

    # Get real values and predictions
    with open(master_file) as file:
        csv_reader = csv.reader(file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count is 0:
                line_count += 1
            else:
                real.append(row[2])
                # Copy-Paste pathologist's predictions into this row
                predictions.append(row[3])
                names.append(row[0])

    assert len(real) is len(predictions)

    # True positive -> predict fake is fake
    # False positive -> predict fake is real

    tp, fp, tn, fn = 0, 0, 0, 0
    for index in range(len(real)):

        # True negative
        if real[index] is 'R' and predictions[index] in real_answers:
            tn += 1
        # False negative
        elif real[index] is 'F' and predictions[index] in real_answers:
            fn += 1
        # True positive
        elif real[index] is 'F' and predictions[index] in fake_answers:
            tp += 1
        # False positive
        elif real[index] is 'R' and predictions[index] in fake_answers:
            fp += 1

    print("True Positives: {}".format(tp))
    print("False Positives: {}".format(fp))
    print("True Negatives: {}".format(tn))
    print("False Negatives: {}".format(fn))

    precision = 1.0*tp/(tp+fp+EPS)
    recall = 1.0*tp/(tp+fn+EPS)
    f1 = 2.0*precision*recall/(precision+recall+EPS)

    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1: {:.4f}".format(f1))

    return real, predictions, names


def retrieve_images(master_file, real_answers, fake_answers):
    """<EXPLAIN THIS FUNCTION HERE>

    Args:
        master_file (<TYPE>): <EXPLAIN THIS PARAM HERE>
        real_answers (<TYPE>): <EXPLAIN THIS PARAM HERE>
        fake_answers (<TYPE>): <EXPLAIN THIS PARAM HERE>
    """
    outcomes = ["true_negatives",
                "false_negatives",
                "true_positives",
                "false_positives"]
    real, predictions, names = calculate_conf_matrix(master_file, real_answers, fake_answers)

    # Create folders
    for outcome in outcomes:
        os.makedirs(outcome, exist_ok=True)

    # Save images
    for index in range(len(real)):
        real_label = real[index]
        pred_label = predictions[index]
        outcome = None
        if real_label is 'R' and pred_label in real_answers:
            outcome = outcomes[0]
        elif real_label is 'F' and pred_label in real_answers:
            outcome = outcomes[1]
        elif real_label is 'F' and pred_label in fake_answers:
            outcome = outcomes[2]
        elif real_label is 'R' and pred_label in fake_answers:
            outcome = outcomes[3]
        else:
            # Unrecognized outcome
            continue
        if outcome is not None:
            filepath = names[index]
            command = "cp -r {} {}".format(filepath, join(outcome, basename(filepath)))
            os.system(command)


if __name__ == "__main__":
    # List of ways pathologists can mark as real
    real_answers = ['R', ' ', '', 'r', 'real', 'Real']
    # List of ways pathologists can mark as fake
    fake_answers = ['F', 'f', 'fake']
    retrieve_images("master_file.csv", real_answers, fake_answers)
