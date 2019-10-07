import csv
import os


def calculate_conf_matrix(master_file, real_answers, fake_answers):
    real = []
    predictions = []
    names = []

    # Get real values and predictions
    with open(master_file) as file:
        csv_reader = csv.reader(file, delimiter = ',')
        line_count = 0

        for row in csv_reader:
            if line_count is 0:
                line_count+= 1
            else:
                real.append(row[2])
                predictions.append(row[3]) # Copy-Paste pathologist's predictions into this row
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

    print("True Positives: " + str(tp))
    print("False Positives: " + str(fp))
    print("True Negatives: " + str(tn))
    print("False Negatives: " + str(fn))

    precision = 1.0*tp/(tp+fp)
    recall = 1.0*tp/(tp+fn)
    f1 = 2.0*precision*recall/(precision+recall)

    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))

    return real, predictions, names

def retrieve_images(master_file, real_answers, fake_answers):
    real, predictions, names = calculate_conf_matrix(master_file, real_answers, fake_answers)

    #Create folders
    if not os.path.exists("true_positives/"):
        os.makedirs("true_positives/")
    if not os.path.exists("false_negatives/"):
        os.makedirs("false_negatives/")
    if not os.path.exists("true_negatives/"):
        os.makedirs("true_negatives/")
    if not os.path.exists("false_positives/"):
        os.makedirs("false_positives/")

    #Save images
    for index in range(len(real)):

        #True negative
        if real[index] is 'R' and predictions[index] in real_answers:
            os.system("cp -r " + names[index] + " true_negatives/" + names[index].split('/')[1])
        #False negative
        elif real[index] is 'F' and predictions[index] in real_answers:
            os.system("cp -r " + names[index] + " false_negatives/" + names[index].split('/')[1])
        #True positive
        elif real[index] is 'F' and predictions[index] in fake_answers:
            os.system("cp -r " + names[index] + " true_positives/" + names[index].split('/')[1])
        #False positive
        elif real[index] is 'R' and predictions[index] in fake_answers:
            os.system("cp -r " + names[index] + " false_positives/" + names[index].split('/')[1])




if __name__ == "__main__":

    real_answers = ['R', ' ', '', 'r', 'real', 'Real']                  # List of ways pathologists can mark as real
    fake_answers = ['F', 'f', 'fake']                                   # List of ways pathologists can mark as fake

    retrieve_images("master_file.csv", real_answers, fake_answers)