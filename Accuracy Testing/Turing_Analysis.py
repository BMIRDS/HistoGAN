import csv
import os


def calculate_conf_matrix(master_file):
	real = []
	predictions = []
	names = []

	#Get real values and predictions
	with open(master_file) as file:
		csv_reader = csv.reader(file, delimiter = ',')
		line_count = 0

		for row in csv_reader:
			if line_count == 0:
				line_count+= 1
			else:
				real.append(row[2])
				predictions.append(row[3]) #Copy-Paste pathologist's predictions into this row
				names.append(row[0])

	assert len(real) == len(predictions)

	#True positive -> predict fake is fake
	#False positive -> predict fake is real

	tp, fp, tn, fn = 0, 0, 0, 0
	for index in range(len(real)):

		#True negative
		if real[index] == 'R' and (predictions[index] == 'R' or predictions[index] == ' ' or predictions[index] == '' or predictions[index] == 'r'):
			tn += 1
		#False negative
		elif real[index] == 'F' and (predictions[index] == 'R' or predictions[index] == ' ' or predictions[index] == '' or predictions[index] == 'r'):
			fn += 1
		#True positive
		elif real[index] == 'F' and (predictions[index] == 'F' or predictions[index] == 'f'):
			tp += 1
		#False positive
		elif real[index] == 'R' and (predictions[index] == 'F' or predictions[index] == 'f'):
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

def retrieve_images(master_file):
	real, predictions, names = calculate_conf_matrix(master_file)

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
		if real[index] == 'R' and (predictions[index] == 'R' or predictions[index] == ' ' or predictions[index] == ''):
			os.system("cp -r " + names[index] + " true_negatives/" + names[index].split('/')[1])
		#False negative
		elif real[index] == 'F' and (predictions[index] == 'R' or predictions[index] == ' ' or predictions[index] == ''):
			os.system("cp -r " + names[index] + " false_negatives/" + names[index].split('/')[1])
		#True positive
		elif real[index] == 'F' and (predictions[index] == 'F' or predictions[index] == 'f'):
			os.system("cp -r " + names[index] + " true_positives/" + names[index].split('/')[1])
		#False positive
		elif real[index] == 'R' and (predictions[index] == 'F' or predictions[index] == 'f'):
			os.system("cp -r " + names[index] + " false_positives/" + names[index].split('/')[1])





### Main ###

retrieve_images("master_file.csv")