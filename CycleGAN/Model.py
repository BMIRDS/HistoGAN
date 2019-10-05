import os
import argparse

########## PRECONDITIONS ###########
# Select number of epochs
# The model_folder should be as follows:
#	requires these folders
#	"datasets/dataset_dir/trainA/"
#	"datasets/dataset_dir/trainB/"
#	"datasets/dataset_dir/testA/"
#	"datasets/dataset_dir/testB/"
# You should be in the path for the model you're training
# e.g. train_and_eval_model(50, "NOtoTU", "nototu_8", 1, "false", 100, "tu", "results", "resnet18_e11_va0.860.pt", False)
####################################
def train_and_eval_model(epoch, dataset_dir, model_folder, CUDA_DEVICE, continue_train, num_image, class_to_use, output_folder, model_path, misclassified):
	assert os.exists(dataset_dir)
	assert continue_train == "true" or continue_train == "false"

	#Train the model
	os.system("CUDA_VISIBLE_DEVICES=" + str(CUDA_DEVICE) + 
			  " python ../../code/main.py --phase=train --epoch=" + str(epoch) + 
			  " --dataset_dir=" + dataset_dir + " --continue_train=" + continue_train)

	#Test the model
	os.system("CUDA_VISIBLE_DEVICES=" + str(CUDA_DEVICE) +
			  " python ../../code/main.py --phase=test --dataset_dir=" + dataset_dir) 

	#Move folder for analysis
	os.system("cd ../../")
	os.makedirs(os.join("accuracy_testing", model_folder))
	os.system("cd models/" + model_folder)
	os.system("cp -r test/ " + os.join(os.join(accuracy_testing, model_folder), model_folder))
	os.system("cd ../../accuracy_testing/")

	#Get classifier's accuracay on the model
	os.system("python code/filter.py --input_folder=" + os.join(model_folder, model_folder) +
			  " --top_image=" + str(num_image) + " --class_to_use=" + class_to_use + " --output_folder=" + output_folder +
			  " --model_path=" + model_path + " --save_misclassified=" + str(misclassified))
	os.system("cd ../models/" + model_folder)


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type = int, help="number of epochs to train model for")
parser.add_argument("--dataset_dir", type = str, help = "name of datasets/dataset_dir/testA")
parser.add_argument("--model_folder", type = str, help = "name of folder you're currently in, where the model is")
parser.add_argument("--CUDA_DEVICE", type = int, help = "which GPU to use")
parser.add_argument("--continue_train", type = str, help = "true or false")
parser.add_argument("--num_image", type = int, help = "number of image to save")
parser.add_argument("--class_to_use", type = int, help = "target class of the model")
parser.add_argument("--output_folder", type = str, help = "where to output the top N images")
parser.add_argument("--model_path", type = str, help = "path to the classifier given you're in ./accuracy_testing/")
parser.add_argument("--misclassified", type = bool, help = "whether or not to save misclassified images")
args = parser.parse_args()

train_and_eval_model(args.epoch, args.dataset_dir, args.model_folder. args.CUDA_DEVICE, args.continue_train, args.num_image
					 args.class_to_use, args.output_folder, args.model_path, args.misclassified)










