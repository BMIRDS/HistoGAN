import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type = str, help = "folder who's names need to be changed")
parser.add_argument("--counter", type = int, help = "number to put at the beginning")
parser.add_argument("--copy", type = str, help = "True means it'll create copies, otherwise files are directly renamed")
args = parser.parse_args()


output_folder = args.input_folder + str(args.counter)
if not os.path.exists(output_folder) and (args.copy == "True" or args.copy == "true"):
	os.makedirs(output_folder)

for each in os.listdir(args.input_folder):
	original = os.path.join(args.input_folder, each)
	new = os.path.join(output_folder, str(args.counter) + each)
	print(original)

	if args.copy == "True" or args.copy == "true":
		os.system("cp -r " + original + " " + new)
	else:
		os.rename(original, new)