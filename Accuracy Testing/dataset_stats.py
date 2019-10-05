import os
import numpy as np
import csv
from Image_Class import Image_Class
import seaborn as sns
import pandas as pd
from matplotlib import pyplot

def get_stats(input_folder):
	#Empty lists for necessary statistics
	side_lengths = []
	pixel_areas = []
	sizes_bytes = []

	#Loop through images
	for each in os.listdir(input_folder):
		#Current image
		current_image_path = os.path.join(input_folder, each)
		current_image = Image_Class(current_image_path)

		#Add in the statistics
		side_lengths.append(current_image.get_side_lengths()[0]) 	#Width
		side_lengths.append(current_image.get_side_lengths()[1]) 	#Height
		pixel_areas.append(current_image.get_area())				#Area
		sizes_bytes.append(current_image.get_size())				#Size

	return side_lengths, pixel_areas, sizes_bytes


def get_overall_stats(input_folders):
	#Empty lists for necessary statistics
	side_lengths = []
	pixel_areas = []
	sizes_bytes = []
	category_side_lengths = []
	category_pixels = []
	category_bytes = []

	for input_folder in input_folders:
		lengths, areas, sizes = get_stats(input_folder)

		for each in lengths:
			side_lengths.append(each)
																#CHANGE THE CLASSES TO FIT YOUR OWN CLASSES
			if input_folder.split("/")[-1] == 'no':
				category_side_lengths.append('NO')
			elif input_folder.split("/")[-1] == 'hp':
				category_side_lengths.append('HP')
			elif input_folder.split("/")[-1] == 'tu':
				category_side_lengths.append('TA')
			elif input_folder.split("/")[-1] == 'tv':
				category_side_lengths.append('TVA')
			elif input_folder.split("/")[-1] == 'ss':
				category_side_lengths.append('SSA')

		for each in areas:
			pixel_areas.append(1.0*each/1000)

			if input_folder.split("/")[-1] == 'no':
				category_pixels.append('NO')
			elif input_folder.split("/")[-1] == 'hp':
				category_pixels.append('HP')
			elif input_folder.split("/")[-1] == 'tu':
				category_pixels.append('TA')
			elif input_folder.split("/")[-1] == 'tv':
				category_pixels.append('TVA')
			elif input_folder.split("/")[-1] == 'ss':
				category_pixels.append('SSA')

		for each in sizes:
			sizes_bytes.append(1.0*each/1000) #Convert to KB
			
			if input_folder.split("/")[-1] == 'no':
				category_bytes.append('NO')
			elif input_folder.split("/")[-1] == 'hp':
				category_bytes.append('HP')
			elif input_folder.split("/")[-1] == 'tu':
				category_bytes.append('TA')
			elif input_folder.split("/")[-1] == 'tv':
				category_bytes.append('TVA')
			elif input_folder.split("/")[-1] == 'ss':
				category_bytes.append('SSA')


	for i in range(len(sizes_bytes)-1, 0, -1):
		if sizes_bytes[i] > 1000:
			del sizes_bytes[i]
			del category_bytes[i]
			i-=1

	for i in range(len(pixel_areas)-1, 0, -1):
		if pixel_areas[i] > 1000:
			del pixel_areas[i]
			del category_pixels[i]
			i-=1

	for i in range(len(side_lengths)-1, 0, -1):
		if side_lengths[i] > 1000:
			del side_lengths[i]
			del category_side_lengths[i]
			i-=1

	#Create violinplots for each statistic
	sns.set_style("ticks", {"ytick.left":False}) 

	fig, ax = pyplot.subplots(figsize=(6, 3))	
	sns.set_palette(['blue', 'orange', 'red', 'yellow', 'green'])
	d = {'Crop Side Length (Pixels)': side_lengths, 'Polyp Type': category_side_lengths}
	df = pd.DataFrame(data=d)

	sns.violinplot(x = "Crop Side Length (Pixels)", y = "Polyp Type", data = df, scale = 'width', cut = 0, order = ['HP', 'NO', 'TA', 'TVA', 'SSA'], ax = ax)
	fig = ax.get_figure()
	fig.savefig('sidelengths.png', dpi=500, bbox_inches='tight')
	fig.clf()


	fig, ax1 = pyplot.subplots(figsize=(6, 3))
	d = {'Crop Area (Kilobytes)': sizes_bytes, 'Polyp Type': category_bytes}
	df = pd.DataFrame(data = d)

	sns.violinplot(x = "Crop Area (Kilobytes)", y = 'Polyp Type', data=df, scale = 'width', cut = 0, order = ['HP', 'NO', 'TA', 'TVA', 'SSA'], ax = ax1)
	fig = ax1.get_figure()
	fig.savefig('kbareas.png', dpi=500, bbox_inches='tight')
	fig.clf()


	fig, ax2 = pyplot.subplots(figsize=(6, 3))
	d = {'Crop Area (Thousands of Pixels)': pixel_areas, 'Polyp Type': category_pixels}
	df = pd.DataFrame(data = d)

	sns.violinplot(x = "Crop Area (Thousands of Pixels)", y = 'Polyp Type', data=df, scale = 'width', cut = 0, order = ['HP', 'NO', 'TA', 'TVA', 'SSA'], ax = ax2)
	fig = ax2.get_figure()
	fig.savefig('pixelareas.png', dpi=500, bbox_inches='tight')
	fig.clf()


	# #Write out stats to a csv if you want
	# with open('stats.csv', mode = 'w') as file:
	# 	writer = csv.writer(file, delimiter = ',', quoting = csv.QUOTE_MINIMAL)
	# 	writer.writerow(['Class', 'Pixels (thousands)', 'Area (Kilobytes)'])

	# 	for i in range(len(pixel_areas)):
	# 		writer.writerow(['Sessile Serrated', str(pixel_areas[i]/1000), str(sizes_bytes[i]*1000)])

	#Calculate stats
	average_side_length = round(np.mean(side_lengths), 4)
	stdev_side_length = round(np.std(side_lengths), 4)
	average_area = round(np.mean(pixel_areas), 4)
	stdev_area = round(np.std(pixel_areas), 4)
	total_area = np.sum(pixel_areas)
	average_size = round(np.mean(sizes_bytes), 4)
	stdev_size = round(np.std(sizes_bytes), 4)
	total_size = round(np.sum(sizes_bytes), 4)

	print("Side Lengths (Pixels): ", average_side_length, stdev_side_length)
	print("Total Area (Pixels): ", total_area)
	print("Areas (Pixels): ", average_area, stdev_area)
	print("Sizes (MB): ", average_size, stdev_size)
	print("Total Size (MB): ", total_size)


# Main #

#List of folders containing images to calculate statistics on
input_folders = ['simple_crops/no', 'simple_crops/tu', 'simple_crops/tv', 'simple_crops/hp', 'simple_crops/ss']

get_overall_stats(input_folders)





