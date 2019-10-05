import os

OG_path = '../pggan_256_sprue/repo/save/resl_8/'


def get_alpha_name(number):
	alphas = "abcdefghijklmnopqrstuvwxyz"
	first = int(number / 26 / 26)
	second = int(number / 26) % 26
	third = number % 26
	return alphas[first] + alphas[second] + alphas[third]


for j, each in enumerate(os.listdir(OG_path)):
	path = OG_path + each
	os.rename(path, OG_path + get_alpha_name(j)+'.jpg')