from os.path import join, isfile, isdir
from torchvision import datasets, transforms


# getting the classes for classification
def get_classes(folder):
    subfolder_paths = sorted([f for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)])
    return subfolder_paths


# get full image paths
def get_image_paths(folder):
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    if join(folder, '.DS_Store') in image_paths:
        image_paths.remove(join(folder, '.DS_Store'))
    image_paths = sorted(image_paths)
    return image_paths


def get_dataset(data_path):
    PATH_MEAN = [0.7, 0.6, 0.7]
    PATH_STD = [0.15, 0.15, 0.15]
    data_transforms = {
        'normalize': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(PATH_MEAN, PATH_STD)
        ]),
    }
    return datasets.ImageFolder(data_path, data_transforms['normalize'])
