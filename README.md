# PolypGAN

Source code for *Generative Image Translation for Data Augmentation in Colorectal Histopathology Images (Wei et al.)*

<p align="center">
  <img width="460" height="600" src="https://github.com/BMIRDS/PolypGAN/blob/master/TransformationBases.png">
</p>


## 1. Packages used (dependencies):
- Numpy 1.15.2
- PyTorch 0.4.1
- Torchvision 0.2.1
- SciPy 1.3.0
- Seaborn 0.9.0
- Matplotlib 3.0.0
- Pandas 0.23.4
- OpenCV 3.4.2
- Scikit-Image 0.14.0
- Scikit-Learn 0.20.0
- Pillow 6.0.0
- Tensorflow-GPU 1.4.0

## 2. Folders included in this repository:
1. Accuracy Testing - all code used to analyze images (e.g. calculate accuracy, filter by confidence)
2. CycleGAN - all code used to train CycleGAN models. Original implementation from [xhujoy](https://github.com/xhujoy/CycleGAN-tensorflow).
3. DCGAN - all code used to train DCGAN models. Original implementation from [carpedm20](https://github.com/carpedm20/DCGAN-tensorflow).
4. DiscoGAN - all code used to train DiscoGAN models. Original implementation from [carpedm20](https://github.com/carpedm20/DiscoGAN-pytorch).
5. ResNet - all code used to train ResNet classifier models. Original implementation from 

## 3. Basic Usage
### 1. Train Generative Model(s)
A. Training CycleGAN
  - Make a "datasets/class1TOclass2/" folder
      - Subfolders: trainA (training images for class #1), trainB (training images for class #2), testA (original class 
        #1 images that will be used to generate fake class #2 images), testB (original class #2 images that will be used to 
        generate fake class #1 images)
  - Run CycleGAN/main.py and specify options with argparse (look at main.py for details about parameters); --phase should be 
    "train"

B. Training DCGAN
  - Make a "data/" folder
      - Subfolder: class1 (training images for class #1). Since DCGAN uses random noise for generation, no other folders are 
        needed.
  - Run DCGAN/main.py and specify options in code; "train" should be set to True

C. Training DiscoGAN
  - Make a "data/class1TOclass2/" folder
      - Subfolders: class1 (training images for class #1), class2 (training images from class #2)
  - Run DiscoGAN/main.py and specify options in DiscoGAN/config.py; "is_train" should be set to True
  
D. Training with Path-Rank-Filter from our paper
  - Use "Accuracy Testing/filter.py" to get a folder of most confident images (Note: pretrain a ResNet classifier on your 
    dataset in order to do this)
  - Use that folder as your new training folder
  
### 2. Generating Synthetic Images
A. Using CycleGAN
  - Run CycleGAN/main.py and specify options with argparse; --phase should be "test"
  - Generated images can be viewed in "CycleGAN/test/*.jpg"
  
B. Using DCGAN
  - Run DCGAN/main.py and specify options; "train" should be False, "visualize" should be True
  - Generated images can be viewed in "DCGAN/out/data - class1/samples/"

C. Using DiscoGAN
  - Run DiscoGAN/main.py and specify options in DiscoGAN/config.py; "is_train" should be False
  - Generated images can be viewed in "DiscoGAN/logs/class1TOclass2_timestamp/test/"
  
### 3. Training new ResNet with generated images
A. Data preparation
  - ResNet training requires "train_folder/train/class1/", "train_folder/train/class2/", "train_folder/val/class1", 
    "train_folder/val/class2"
  - Move generated images and real images into respective training and validation folders
B. Train ResNet
  - Run ResNet/3_train.py and specify options (e.g. number of layers) in ResNet/config.py
  - Models will be saved in "ResNet/checkpoints/"
  
### 4. Testing ResNet
A. Prepare folder
  - Place testing folders for each class in a folder of the same class name (e.g. "testing/class1/class1/" and 
    "testing/class2/class2")
  - Run "Accuracy Testing/overall_accuracy/" and specify options inside code file

## 4. Specific details about files in folder *Accuracy Testing*
- accuracy_tester.py: will return the accuracy of a multiclass model given a model path and a folder to test on
    - model path needs to be the direct file path to the model you want to test (e.g. "models/resnet18.pt")
    - The folder that you are testing on should include each class in a subfolder of the same name (e.g. if folder_to_test_on       
      = "val" and you are testing on class1 and class2, "val/class1/class1/" and "val/class2/class2/" must exist)
    - **Edit parameters inside code file**
- compare_images.py: will combine images of the same name from input folders next to each other
    - input folders should be a list of paths to folders containing images (e.g. "test/class1/" and "test/class2"); the order
      of the folders will also be the order of the images
        - Images *with the same name* will automatically be joined with a black border 
    - output folder is the folder that will be created to save the combined images
    - **Edit parameters inside code file**
    - Will yield something like 
![](https://github.com/BMIRDS/PolypGAN/blob/master/Example1.png)
- compress.py: contains various functions that can be used to change images
    - input folder will contain images that you want to modify (e.g. "images/a.jpg")
    - Automatically removes duplicate images (an image name that has "dup" in it)
    - Other parameters are explained in code
    - **Edit parameters using argparse when running code (e.g. python compress.py --compress=True)**
- dataset_stats.py: calculates statistics for your dataset (e.g. image area, image side lengths, image sizes)
    - input folders should be a list of paths to folders containing images (e.g. "simple_crops/*.jpg")
        -Each input folder should be a different class; the code calculates statistics for each folder you input
    - Replace each instance of our class with what your classes are
    - **Edit parameters inside code file**
- filter.py: saves the top n% of images by model confidence in a separate folder
    - model path needs to be the direct file path to the model you want to test (e.g. "models/resnet18.pt")
    - The folder that you are testing on should include each class in a subfolder of the same name (e.g. if folder_to_test_on       
      = "val" and you are testing on class1 and class2, "val/class1/class1/" and "val/class2/class2/" must exist)
    - Class to use is the name of the class that the input folder's images are of. Should match the name of the class you used 
      when training your classifier model
    - Top image is the number of images to save; to save the top 25% of images, set top image to 0.25 * total_number_of_images
    - Other parameters are explained in code
    - **Edit parameters using argparse when running code (e.g. python filter.py --top_image=12)**
- generate_images.py: can be used if augmented data is still not enough. This function was not tested in the paper. Will use 
  the left half of one image and the right half of another image to make a technically new image
    - Input folder is the path to a folder containing images you want to generate more data from
    - Other parameters are explained in code
    - **Edit parameters using argparse when running code**
- Image_Class.py: class file for any image. Used in other code files
- overall_accuracy.py: tests multiple models on multiple classes. Plots ROC curves for each model if specified.
    - Input folders: list of folders, each folder contains images of a separate class. Class should be in a subfolder of the 
      same name (e.g. if folder_to_test_on = "val" and you are testing on class1 and class2, "val/class1/class1/" and 
      "val/class2/class2/" must exist)
    - model paths should be the direct path to each model that will be tested. The order of the models should correspond with 
      the label for plotting
    - Change our classes with your classes 
    - **Edit parameters inside code file**
- rename_images.py: anonymize images by renaming them with numbers.
    - Parameters explained in code
    - **Edit parameters using argparse when running code**
- Turing_Analysis.py: analyzes a csv with pathologists' predictions vs true labels
    - CSV file should have these columns in this order: "Original Image Name", "New Name", "Real/Fake", "Pathologist 
      Prediction"
- Turing_Test.py: generates a turing test for pathologists
    - Also has an operation where you can get N random images from a folder
    - Other parameters explained in code
    - **Edit parameters using argparse when running code**
