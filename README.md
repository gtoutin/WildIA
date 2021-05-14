# WildIA
UT Austin Computational Engineering Senior Project

Huge thanks goes to Alex Witt for his help on this project.

# Out-of-the-box usage

Out of the box, this model provides 70% accuracy on classifying images of swift parrots and 50% accuracy or under on sulphur-crested cockatoos, gang-gang cockatoos, rainbow lorikeets, alexandrine parakeets, and images without birds. For this reason, it is recommended that this model be used out-of-the-box solely on swift parrots as the other categories are not very reliable as of now.

## System requirements

You must be running Linux or MacOS to use the CLI. Windows Subsystem for Linux 1 is currently unsupported, and so is Windows itself.

## Installing the CLI

To install the CLI, ```cd``` into the root folder of this project and install the dependencies. An output folder is required.
```bash
cd WildIA
pip install -r requirements.txt
mkdir Output
```

## Using the CLI

To use the CLI, stay in the project folder and run ```python3 WildIA_CLI.py predict``` 
The model will be loaded and soon after a small window will appear to allow you to select the input images and output CSV file.  
After the CLI is finished running, you will have your top-1 classification for each image in the CSV file.


# Customizing the model

This model can be fine-tuned for your needs. Fork or clone this repository and retrain the model to suit your requirements.

## Training the model

Model training happens in 

## Data augmentation

These are relatively small training datasets, so run ```data_augmentation.py``` to apply image filters to strengthen the model.

Credit for this file goes to Akash Kumar https://github.com/AKASH2907/bird_species_classification.

## Create validation datasets.

The model must be validated. Run ```create_validation.py``` to do this. This will create a folder called ```valid``` where the validation images are stored.

Credit for this file goes to Akash Kumar https://github.com/AKASH2907/bird_species_classification.

## Train the model

Model training occurs in ```train_model.py```.
Ensure that the correct path to the training data is defined in the ```TRAINDIR``` variable. Set ```num_classes``` to the number of categories of labeled data there are (default is 4).
Run the file to load, train, and save the model.

Credit goes to Alex Witt for his assistance.

## Test the model

Model testing occurs in ```test_model.py```.
Ensure that the correct path to the testing data is defined in the ```TESTDIR``` variable. Set ```num_classes``` to the number of categories of labeled data there are (default is 4). Set ```batch_size``` to be how many images are tested at once (default is 4).
Run the file to load, train, and save the model.
