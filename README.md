# WildIA
UT Austin Computational Engineering Senior Project

## Rename images to standardized names

Load in a folder called ```train_data``` of images that are intended for training and validation. This folder should have one subfolder per species to analyze.
Run the ```modify_data.py``` file. This creates a directory called ```train``` with subfolders for each species.

Credit for this file goes to Akash Kumar https://github.com/AKASH2907/bird_species_classification.

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