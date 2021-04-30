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