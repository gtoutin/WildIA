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
