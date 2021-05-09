# this file defines a way to abstract away all the AI image in -> prediction out
import model
import torch
import os

CLASSES = [
    "RAINBOW_LORIKEET",
    "SULPHUR_CRESTED_COCKATOO",
    "ALEXANDRINE_PARAKEET",
    "GANG_GANG_COCKATOO",
    "SWIFT PARROT",
    "EMPTY"
]
MODELPATH = 'WildIA_net.pth'
batch_size = 16

net = model.Net(len(CLASSES))

if os.path.isfile(MODELPATH):
    print("Loading previous model...")
    net.load_state_dict(torch.load(MODELPATH))
    print("Model loaded.\n")

def classify(imagepath):
    '''
    Input: image path
    Output: species name, probability
    '''
    classifications = []    # list of classified things in order


    print("Loading images...")
    classify_imgs = model.ParrotDataset(imagepath, CLASSES)
    print("Images found...")
    classify_loader = torch.utils.data.DataLoader(
        classify_imgs, batch_size=batch_size, shuffle=True, num_workers=2
    )
    print("Images loaded.\n")


    print("Starting inference...\n")

    with torch.no_grad():
        # Once done with train_loader, run validation
        for images, labels in classify_loader:
            images = images.permute(0,3,1,2)
            outputs = net(images)
            probs, predicted = torch.max(outputs.data, 1)
            for prediction, label in zip(predicted, labels):
                class_idx = label.item()
                classifications.append(CLASSES[prediction])

    return classifications
