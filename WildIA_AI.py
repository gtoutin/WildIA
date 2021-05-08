# this file defines a way to abstract away all the AI image in -> prediction out
import model
import torch

CLASSES = [
    "RAINBOW_LORIKEET",
    "SULPHUR_CRESTED_COCKATOO",
    "ALEXANDRINE_PARAKEET",
    "GANG_GANG_COCKATOO",
    "SWIFT PARROT",
    "EMPTY"
]

batch_size = 16

net = model.Net(len(CLASSES))

def classify(imagepath):
    '''
    Input: image path
    Output: species name, probability
    '''
    classifications = {}    # dict with classification: probability


    print("Loading images...")
    classify_imgs = model.ParrotDataset(imagepath, CLASSES)
    print("Images found...")
    classify_loader = torch.utils.data.DataLoader(
        classify_imgs, batch_size=batch_size, shuffle=False, num_workers=2
    )
    print("Images loaded.\n")

    # class_accuracy = {idx: [] for idx in range(len(CLASSES))}

    for idx, (images, labels) in enumerate(classify_loader):
        outputs = net(images)
        probs, predicted = torch.max(outputs.data, 1)

    #     for prediction, label in zip(predicted, labels):
    #         class_idx = label.item()
    #         class_accuracy[class_idx].append(prediction)

    # print(class_accuracy)

        print('Predicted: ', ' '.join('%5s' % CLASSES[predicted[j]]
                                for j in range(4)))

    return probs, predicted