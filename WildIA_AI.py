import model
import torch

net = model.Net(4)

def classify(images):
    '''
    Input: image path
    Output: species name, probability
    '''

    outputs = net(images)
    probs, predicted = torch.max(outputs.data, 1)

    print(probs)
    print(predicted)

    return probs, predicted
