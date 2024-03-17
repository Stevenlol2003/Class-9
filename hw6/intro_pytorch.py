import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if training:
        # train_set
        data_set = datasets.FashionMNIST('./data',train=True,download=True,transform=custom_transform)
    else:    
        # test_set
        data_set = datasets.FashionMNIST('./data', train=False,transform=custom_transform)

    loader = torch.utils.data.DataLoader(data_set, batch_size = 64)

    return loader

def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    return model

def train_model(model, train_loader, criterion, T):
    """
    Train a neural network model.

    INPUT:
        model - the model produced by the previous function
        train_loader - the train DataLoader produced by the first function
        criterion - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        total_loss = 0.0
        correct = 0
        total_size = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_size += labels.size(0)
            # torch.max example from https://pytorch.org/docs/stable/generated/torch.max.html
            max_value, predicted = torch.max(outputs, 1) # max_value is not used, goal is to get predicted value
            correct += (predicted == labels).sum().item()

        accuracy = correct / total_size
        accuracy = round(accuracy, 2)
        average_loss = total_loss / len(train_loader)
        average_loss = round(average_loss, 3)

        print(f"Train Epoch: {epoch}    Accuracy: {correct}/{total_size} ({accuracy}%)   Loss: {average_loss}")
        
def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    total_loss = 0.0
    correct = 0
    total_size = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_size += labels.size(0)
            max_value, predicted = torch.max(outputs, 1) # max_value is not used, goal is to get predicted value
            correct += (predicted == labels).sum().item()

    accuracy = correct / total_size * 100.0
    accuracy = round(accuracy, 2)

    if show_loss:
        average_loss = total_loss / len(test_loader)
        average_loss = round(average_loss, 4)
        print(f"Average loss: {average_loss}")

    print(f"Accuracy: {accuracy}%")
    
def predict_label(model, test_images, index):
    """
    INPUT:
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    image = test_images[index]  # specific index of the image to be tested

    # Get the output logits
    logits = model(image.unsqueeze(0))

    # Softmax
    probabilities = F.softmax(logits, dim=1)
    # print(probabilities)

    # Convert to a list
    probabilities = probabilities.squeeze().tolist()

    # Create a list of class name and its prob
    class_probabilities = list(zip(class_names, probabilities))
    # print(class_probabilities)

    def sorting_key(item):
        return item[1]

    # Sort pairs list in descending order
    class_probabilities.sort(key=sorting_key, reverse=True)

    # Print result, top 3 classes and their probabilities
    for i in range(3):
        class_name, prob = class_probabilities[i]
        print(f'{class_name}: {prob * 100:.2f}%')

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    train_loader = get_data_loader()
    # print(type(train_loader))
    # print(train_loader.dataset)
    test_loader = get_data_loader(False)
    model = build_model()
    # print(model)
    model.train()

    criterion = nn.CrossEntropyLoss()
    T = 5
    train_model(model, train_loader, criterion, T)

    model.eval()

    evaluate_model(model, test_loader, criterion, show_loss=False)

    index = 1
    test_images = next(iter(test_loader))[0]
    # print(next(iter(test_loader))[0].shape)
    predict_label(model, test_images, index)