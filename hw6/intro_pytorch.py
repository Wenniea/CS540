import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
def get_data_loader(training=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.FashionMNIST('./ data', train=True,
                                      download=True, transform=transform)
    test_set = datasets.FashionMNIST('./ data', train=False,
                                     transform=transform)

    if training:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
        return train_loader
    else:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        return test_loader


def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        running_loss = 0
        model.train()
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            batch_size = inputs.size(0)

            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{total}({100 * correct / total:.2f}%) '
              f'Loss: {running_loss / total:.3f}')


def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        running_loss = 0
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        if show_loss:
            print(f'Average loss: {running_loss / total :.4f}')
            print(f'Accuracy: {100 * accuracy:.2f}%')
        else:
            print(f'Accuracy: {100 * accuracy:.2f}%')


def predict_label(model, test_images, index):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt'
        , 'Sneaker', 'Bag', 'Ankle Boot']

    with torch.no_grad():
        logits = model(test_images[index].unsqueeze(0))
        probs = F.softmax(logits, dim=1).squeeze()

    top_probs, top_labels = torch.topk(probs, 3)

    for i in range(len(top_probs)):
        prob_percent = top_probs[i] * 100
        label = class_names[top_labels[i].item()]
        print(f'{label}: {prob_percent:.2f}%')


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    model = build_model()
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    test_images, test_labels = next(iter(test_loader))
    predict_label(model, test_images, 1)
