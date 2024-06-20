import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def get_weight(data, device = "cuda"):
    labels = [d[1] for d in data]
    total = len(labels)
    labels = dict(Counter(labels))

    for key in labels:
        labels[key] = total / labels[key] 

    weights = torch.tensor([labels[i] for i in range(7)], dtype=torch.float32).to(device)
    return weights

def fit(model, train_data, test_data, loss_fn, opti, device):
    model.train()

    ret = []

    total = len(train_data.dataset)
    correct = 0
    loss = 0
    
    for X, y in train_data:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)

        L = loss_fn(y_hat, y)
        loss += L.item()
        L.backward()
        correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

        opti.step()
        opti.zero_grad()
    loss /= len(train_data)
    ret.extend([loss, correct/total*100])

    model.eval()

    total = len(test_data.dataset)
    correct = 0
    loss = 0

    with torch.no_grad():   
        for X, y in test_data:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            L = loss_fn(y_hat, y)
            loss += L.item()
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    loss /= len(test_data)
    ret.extend([loss, correct/total*100])

    return ret

def plot_result(results):
    plt.plot(results[:,0], label = "train_loss")
    plt.plot(results[:,2], label = "test_loss")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Loss")
    plt.ylabel("Epochs")
    plt.show()

    plt.plot(results[:,1], label = "train_accuracy")
    plt.plot(results[:,3], label = "test_accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Accuracy")
    plt.ylabel("Epochs")
    plt.show()

def make_confusion_matrix(model, test_data, num_class, device = "cuda"):
    confusion_matrix = torch.zeros((num_class, num_class))
    model = model.to(device)
    with torch.no_grad():
        for X, y in test_data:
            X, y = X.to(device), y.to(device)
            y_hat = model(X).argmax(1)
            y_hat = torch.vstack((y, y_hat)).T.to("cpu")
            for i in range(num_class):
                for j in range(num_class):
                    confusion_matrix[i][j] += ((y_hat == torch.tensor([i,j])).sum(axis=1) == 2).sum().item()
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, tick_label):
    plt.figure(dpi = 150)
    sns.heatmap(confusion_matrix, xticklabels = tick_label, yticklabels = tick_label, annot=True, fmt=".2g")
    plt.title("confusion_matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()