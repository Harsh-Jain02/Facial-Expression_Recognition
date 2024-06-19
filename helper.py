import torch
import matplotlib.pyplot as plt
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