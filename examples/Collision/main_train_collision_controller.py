import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 50),
            nn.ReLU(),
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def load_data():
    """
    This function is heavily borrowed from https://github.com/XuankangLin/ART.git
    """
    fpath = 'data/collisions.csv'
    inputs, targets = [], []
    with open(fpath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            input = [float(s) for s in line[:-1]]
            inputs.append(input)

            label = int(line[-1])
            assert label in [0, 1]
            targets.append(label)

    inputs = torch.tensor(inputs).to(device)
    targets = torch.tensor(targets).to(device)
    labels = torch.zeros((len(inputs),2)).to(device)
    labels[range(len(labels)),targets] = 1.0
    return inputs, labels


def eval_test(torch_model, inputs, labels):
    with torch.no_grad():
        preds = torch_model(inputs)
        pred_labels = torch.argmax(preds,dim=1)
        true_labels = torch.argmax(labels,dim=1)
        accu = len(torch.nonzero(pred_labels==true_labels))/len(preds)
        print('Accuracy: ', accu)


if __name__ == '__main__':
    inputs, labels = load_data()
    torch_model = Net().to(device)
    lr = 0.01
    batch_size = 100
    epochs = 1000
    optimizer = optim.SGD(torch_model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()

    training_dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(training_dataset, batch_size, shuffle=True)

    for e in range(epochs):
        print('Epoch :' + str(e))
        eval_test(torch_model,inputs,labels)
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            predicts = torch_model(data)
            loss = criterion(target, predicts)
            loss.backward()
            optimizer.step()

    torch.save(torch_model, "collection_detection_model.pt")