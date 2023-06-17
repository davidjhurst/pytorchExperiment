import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):

        # Defines a 4 lapyer neural network

        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):

        # The Rectified Linear Unit (relu) is the most commonly used activation function in
        # deep learning models. The function returns 0 if it receives any negative input,
        # but for any positive value x it returns that value back.
        # So it can be written as f(x)=max(0,x)

        x = F.relu(self.fc1(x))
        x = F.relu(selfelf.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # For the 4th layer. We pass x.
        # Then we return the softmax activiation statement

        return F.log_softmax(x, dim=1)

if __name__ == '__main__':

    print ("Starting....")

    train = datasets.MNIST("", train=True, download=True,
                           transform = transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

    net = Net()

    # X = torch.rand((28,28))  # 28 pixels * 28 pixels - 2d matrix
    # X = X.view(-1,28*28)     # convert to 1-d array

    # output = net(X)
    # print(output)

    # extract the paramaters tha can be optimised - not all layers are always optimised
    # - lr = Learning Rate. We want a decaying rate - but not this time

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    EPOCHS = 3   # Which means we go through our data 3 times

    for epoch in range (EPOCHS):
        for data in trainset:
            X, y = data
            net.zero_grad()
            output = net(X.view(-1, 28*28)) # flatten picture to 1-d array
            loss = F.nll_loss(output, y)    # calculate loss - there 2 methods
            loss.backward()
            optimizer.step()
        print(loss)

    print("End....ab branch1")