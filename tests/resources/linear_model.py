# pylint: disable=W0223
# IMPORTS SECTION #

import numpy as np
import torch
from torch.autograd import Variable

# SYNTHETIC DATA PREPARATION #

x_values = [i for i in range(11)]

x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2 * i + 1 for i in x_values]

y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


# DEFINING THE NETWORK FOR REGRESSION #


class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def main():
    # SECTION FOR HYPERPARAMETERS #

    inputDim = 1
    outputDim = 1
    learningRate = 0.01
    epochs = 100

    # INITIALIZING THE MODEL #

    model = LinearRegression(inputDim, outputDim)

    # FOR GPU #
    if torch.cuda.is_available():
        model.cuda()

    # INITIALIZING THE LOSS FUNCTION AND OPTIMIZER #

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    # TRAINING STEP #

    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train).cuda())
            labels = Variable(torch.from_numpy(y_train).cuda())
        else:
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        print("epoch {}, loss {}".format(epoch, loss.item()))

    # EVALUATION AND PREDICTION #

    with torch.no_grad():
        if torch.cuda.is_available():
            predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
        else:
            predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
        print(predicted)

    # SAVING THE MODEL #
    torch.save(model.state_dict(), "tests/resources/linear_state_dict.pt")
    torch.save(model, "tests/resources/linear_model.pt")
