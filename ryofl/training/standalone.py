import torch
import torch.nn as nn
import torch.optim as optim

from ryofl.data import utils_data


def standalone_training(model, trn_x, trn_y, transform, epochs, batch):

    # Select device to run the computation on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a DataLoader from the given arrays
    trn_dl = utils_data.make_dataloader(
        trn_x, trn_y, transform, shuffle=True, batch=batch)

    # Define a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(epochs):

        cur_loss = 0.0
        for i, data in enumerate(trn_dl, 0):
            # Send the data to the selected device
            x, y = data[0].to(device), data[1].to(device)

            # The gradients are zeroed out at each epocs to avoid accumulation
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Print loss every 10 mini-batches
            cur_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch, i + 1, cur_loss / 10))
                cur_loss = 0.0

