# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()
        self.lrate = lrate
        self.loss_fn = loss_fn
        self.in_size = in_size
        self.out_size = out_size
        self.model = nn.Sequential(nn.Linear(self.in_size, self.out_size), nn.ReLU(), nn.Linear(self.out_size, 2))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lrate)


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        # return torch.ones(x.shape[0], 1)
        return self.model(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        # initialize gradients to zero
        self.optimizer.zero_grad()

        # forward + backward + optimize
        loss = self.loss_fn(self.forward(x), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """
    net = NeuralNet(0.001, nn.CrossEntropyLoss(), 3072, 32)
    losses = []
    yhats = np.zeros(dev_set.shape[0])
    start = 0
    N = train_set.shape[0]

    # Data Standardization
    train_set = (train_set-(torch.mean(train_set, dim=0)))/torch.std(train_set, dim=0)
    dev_set = (dev_set-(torch.mean(dev_set, dim=0)))/torch.std(dev_set, dim=0)

    # train and get loss value
    for i in range(n_iter):
        end = int((i % (N/batch_size)) * batch_size) + batch_size
        if end == 0:
            end = N
        loss_val = net.step(train_set[start:end, :], train_labels[start:end])
        losses.append(loss_val)
        start += batch_size
        if start == N:
            start = 0

    with torch.no_grad():
        for i, data in enumerate(dev_set):
            outputs = net.forward(data)
            if torch.argmax(outputs) == 1:
                yhats[i] = 1

    return losses,yhats,net
