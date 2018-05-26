from pyt.modules import MLP, Swish
from pyt.testing import quick_dataset

from gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as td

import numpy as np
from colour import hsl2rgb
import arrow

from matplotlib import pyplot as plt


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 x_transform=None, y_transform=None):
        self.x_transform = x_transform
        self.y_transform = y_transform
        if x_transform is not None:
            train_x = self.x_transform(train_x)
        if y_transform is not None:
            train_y = self.y_transform(train_y)
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Our mean function is constant in the interval [-1,1]
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        # We use the RBF kernel as a universal approximator
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Return moddl output as GaussianRandomVariable
        return GaussianRandomVariable(mean_x, covar_x)

    def add_data(self, xs=None, ys=None):
        if self.x_transform:
            xs = self.x_transform(xs)
        xs = torch.cat([self.train_x, xs], 0)
        if self.y_transform:
            ys = self.y_transform(xs)
        ys = torch.cat([self.train_y, ys], 0)
        self.set_train_data(xs, ys)
        return


def log1p(inputs):
    return inputs.add(1.).log()


def xor(a, b):
    return a is not b


class TimeEstimator(nn.Module):
    # initialize likelihood and model
    likelihood = GaussianLikelihood(log_noise_bounds=(-5, 5))
    y_transform = lop1p

    def __init__(self, train_x, train_y, optimizer_fn=None):
        self.model = ExactGPModel(train_x,
                                  train_y,
                                  likelihood=TimeEstimator.likelihood,
                                  y_transform=TimeEstimator.y_transform)
        self.optimizer = optimizer_fn(self.model.parameters())

    def forward(self, inputs):
        return self.model(inputs)

    def optimize(training_iter,
                 train_x=None, train_y=None,
                 add_or_reset_data='add'
                 optimizer=None):

        assert (train_x is None) is (train_y is None)  # both or neither
        if train_x is not None:
            if add_or_reset_data == 'add':
                self.model.add_data(train_x, train_y)
            elif add_or_reset_data == 'reset':
                self.model.set_train_data(train_x, train_y)
            else:
                raise ValueError('add_or_reset_data must be "add" or "reset"')

        optimizer = optimizer or self.optimizer

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                       self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.model.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
                i + 1, training_iter, loss.data[0],
                self.covar_module.log_lengthscale.data[0, 0],
                self.likelihood.log_noise.data[0]
            ))
            optimizer.step()
        return self.model, self.likelihood

    def predict(self, xs):
        # Put model and likelihood into eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        with gpytorch.fast_pred_var():
            observed_pred = self.likelihood(self.model(xs))
        lower, upper = observed_pred.confidence_region()
        mean = observed_pred.mean()
        return mean, lower, upper
