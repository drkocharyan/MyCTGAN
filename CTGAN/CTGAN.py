from ctgan_tools import RNN_layer
import torch
from torch import nn



class Discriminator(nn.Module):
    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim

        dims = [dim] + discriminator_dim

        layers = [
            nn.Sequential(
                nn.Linear(dims[i - 1], dims[i]),
                nn.ReLU(),
                nn.Dropout(0.5)
            ) for i in range(1, len(dims))
        ]
        layers.append(nn.Linear(dims[-1], 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, input):
        assert input.size()[0] % self.pac == 0

        input = input.view(-1,self.pacdim)

        for layer in self.layers:
            input = layer(input)

        return input

    def get_gradient_penalty(self,real_data, fake_data, device='cpu', pac=10, lambda_=10):
        pass


class Generator(nn.Module):
    def __init__(self, emb_dim:int, hidden_states:list, output_dim:int ):
        super(Generator, self).__init__()

        dims = [emb_dim]
        for hidden_dim in hidden_states:
            dims.append(dims[-1] + hidden_dim)

        layers = [
            RNN_layer(dims[i - 1], dims[i]) for i in range(1, len(dims))
            ]
        layers.append(nn.Linear(dims[-1], output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
