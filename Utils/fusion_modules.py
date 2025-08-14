# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:14:15 2023

@author: Robbe Neyns
"""

import torch
import torch.nn as nn


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.ffc_x = nn.Linear(input_dim, output_dim)
        self.ffc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.ffc_x(x) + self.ffc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.ffc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.ffc_out(output)
        return x, y, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.ffc = nn.Linear(input_dim, 2 * dim)
        self.ffc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.ffc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.ffc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.ffc_x = nn.Linear(input_dim, dim)
        self.ffc_y = nn.Linear(input_dim, dim)
        self.ffc_out = nn.Linear(dim, output_dim)

        self.ffcx_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.ffc_x(x)
        out_y = self.ffc_y(y)

        if self.ffcx_gate:
            gate = self.sigmoid(out_x)
            output = self.ffc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.ffc_out(torch.mul(out_x, gate))

        return x, y, output