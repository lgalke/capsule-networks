#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from capsule_network import CapsuleLayer
import networkx as nx
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt


# Number of segments to consider
N_SEGMENTS = 10 

# We sample new loads for each timestep
N_TIMESTEPS = 1


def show_city(adjacency):
    adjacency = np.asarray(adjacency)
    graph = nx.Graph(adjacency)
    plt.figure(1)
    nx.draw(graph, with_labels=True)
    plt.show()


def generate_data(n_segments, n_timesteps):
    """ Generate toy data """
    # Random lobster looks as messy as typical (german) cities
    graph = nx.random_lobster(n_segments, 0.9, 0.9)
    adjacency = torch.from_numpy(nx.to_numpy_array(graph))
    # Connections among segments
    # adjacency = torch.rand(n_segments, n_segments) < 0.1
    # # Make connections symetric
    # adjacency |= adjacency.t()
    # # Insert diagonal for self-connections
    # adjacency |= torch.eye(n_segments).byte()

    loads = torch.softmax(torch.rand(n_timesteps, n_segments), dim=1)
    # passengers: [timestep, src, dst]
    # If passenger is on diagonal, src == dst
    passengers = torch.softmax(torch.rand(n_timesteps, n_segments, n_segments), dim=1)

    # Inter-segment adjacency is static
    return adjacency, loads, passengers

def main():
    """ Run a toy experiment """
    print("Initializing %d segments randomly" % N_SEGMENTS)
    adjacency, loads, passengers = generate_data(N_SEGMENTS, N_TIMESTEPS)
    print("Load per segments:", loads, loads.size(), sep='\n')
    print("Passengers:", passengers, passengers.size(), sep='\n')
    print("Adjacency:", adjacency, adjacency.size(), sep='\n')

    caps1 = CapsuleLayer(N_SEGMENTS, N_SEGMENTS, 1, 1, num_iterations=1)
    print("Apply caps layer to passengers")
    out = caps1(passengers[0].unsqueeze(-1))
    print("Output size:", out.size())
    print("Output:", out.squeeze())
    # TODO check how to proceed in other impl

    target = torch.eye(N_SEGMENTS)
    # criterion = nn.MSELoss(passengers, target)
    # print("Initial loss: ", criterion(passengers))

    show_city(adjacency)

if __name__ == '__main__':
    main()

