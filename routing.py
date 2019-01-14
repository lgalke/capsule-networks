#!/usr/bin/env python3
import torch
from capsule_network import CapsuleLayer


"""
A toy routing example:

Consider dummy city
with following segments


-------------
|   |   |   |
| A | B | C |
|   |   |   |
-------------
|   |   |   |
| D | E | F |
|   |   |   |
-------------

with connections between adjacent segments

A -- B
A -- D
B -- C
B -- E
C -- F
D -- E
E -- F

"""

# Number of segments to consider
N_SEGMENTS = 6

# We sample new loads for each timestep
N_TIMESTEPS = 1

def generate_data(n_timesteps, n_segments):
    """ Generate toy data """
    # Connections among segments
    adjacency = torch.rand(n_segments, n_segments) < 0.5

    loads = torch.softmax(torch.rand(n_timesteps, n_segments), dim=1)
    # passengers: [timestep, src, dst]
    # If passenger is on diagonal, src == dst
    passengers = torch.softmax(torch.rand(n_timesteps, n_segments, n_segments), dim=1)

    # Inter-segment adjacency is static
    return adjacency, loads, passengers

def main():
    """ Run a toy experiment """
    print("Initializing %d segments randomly" % N_SEGMENTS)
    adjacency, loads, passengers = generate_data(N_TIMESTEPS, N_SEGMENTS)
    print("Load per segments:", loads, loads.size(), sep='\n')
    print("Passengers:", passengers, passengers.size(), sep='\n')
    print("Adjacency:", adjacency, adjacency.size(), sep='\n')

    caps1 = CapsuleLayer(N_SEGMENTS, N_SEGMENTS, 1, 1, num_iterations=1)
    print("Apply caps layer to passengers")
    out = caps1(passengers[0].unsqueeze(-1))
    print("Output size:", out.size())
    print("Output:", out.squeeze())
    # TODO check how to proceed in other impl


if __name__ == '__main__':
    main()

