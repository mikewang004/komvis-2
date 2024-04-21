import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as spc
from tqdm import tqdm

from visplot2 import *


class Lattice:
    def __init__(self, n_spins, temperature, init_type="random"):
        """Lattice of spins with a number of lattice operations
        Args:
            n_spins (int): number of spins along one of the axes of the square grid
            temperature (float): temperature of the system
            init_type (str): describes how the system should be initialized, either random or ordered
        """

        self.n_spins = n_spins
        self.temperature = temperature
        self.rng = np.random.default_rng()
        self.spingrid = self.initialize(init_type)
        self.proposed_state = np.zeros((self.n_spins, self.n_spins))
        self.latest_attempted_flip = (0, 0)
        self.J = 1
        self.H = 1

        return

    def initialize(self, init_type):
        if init_type == "random":
            generated_state = (
                2 * self.rng.integers(low=0, high=2, size=(self.n_spins, self.n_spins))
                - 1
            )
        else:
            print("not supported")
            generated_state = None

        return generated_state

    def generate_proposed_state(self):
        x_index = self.rng.integers(
            low=0,
            high=self.n_spins - 1,
        )
        y_index = self.rng.integers(
            low=0,
            high=self.n_spins - 1,
        )
        self.latest_attempted_flip = (x_index, y_index)

        # write the change to the state, revert later if necessary
        self.spingrid[self.latest_attempted_flip] = (
            -1 * self.spingrid[self.latest_attempted_flip]
        )
        return

    def revert_proposed_state(self):
        self.spingrid[self.latest_attempted_flip] = (
            -1 * self.spingrid[self.latest_attempted_flip]
        )
        return

    def get_proposed_delta_energy(self):
        """
        calculate the energy difference of a proposed state change
        """
        x_flip, y_flip = self.latest_attempted_flip
        delta_energy = (
            -2
            * self.J
            * self.spingrid[x_flip, y_flip]
            * (  # factor 2 due to taking difference between flipped and unflipped state.
                # use the modulo operator such that N interacts with ( N + 1) % N = 1
                self.spingrid[(x_flip + 1) % self.n_spins, y_flip]
                + self.spingrid[x_flip - 1, y_flip]
                + self.spingrid[x_flip, (y_flip + 1) % self.n_spins]
                + self.spingrid[x_flip, y_flip - 1]
            )
        )  # + self.H * (
        #  self.spingrid[x_flip, y_flip]
        # )
        return delta_energy

    def validate_or_revert_proposition(self):
        k_b = 1.3806503e-23
        delta_energy = self.get_proposed_delta_energy()
        beta = 1 / (k_b * self.temperature)
        if delta_energy < 0:
            pass
        else:
            acceptance_probability = np.exp(-beta * delta_energy)
            random_accept = self.rng.choice(
                [True, False], p=[acceptance_probability, 1 - acceptance_probability]
            )
            if random_accept:
                pass
            else:
                self.revert_proposed_state()


A = Lattice(50, 1)

print(A.latest_attempted_flip)

for i in range(0, 10000):
    A.generate_proposed_state()
    A.validate_or_revert_proposition()

plot_lattice_parallel(A.spingrid, A.spingrid)


#
