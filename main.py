import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as spc
from tqdm import tqdm

from visplot2 import *

J = 1
kb = 1

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
        self.energy = self.get_initial_energy(self.spingrid)
        self.latest_attempted_flip = (0, 0)
        self.J = 1
        self.H = 1

        return

    def initialize(self, init_type = "random"):
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
            high=self.n_spins,
        )
        y_index = self.rng.integers(
            low=0,
            high=self.n_spins,
        )
        self.latest_attempted_flip = (x_index, y_index)

        # write the change to the state, revert later if necessary
        self.spingrid[self.latest_attempted_flip] = (
            -1 * self.spingrid[self.latest_attempted_flip]
        )
        return

    def get_initial_energy(self, lattice):
        """Calculates the energy according to E = -J nearest-neighbours-sum s_i*s_j - H sum_i s_i"""
        # First shift to the four nearest neighbours, calculate product and then sum the whole thing
        lattice_product = np.zeros([4, self.n_spins, self.n_spins])
        k = 0
        for i in range(0, 2):
            for j in range(0, 2):
                # Shift array
                lattice_shift = np.roll(lattice, 2 * j - 1, axis=i)
                lattice_product[k, :] = lattice_shift * lattice
                k = k + 1
        self.energy = -J * np.sum(lattice_product)
        return 0;

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
                + self.spingrid[(x_flip - 1) % self.n_spins, y_flip]
                + self.spingrid[x_flip, (y_flip + 1) % self.n_spins]
                + self.spingrid[x_flip, (y_flip - 1) % self.n_spins]
            )
        )  # + self.H * (
        #  self.spingrid[x_flip, y_flip]
        # )
        return delta_energy

    def validate_or_revert_proposition(self):
        #k_b = 1.3806503e-23
        k_b = 1
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
                self.energy = self.energy + delta_energy
                return 0;
            else:
                self.revert_proposed_state()
                return 0;


    def get_total_magnetisation(self, lattice):
        """Calculates total magnetisation according to M = sum_i s_i"""
        return np.sum(lattice)



class Simulation:
    def __init__(self, system, n_timesteps):
        self.system = system
        self.n_timesteps = n_timesteps
        self.results = Results()


    def run_simulation(self):
        for i in tqdm(range(0, self.n_timesteps), desc="runnin"):
            self.system.generate_proposed_state()
            self.system.validate_or_revert_proposition()
        return 0;



    def run_multiple_temperatures(self, temps):
        """temps input as array"""
        magnetisation_multiple_temps = np.zeros([len(temps), self.n_timesteps])
        i = 0
        for temp in temps:
            self.system.temperature = temp
            self.run_simulation()
            magnetisation_multiple_temps[i, :] = self.system.get_total_magnetisation(self.system.spingrid)
            self.system.initialize()
            i = i + 1
        self.results.magnetisation_multiple_temps = magnetisation_multiple_temps


class Results(object):
    def __init__(self):
        return



def main():
    n_spins = 50
    temperature = 1.5; temps = np.linspace(1.0, 4.0, 20)
    n_timesteps = 1000000*3
    lattice = Lattice(n_spins, temperature)
    lattice_start = np.copy(lattice.spingrid)
    simulation = Simulation(lattice, n_timesteps)
    simulation.run_simulation()
    #simulation.run_multiple_temperatures(temps)
    lattice_end = simulation.system.spingrid
    #plot_magnetisation_multiple_temps(simulation.results.magnetisation_multiple_temps, temps, n_spins)
    plot_lattice_parallel(lattice_start, lattice_end)


if __name__ == "__main__":
    main()
