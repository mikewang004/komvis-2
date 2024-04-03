import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import generate_binary_structure, convolve
import scipy.constants as spc

kb = spc.Boltzmann
N = 50 #size of lattice square given by N x N 
J = 1 # coupling constant 
seed = 12
T = 3


def flip_spin(x):
    """Changes +1 spin to -1 and if spin = -1 to +1"""
    if x == 1:
        return -1
    elif x == -1:
        return +1
    else:
        raise Expection("Input has to be +1 or -1.")


class Lattice():
    def __init__(self, N, J, T, seed = None):
        self.N = N
        self.J = J
        self.T = T
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

        self.generate_spins()
        self.energy = self.get_energy(self.lattice)
        return 


    def generate_spins(self):
        """Takes N input size and generates N x N lattice. ALso assignes +1 (spin-up) or -1 (spin-down) value to it"""
        self.lattice = self.rng.integers(low = 0, high = 2, size = (N, N))*2 -1


    def get_energy(self, lattice):
        """Calculates the energy according to E = -J nearest-neighbours-sum s_i*s_j - H sum_i s_i"""
        #TODO double verify if it works; it should work now
        # First shift to the four nearest neighbours, calculate product and then sum the whole thing 
        lattice_product = np.zeros([4, self.N, self.N])
        k = 0
        for i in range(0, 2):
            for j in range(0, 2):
                # Shift array 
                lattice_shift = np.roll(lattice, 2*j - 1, axis = i)
                lattice_product[k, :] = lattice_shift * lattice
                k = k + 1
        energy = -J * np.sum(lattice_product)
        return energy

    def get_total_magnetisation(self, lattice):
        """Calculates total magnetisation according to M = sum_i s_i"""
        return np.sum(lattice)



class Simulation():
    def __init__(self, system, n_timesteps):
        self.system = system
        self.n_timesteps = n_timesteps


    def modify_system(self):
        """Modifies the system by one spin"""
        #TODO verify if np rng has to be called again each time 
        rng = np.random.default_rng()
        self.system.new_lattice = np.copy(self.system.lattice)
        index = (rng.integers(low = 0, high = self.system.N, size = 2))
        # Get value and change spin
        value = self.system.lattice[index[0], index[1]]
        new_value = flip_spin(value)
        self.system.new_lattice[index[0], index[1]] = new_value
        return 0;

    def run_simulation_one_step(self):
        self.modify_system()
        self.system.new_energy = self.system.get_energy(self.system.new_lattice)
        delta_energy = self.system.new_energy - self.system.energy
        if delta_energy <=0 or np.exp((-1/self.system.T) * delta_energy) > np.random.default_rng().random():
            #TODO verify if correct
            self.system.lattice = self.system.new_lattice
            self.system.energy = self.system.new_energy

    def run_simulation(self):
        magnetisation = np.zeros([self.n_timesteps])
        for i in range(0, self.n_timesteps):
            self.run_simulation_one_step()
            magnetisation[i] = self.system.get_total_magnetisation(self.system.lattice)
        return magnetisation



class Results(object):
    def __init__(self):
        return


def main():
    lattice = Lattice(N, J, T)
    simulation = Simulation(lattice, 100000)
    magnetisation = simulation.run_simulation()
    plt.plot(magnetisation/N**2)
    plt.title("Mean magnetisation")
    plt.xlabel("time")
    plt.ylabel("magnetisation")
    plt.show()

if __name__ == "__main__":

    main()