import numpy as np 
import matplotlib.pyplot as plt
import scipy.constants as spc
from tqdm import tqdm
kb = spc.Boltzmann
N = 50 #size of lattice square given by N x N 
J = 1 # coupling constant 
seed = 12
T = 1.0


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
        self.seed = seed
        self.reset_system()
        return 

    def reset_system(self):
        if self.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed)

        self.generate_spins()
        self.energy = self.get_energy(self.lattice)
        return 0

    def modify_system_one_spin(self):
        """Modifies the system by one spin"""
        rng = np.random.default_rng()
        self.new_lattice = np.copy(self.lattice)
        index = (rng.integers(low = 0, high = self.N, size = 2))
        # Get value and change spin
        value = self.lattice[index[0], index[1]]
        new_value = flip_spin(value)
        self.new_lattice[index[0], index[1]] = new_value
        return index;

    def generate_spins(self, start_temp = "inf"):
        """Takes N input size and generates N x N lattice. ALso assignes +1 (spin-up) or -1 (spin-down) value to it.
            start_temp = "inf" or "zero" """
        if start_temp == "inf":
            self.lattice = self.rng.integers(low = 0, high = 2, size = (N, N))*2 -1
        elif start_temp == "zero":
            self.lattice = np.ones([self.N, self.N])
        else:
            raise("start_temp has to be either 'zero' or 'inf'. ")

    def get_energy_single_spin(self, lattice, index):
        """Looks up spin at given index. Index should be a list with two entries."""
        energy = np.zeros(4); index_0 = index[0]; index_1 = index[1]
        for i in range(0, 2):
           for j in range(0, 2):
               k = 2*i - 1; l = 2*j - 1
               energy[i + j] = lattice[index[0], index[1]] * lattice[(index[0]+ k) % N, (index[1]+ l) % N]
        return -J * np.sum(energy) 

    def calc_delta_energy(self, lattice_old, lattice_new, index):
        """Calculates difference in energy when one spin is changed in the system"""
        delta = self.get_energy_single_spin(lattice_new, index) - self.get_energy_single_spin(lattice_old, index)
        return delta
        

    def get_energy(self, lattice):
        """Calculates the energy according to E = -J nearest-neighbours-sum s_i*s_j - H sum_i s_i"""
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
        self.results = Results()


    def run_simulation_one_step(self):
        index = self.system.modify_system_one_spin()
        self.system.new_energy = self.system.get_energy(self.system.new_lattice)
        delta_energy = self.system.new_energy - self.system.energy
        randfloat = np.random.default_rng().random()
        #print(f"delta energy = {delta_energy}, chance = {np.exp((-1/self.system.T) * delta_energy)}, float ={randfloat} ")
        if delta_energy <0 or np.exp((-1/self.system.T) * delta_energy) > 1 - randfloat:
            #TODO verify if correct
            #print("ok!")
            self.system.lattice = self.system.new_lattice
            self.system.energy = self.system.new_energy
        return 0;
    
    def run_simulation_one_step_one_spin(self):
        index = self.system.modify_system_one_spin()
        delta_energy = self.system.calc_delta_energy(self.system.lattice, self.system.new_lattice, index)
        self.system.new_energy = self.system.energy + delta_energy
        randfloat = np.random.default_rng().random()
        if delta_energy <0 or np.exp((-1/self.system.T) * delta_energy) > randfloat:
            #TODO verify if correct
            self.system.lattice = self.system.new_lattice
            self.system.energy = self.system.new_energy
        return 0;

    def run_simulation(self):
        magnetisation = np.zeros([self.n_timesteps])
        #for i in range(0, self.n_timesteps):
        for i in tqdm(range(0, self.n_timesteps), desc="runnin"):
            #self.run_simulation_one_step()
            self.run_simulation_one_step_one_spin()
            magnetisation[i] = self.system.get_total_magnetisation(self.system.lattice)
        self.results.magnetisation = magnetisation
        return magnetisation

    def run_multiple_temperatures(self, temps):
        """temps input as array"""
        magnetisation_multiple_temps = np.zeros([len(temps), self.n_timesteps])
        i = 0
        for temp in temps:
            self.system.T = temp
            magnetisation_multiple_temps[i, :] = self.run_simulation()
            self.system.reset_system()
            i = i + 1
        self.results.magnetisation_multiple_temps = magnetisation_multiple_temps


class Results(object):
    def __init__(self):
        return


def plot_magnetisation(magnetisation):
    #plt.scatter(temps, magnetisation[:, -1]/N**2)
    plt.plot(magnetisation/N**2)
    plt.title("Mean magnetisation")
    plt.xlabel("time")
    plt.ylabel("magnetisation")
    #plt.legend()
    plt.show()

def plot_magnetisation_multiple_temps(magnetisation, temps):
    plt.scatter(temps, np.abs(magnetisation[:, -1]/N**2))
    plt.title("Mean magnetisation")
    plt.xlabel("temperature")
    plt.ylabel("magnetisation")
    #plt.legend()
    plt.show()

def plot_lattice(lattice_start, lattice_end):
    #Plot lattice
    plt.matshow(lattice_start)
    plt.show()

    plt.matshow(lattice_end)
    plt.show()

def main():
    n_timesteps = 50000
    T = 1.0
    lattice = Lattice(N, J, T)
    lattice_start = lattice.lattice
    simulation = Simulation(lattice, n_timesteps)
    #temps = np.linspace(1.0, 4.0, 8)
    simulation.run_simulation()
    #simulation.run_multiple_temperatures(temps)
    lattice_end = simulation.system.lattice
    #plot_magnetisation_multiple_temps(simulation.results.magnetisation_multiple_temps, temps)
    plot_lattice(lattice_start, lattice_end)

if __name__ == "__main__":

    main()
