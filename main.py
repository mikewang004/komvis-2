import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as spc
from scipy.linalg import hankel
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
        self.init_type = init_type
        self.rng = np.random.default_rng()
        self.spingrid = self.initialize()
        self.proposed_state = np.zeros((self.n_spins, self.n_spins))
        self.energy = self.get_initial_energy(self.spingrid)
        self.magnetisation = 0
        self.random_acceptance_probabilities = 0
        self.latest_attempted_flip = (0, 0)
        self.J = 1
        self.H = 1

        return

    def initialize(
        self,
    ):
        if self.init_type == "random":
            generated_state = (
                2 * self.rng.integers(low=0, high=2, size=(self.n_spins, self.n_spins))
                - 1
            )
            k_b = 1
            beta = 1 / (k_b * self.temperature)
            possible_delta_energy_values = np.array([0, 4, 8])
            self.random_acceptance_probabilities = np.exp(
                -beta * possible_delta_energy_values
            )
            print(f"{self.random_acceptance_probabilities=}")
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
        return 0

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
        delta_energy = self.get_proposed_delta_energy()
        if delta_energy < 0:
            pass
        else:
            if delta_energy == 0:
                acceptance_probability = self.random_acceptance_probabilities[0]
            elif delta_energy == 4:
                acceptance_probability = self.random_acceptance_probabilities[1]
            elif delta_energy == 8:
                acceptance_probability = self.random_acceptance_probabilities[2]
            else:
                acceptance_probability = np.exp(-possible_delta_energy_values)

            random_accept = self.rng.random() < acceptance_probability

            if random_accept:
                self.energy = self.energy + delta_energy
                return 0
            else:
                self.revert_proposed_state()
                return 0

    def update_magnetisation(self):
        lattice = self.spingrid
        """Calcnp.sum(lattice)ulates total magnetisation according to M = sum_i s_i"""
        magnetisation = np.sum(self.spingrid)
        self.magnetisation = np.abs(magnetisation)
        return 0


class Simulation:
    def __init__(self, system, n_timesteps, tau = None):
        self.system = system
        self.n_timesteps = n_timesteps
        self.results = Results()
        if tau is not None:
            self.correlation_times = tau

    def initialize_simulation(self):
        self.results.magnetisation_over_time = np.zeros(self.n_timesteps)
        return

    def run_simulation(self, temp):
        self.initialize_simulation()

        for t in tqdm(range(0, self.n_timesteps), desc="runnin"):
            self.system.generate_proposed_state()
            self.system.validate_or_revert_proposition()
            self.system.update_magnetisation()
            self.results.magnetisation_over_time[t] = self.system.magnetisation
        return 0

    def thermalize(self):
        block_size = 20000
        fluctuation = 1e10
        fluctuation_prev = 1e10
        lookback = 5
        threshold = 5

        # make an array with large variance so the loop starts off True
        previous_n_fluctuations = [fluctuation_prev,0.1*fluctuation_prev,0.01* fluctuation_prev,0.001* fluctuation_prev,fluctuation_prev]

        condition = True

        i=0
        while condition:
            condition = np.std(previous_n_fluctuations[-lookback:])  > threshold
            thermalize_magnetisation_over_time = np.zeros((block_size))
            for t in range(0, block_size):
                self.system.generate_proposed_state()
                self.system.validate_or_revert_proposition()
                total_spin = np.sum(self.system.spingrid)
                thermalize_magnetisation_over_time[t] = total_spin

            fluctuation  = np.std(thermalize_magnetisation_over_time)
            previous_n_fluctuations.append(fluctuation)


            i +=1

        return 0

    def run_multiple_temperatures(self, temps, n_reps=1):
        magnetisation_multiple_temps = {}
        i = 0
        for temp in temps:
            run_name = str(temp)
            self.system.temperature = temp
            self.system.initialize()
            self.thermalize()
            self.run_simulation(temp)
            magnetisation_multiple_temps[run_name] = (
                self.results.magnetisation_over_time
            )
            i = i + 1
        self.results.magnetisation_multiple_temps = magnetisation_multiple_temps


class Results(object):
    def __init__(self):
        self.correlation_function = 0
        return

    def get_correlation_functions(self):
        """Depricated, use new_correlation_functions instead.
        Calculate correlation function from the definition.
        The first large sum is calculated as: sum of j=0 to t_i - t_max of m(t_i) * m(t_i + t_j)
        In matrix representation this is the same as duplicating the vector for each column
        and then shifting the m(t_i) vector i up times and replacing the below-diagonal indices
        with zeros. This is accomplished with the scipy linalg hankel function.
        We only calculate the correlation function for t < t_max as the
        normalization diverges at t = t_max

        """
        # TODO implement proper time in MCMC steps
        # TODO also implement proper start after equilisation system
        correlation_functions = {}

        for name, run in self.magnetisation_multiple_temps.items():
            n_steps = len(run)
            matrix_shape = (n_steps, n_steps)
            time_axis = np.linspace(0, n_steps, num=n_steps)

            ones_above_antidiagonal = np.flip(np.triu(np.ones(matrix_shape)), axis=1)
            ones_upper_triangle = np.triu(np.ones(matrix_shape))

            normalization_vector = 1 / (n_steps - time_axis)

            # eliminate the last term which diverges due to division by zero
            normalization_vector[-1] = 0
            shifted_rolled_matrix = hankel(run)

            first_term = shifted_rolled_matrix @ run * normalization_vector
            second_term = normalization_vector * (ones_above_antidiagonal @ run)
            third_term = normalization_vector * (ones_upper_triangle @ run)
            correlation_function = first_term - second_term * third_term

            # eliminate the last term which diverges due to division by zero
            correlation_function[-1] = 0
            print(f"{correlation_function}")
            correlation_functions[name] = correlation_function
            self.correlation_functions = correlation_functions
        return correlation_functions


    def new_correlation_functions(self):
        correlation_functions = {}
        for name, run in self.magnetisation_multiple_temps.items():
            eff_run = run[-250000:]
            n_steps = len(eff_run)
            correlation_function = np.zeros([n_steps])
            time_axis = np.linspace(0, n_steps, num=n_steps)
            for t in range(n_steps):
                upper_limit = len(eff_run) - t
                normalisation_factor = 1/upper_limit
                first_term = normalisation_factor * np.correlate(eff_run[:upper_limit], eff_run[t:]) 
                second_term = normalisation_factor * np.sum(eff_run[:upper_limit])
                third_term = normalisation_factor * np.sum(eff_run[t:])
                correlation_function[t] = first_term - (second_term * third_term)
            correlation_function[-1] = 0
            correlation_functions[name] = correlation_function
        self.correlation_functions = correlation_functions
        return 0;


    def get_correlation_time(self):
        """Calculates the correlation times according to tau = chi(t)/chi(0)"""
        self.new_correlation_functions()
        correlation_times = {}
        for name, run in self.correlation_functions.items():
            # Apply first mask on values chi(t) < 0 
            stop_index = np.where(run < 0)[0][0]
            test = np.sum(run[:stop_index]/ run[0])
            correlation_times[name] = np.sum(run[:stop_index]/ run[0])
        print(correlation_times)
        np.save("data/correlation_times-12.npy", correlation_times)
        return correlation_times

    def stat_indep_std(self, corr_time, x):
        np.sqrt(2 * corr_time/t_max * np.mean(x**2) - np.mean(x)**2) 



def main():
    n_spins = 50
    temperature = 1.5
    n_timesteps =  250000*4 #int(250000*4) 
    temps = np.linspace(1.0, 4.0, 16)
    corr_times = np.loadtxt("data/correlation_times.txt")
    lattice = Lattice(n_spins, temperature)
    lattice_before = np.copy(lattice.spingrid)
    simulation = Simulation(lattice, n_timesteps)
    simulation.run_multiple_temperatures(temps)
    #plot_lattice_parallel(lattice_before, lattice.spingrid)
    # print(simulation.results.magnetisation_multiple_temps)
    #corrfuncs = simulation.results.get_correlation_functions()
    simulation.results.get_correlation_time()
    #keuze = corrfuncs["2.0"]
    #print(keuze)
    #plt.figure()
    #plt.plot(keuze)
    #plt.show()

    


if __name__ == "__main__":
    main()
