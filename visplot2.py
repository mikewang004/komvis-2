import matplotlib.pyplot as plt
import numpy as np


def plot_magnetisation(magnetisation):
    # plt.scatter(temps, magnetisation[:, -1]/N**2)
    plt.plot(magnetisation / N**2)
    plt.title("Mean magnetisation")
    plt.xlabel("time")
    plt.ylabel("magnetisation")
    # plt.legend()
    plt.show()


def plot_magnetisation_multiple_temps(magnetisation, temps, n_spins, std=None):
    if std is None:
        plt.scatter(temps, np.abs(magnetisation[:, -1] / n_spins**2))
    else:
        plt.errorbar(temps, np.abs(magnetisation[:, -1] / n_spins**2), yerr=std[:, -1])
    plt.title("Mean magnetisation")
    plt.xlabel("temperature")
    plt.ylabel("magnetisation")
    # plt.legend()
    plt.savefig("magnetisation.pdf")
    plt.show()


def plot_lattice(lattice_start, lattice_end):
    # Plot lattice
    plt.matshow(lattice_start)
    plt.show()

    plt.matshow(lattice_end)
    plt.show()


def plot_lattice_parallel(lattice_start, lattice_end):
    fig, axis = plt.subplots(1, 2, layout="constrained", sharex=True, sharey=True)
    axis[0].set(title="old")
    axis[0].matshow(lattice_start)
    axis[1].set(title="new")
    axis[1].matshow(lattice_end)
    plt.show()


