import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as spc
from scipy.linalg import hankel
from tqdm import tqdm


max_i = 10
temps = np.linspace(1.0, 4.0, 16)
correlation_times = np.zeros([max_i, 16])
for i in range(1, max_i+1):
    x = np.load(f"data/correlation_times-{i}.npy", allow_pickle = True)
    corr_times = np.array(list(x.item().values()))
    correlation_times[i-1] = corr_times/2500


# Now plot correlation times 

for i in range(0, max_i):
    plt.scatter(temps, correlation_times[i, :])
plt.plot(temps, np.mean(correlation_times, axis = 0), marker = "D", color= "g")
plt.title("Correlation times as function of temperature for 2d Ising model, N = 50")
plt.xlabel("Temperature")
plt.ylabel("Correlation times (t/MC sweep)")
plt.show()
