import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as spc
from scipy.linalg import hankel
from tqdm import tqdm


correlation_times_1 = np.load("data/correlation-times.npy", allow_pickle = True)
correlation_times_2 = np.load("data/correlation-times-2.npy", allow_pickle = True)
correlation_times_3 = np.load("data/correlation-times-3.npy", allow_pickle = True)
correlation_times_new = np.load("data/correlation-times-new.npy", allow_pickle = True)


x = correlation_times_1 | correlation_times_2 | correlation_times_3
#x = correlation_times_1 | correlation_times_new
print(x)

plt.plot(np.linspace(1.0, 4.0, 15), x.values())
plt.show()