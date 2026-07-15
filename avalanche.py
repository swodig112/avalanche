import os
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

N_AVCH = 100000
N = 800
alpha = 1
w = alpha / N  # the definition of w here is different from paper
#w is scaled here by N

def unity_activation(nodes):
    return nodes

def power_func(x, A):
    return A * (x**(-1.5))

def evolve_nodes(nodes, params, dt=0.01):
    N = nodes.size
    weight, alpha, activation = params
    A = np.sum(nodes)
    p = np.zeros_like(nodes,dtype=np.float64)
    p[:] = activation(weight * A * (1-nodes[:])) + alpha * nodes[:]
    p *= dt
    rans = np.random.random(N) < p
    n2 = np.copy(nodes)
    n2[rans] = 1 - n2[rans]
    turnons = n2 & (nodes == 0)
    nodes[:] = n2[:]
    return np.sum(turnons)

def direct_sample_avalanche_size(N, params, initial, dt):
    nodes = np.zeros(N, dtype=int)
    nodes[:initial] = 1
    avchsize = initial
    while np.sum(nodes) > 0:
        avchsize += evolve_nodes(nodes, params, dt)
    return avchsize
    
params = (w, alpha, unity_activation)


def wrapper(_=None):
    return direct_sample_avalanche_size(N, params, 1, 0.01)
if __name__ == "__main__":
    workers = os.cpu_count()
    avch = []
    with Pool(workers) as p:
        for res in tqdm(p.imap_unordered(wrapper, range(N_AVCH)), total=N_AVCH):
            avch.append(res)
    bin_edges = np.logspace(np.log10(np.min(avch)),np.log10(np.max(avch)), num=40)
    counts, _ = np.histogram(avch,bins=bin_edges,density=True)
    # bins = (bin_edges[1:] + bin_edges[:-1] ) / 2
    bins = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    fx = bins
    initial_guess = [1.0] 
    A, _ = curve_fit(power_func, bins, counts, p0=initial_guess)
    A = A[0]
    fy = power_func(fx,A)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(fx,fy,label="Fitted Power Law")
    plt.plot(bins, counts, 'o', label="Simulated Data")
    plt.xlabel("Avalanche Size")
    plt.ylabel("Probability Density")
    plt.show()
