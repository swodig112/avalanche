import os
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

N_AVCH = 10000
N = 800
# nodes = np.zeros(N,dtype = int)
# nodes[0]=1
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
    # w, alpha, activation = params
    avchsize = initial
    while np.sum(nodes) > 0:
        avchsize += evolve_nodes(nodes, params, dt)
    return avchsize
    
params = (w, alpha, unity_activation)
# anum = [np.sum(nodes)]
# turnons = [0]
# for i in range(10000):
    # turnons.append(turnons[-1] + evolve_nodes(nodes, params))
    # anum.append(np.sum(nodes))
    
# plt.plot(anum)
# plt.show()
# avch = []
# for i in range(10000):
    # avch.append(direct_sample_avalanche_size(N, params, 1, 0.01))
def wrapper(_=None):
    return direct_sample_avalanche_size(N, params, 1, 0.01)
if __name__ == "__main__":
# if True:
    workers = os.cpu_count()
    avch = []
    with Pool(workers) as p:
    # avch = p.map(lambda x: direct_sample_avalanche_size(N, params, 1, 0.01), range(100))
        for res in tqdm(p.imap_unordered(wrapper, range(N_AVCH)), total=N_AVCH):
            avch.append(res)
        # avch = p.map(wrapper, range(1000))
    # counts, bins = np.histogram(avch,bins=100)#,range=(np.min(avch),np.max(avch)),bins=100000,density=True)
    # np.histogram_bin_edges
    bin_edges = np.logspace(np.log10(np.min(avch)),np.log10(np.max(avch)))
    counts, _ = np.histogram(avch,bins=bin_edges,density=True)
    # counts, _ = np.histogram(avch,bins=bin_edges,weights=np.ones(N_AVCH) / N_AVCH)
    bins = (bin_edges[1:] + bin_edges[:-1] ) / 2
    # fx = np.logspace(np.min(bins),np.max(bins),1000,base=10)
    # fx[0]=1
    # normalcons = np.sum(counts * (bin_edges[1:] - bin_edges[:-1]))
    # counts /= normalcons
    # print(normalcons)
    # print(counts)
    # print(bin_edges)
    # fx = np.logspace(np.log(np.min(bins)),np.log(np.max(bins)))
    fx = bins
    initial_guess = [1.0] 
    A, _ = curve_fit(power_func, bins, counts, p0=initial_guess)
    A = A[0]
    fy = power_func(fx,A)
    # fy = A/np.sqrt(fx*fx*fx)# + B
    # fy = A/np.sqrt(fx ** (3)) + B
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(fx,fy,label="Fitted Power Law")
    plt.plot(bin_edges[1:], counts, 'o', label="Simulated Data")
    plt.xlabel("Avalanche Size")
    plt.ylabel("P_n")
    plt.show()
