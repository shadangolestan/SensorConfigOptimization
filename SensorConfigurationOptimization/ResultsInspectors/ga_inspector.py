import numpy as np
import sys

def plot_convergence(
                     results,
                     xlabel="Number of iterations $n$",
                     ylabel=r"Max objective value after $n$ iterations",
                     ax=None, name=None, alpha=0.2, yscale=None,
                     color=None, true_minimum=None, plotDataPoints = False, fontsize = 10, ls = '-',
                     **kwargs
                    ):
        
    losses = list(results)
    n_calls = len(losses)
    # iterations = range(1, n_calls + 1)
    iterations = range(1, 1001)
    maxs = [np.max(losses[:i]) for i in iterations]
    
    # print('len(maxs) 1', len(maxs))
    
    compensation = 1000 - len(maxs)
    
    for i in range(compensation):
            maxs.append(maxs[len(maxs) - 1])
    
    # print('len(maxs) 2', len(maxs))
    
    min_maxs = min(maxs)
    cliped_losses = np.clip(losses, min_maxs, None)
    
    if plotDataPoints:
        cliped_losses = np.clip(losses, min_maxs, None)
        compensation = 1000 - len(cliped_losses)
        cliped_losses = list(cliped_losses)

        for i in range(compensation):
                cliped_losses.append(None)

        cliped_losses = np.array(cliped_losses)
        
    else:
        cliped_losses = None
        
        
    
    
    return plotter(iterations, maxs, cliped_losses, xlabel, ylabel, ax, name, alpha, yscale, color,
                            true_minimum, ls = ls, fontsize = fontsize, **kwargs)
    
def plotter(
            x, y1, y2,
            xlabel="Number of iterations $n$",
            ylabel=r"Max objective value after $n$ iterations",
            ax=None, name=None, alpha=0.2, yscale=None,
            color=None, true_minimum=None, ls = '-', fontsize = 10,
            **kwargs
           ):
    
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()

    # ax.set_title("Convergence plot")
    ax.set_xlabel(xlabel, labelpad=-2, fontsize=fontsize)
    ax.set_ylabel(ylabel, labelpad=-4, fontsize=fontsize)
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    ax.plot(x, y1, c=color, label=name, linestyle = ls, **kwargs)
    
    
    try:
        ax.scatter(x, y2, c=color, alpha=alpha)
    except:
        pass
    

    if true_minimum is not None:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum is not None or name is not None:
        ax.legend(loc="upper right")
        
    return ax

def read_files(directory):
    import os
    import pickle

    class RenamingUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'SensorOptimizers.SimulatedAnnealing':
                module = 'SensorOptimizers.Greedy'
            return super().find_class(module, name)

    results = []
    
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            with open(os.path.join(directory, filename), 'rb') as f:
                # results.append(pickle.load(f))
                results.append(RenamingUnpickler(f).load())

    return results