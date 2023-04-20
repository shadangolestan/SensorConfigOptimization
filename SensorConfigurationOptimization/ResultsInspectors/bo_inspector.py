import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(
            results,
            xlabel="Number of iterations $n$",
            ylabel=r"Max objective value after $n$ iterations",
            ax=None, 
            name=None, 
            alpha=0.2, 
            yscale=None,
            color=None, 
            true_minimum=None, 
            plotDataPoints = True, 
            fontsize = 10,
            ls = '-',
            marker = '',
            **kwargs):
        """Plot one or several convergence traces.
        Parameters
        ----------
        args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
            The result(s) for which to plot the convergence trace.
            - if `OptimizeResult`, then draw the corresponding single trace;
            - if list of `OptimizeResult`, then draw the corresponding convergence
              traces in transparency, along with the average convergence trace;
            - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
              an `OptimizeResult` or a list of `OptimizeResult`.
        ax : `Axes`, optional
            The matplotlib axes on which to draw the plot, or `None` to create
            a new one.
        true_minimum : float, optional
            The true minimum value of the function, if known.
        yscale : None or string, optional
            The scale for the y-axis.
        Returns
        -------
        ax : `Axes`
            The matplotlib axes.
        """
        losses = list(results)

        n_calls = len(losses)
        iterations = range(1, n_calls + 1)
        maxs = [np.max(losses[:i]) for i in iterations]
        min_maxs = min(maxs)
        
        if plotDataPoints:
            cliped_losses = losses
            # cliped_losses = np.clip(losses, min_maxs, None)
        else:
            cliped_losses = None
        
        

        return plotter(iterations, maxs, cliped_losses, xlabel, ylabel, ax, name, alpha, yscale, color,
                                true_minimum, ls, marker, fontsize, **kwargs)
    
def plotter(
        x, y1, y2,
        xlabel="Number of iterations $n$",
        ylabel=r"Max objective value after $n$ iterations",
        ax=None, 
        name=None, 
        alpha=0.2, 
        yscale=None,
        color=None, 
        true_minimum=None, 
        ls = '-',
        marker = '',
        fontsize = 10,
        **kwargs):
    """Plot one or several convergence traces.
    Parameters
    ----------
    args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
        The result(s) for which to plot the convergence trace.
        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.
    ax : `Axes`, optional
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.
    true_minimum : float, optional
        The true minimum value of the function, if known.
    yscale : None or string, optional
        The scale for the y-axis.
    Returns
    -------
    ax : `Axes`
        The matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()

    # ax.set_title(name)
    ax.set_xlabel(xlabel, labelpad=-2, fontsize=fontsize)
    ax.set_ylabel(ylabel, labelpad=-4, fontsize=fontsize)
    # ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    ax.plot(x, y1, c=color, marker = marker, linestyle = ls, **kwargs)
    
    try:
        ax.scatter(x, y2, c=color, alpha=alpha)
    except:
        pass

    if true_minimum is not None:
        ax.axhline(true_minimum, linestyle="--", marker = marker,
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