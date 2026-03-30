"""
suned_analysis Analysis scripts of output data from SU(N)ED
Copyright (C) 2026  Samuel Gozel, GNU GPLv3

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

#==============================================================================
# SET SOME USEFUL OPTIONS
#==============================================================================
    
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 16,
    'axes.labelsize': 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
})


class EDSpectrumPlotter:
    """
    Class to plot the data stored in an EDSpectrum instance
    """
    
    def __init__(self, edspectrum):
        """
        Constructor
        """
        self.edspectrum = edspectrum
        return
    
    def plot_eigenvalue(self, k: int):
        """
        Plot the convergence history of eigenvalue k.
        X axis: iteration index. 
        Y axis: Ritz value.
        """
        if k not in self.edspectrum.eigvalkbyit:
            raise KeyError(f"Eigenvalue index {k} not found. "
                           f"Available indices: {sorted(self.edspectrum.eigvalkbyit.keys())}")

        its = self.edspectrum.iterates(k)
        vals = self.edspectrum.values(k)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(its, vals, color='steelblue', linewidth=1.5, marker='o',
                markersize=3, label=f'Ritz value {k}')

        ax.set_xlabel('Lanczos iteration')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'Convergence of Ritz value $\\theta_{{{k}}}$')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        plt.show()
        return fig, ax
    
    
    def plot_all_eigenvalues(self, **kwargs):
        """
        Plot convergence history of all eigenvalues on the same plot
        
        Parameters
        ----------
        fromiterate : int [optional][default: 0]
            Lanczos iteration where to start plotting on x axis
        target_it : int [optional][default: max]
            Lanczos iteration where to extract unique eigenvalues
        topylim : float [optional]
            top ylim
        bottomylim : float [optional]
            bottom ylim
        """
        
        fromiterate = kwargs.get('fromiterate', 0)
        target_it = kwargs.get('target_it', self.edspectrum.num_iterates)
        
        eigvals = self.edspectrum.extract_at_iterate(it=target_it)
        
        colors = iter(cm.rainbow(np.linspace(0, 1, self.edspectrum.num_eigvalk)))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        for k in range(0, self.edspectrum.num_eigvalk):            
            c = next(colors)
            its = self.edspectrum.iterates(k)
            vals = self.edspectrum.values(k)
            
            ind = np.argwhere(its>=fromiterate).flatten()
            its = its[ind]
            vals = vals[ind]
            
            ax.plot(its, vals, 
                    color=c, 
                    linewidth=1.5, 
                    marker='o',
                    markersize=3, 
                    label=f'Ritz value {k}')
        
        # plot the unique extracted eigenvalues
        for eigv in eigvals:
            ax.plot(target_it, eigv, 'o', color='k')
        
        bottom, top = ax.get_ylim()
        topylim = kwargs.get('topylim', top)
        bottomylim = kwargs.get('bottomylim', bottom)
        ax.set_ylim(bottom=bottomylim, top=topylim)
        
        ax.set_xlabel('Lanczos iteration')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'Convergence of the first {self.edspectrum.num_eigvalk} Ritz values')
        #ax.legend(fontsize=11, loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        plt.show()
        
        return fig, ax

    
    def plot_successive_differences(self, k: int):
        """
        Plot |θ_k(m+1) - θ_k(m)| on a log scale.
        Reveals convergence rate and ghost arrivals as sudden spikes.
        X axis: iteration index. 
        Y axis: |successive difference| (log scale).
        """
        if k not in self.edspectrum.eigvalkbyit:
            raise KeyError(f"Eigenvalue index {k} not found. "
                           f"Available indices: {sorted(self.edspectrum.eigvalkbyit.keys())}")

        its  = self.edspectrum.iterates(k)
        vals = self.edspectrum.values(k)
        diffs = np.abs(np.diff(vals))

        # mid-points of iteration indices for the differences
        mid_its = its[:-1]

        # mask out exact zeros to avoid log(0)
        nonzero = diffs > 0

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.semilogy(mid_its[nonzero], diffs[nonzero],
                    color='steelblue', 
                    linewidth=1.5, 
                    marker='o', 
                    markersize=3,
                    label=f'$|\\theta_{{{k}}}(m+1) - \\theta_{{{k}}}(m)|$')

        ax.set_xlabel('Lanczos iteration')
        ax.set_ylabel('$|\\Delta\\theta|$')
        ax.set_title(f'Successive differences of Ritz value $\\theta_{{{k}}}$')
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5, which='both')

        fig.tight_layout()
        plt.show()
        return fig, ax
    
    
    def plot_full_spectrum(self, **kwargs):
        """
        Plot full spectrum
        
        Parameters
        ----------
        fromiterate : int [optional][default: 0]
            plot from this Lanczos iterate
        target_it : int [optional][default: max]
            Lanczos iteration where to extract unique eigenvalues for plotting
        topylim : float [optional]
            top ylim
        bottomylim : float [optional]
            bottom ylim
        """
        
        if self.edspectrum.has_full_spectrum:
            fromiterate = kwargs.get('fromiterate', int(0))
            target_it = kwargs.get('target_it', self.edspectrum.num_iterates)
            
            eigvals = self.edspectrum.get_unique_eigvals(it=target_it)
            print('number of identified eigvals: ', len(eigvals))
            
            colors = iter(cm.rainbow(np.linspace(0, 1, self.edspectrum.num_iterates)))
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for k_key in self.edspectrum.all_eigvalsbyk.keys():
                c = next(colors)            
                ax.plot(self.edspectrum.all_iterates[k_key], self.edspectrum.all_eigvalsbyk[k_key],
                        color=c,
                        linewidth=1.5, 
                        marker='o',
                        markersize=3)
            
            # plot unique eigenvalues
            for eigv in eigvals:
                ax.plot(target_it, eigv, 'o', color='k')
            
            ax.set_xlim(left=fromiterate)
            
            bottom, top = ax.get_ylim()
            topylim = kwargs.get('topylim', top)
            bottomylim = kwargs.get('bottomylim', bottom)        
            ax.set_ylim(bottom=bottomylim, top=topylim)
            
            ax.set_xlabel('Lanczos iteration')
            ax.set_ylabel('Eigenvalue')
            ax.set_title('Convergence of all Ritz values')
            ax.legend(fontsize=11, loc='lower left')
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()
            plt.show()
            return fig, ax
        else:
            raise ValueError("Missing full spectrum data.")
        return
