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

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 16,
    'axes.labelsize': 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
})



class CounterLog:
    """
    Class to load and plot the counts of off-diag pairs, which represent the
    communication pattern of MPI jobs
    """
    
    def __init__(self, folderpath):
        """
        Constructor
        """
        if not os.path.isdir(folderpath):
            raise FileNotFoundError(f'Folder {folderpath} does not exist.')
        self.folderpath = folderpath
        
        # get all files
        pattern = re.compile(r'^counts_k(\d+)\.log$')
        self.filenames = [
            os.path.join(self.folderpath, f) for f in os.listdir(self.folderpath)
            if os.path.isfile(os.path.join(self.folderpath, f)) and pattern.match(f)
        ]
        
        # put files in ascending order relative to k value
        self.filenames.sort(key=lambda f: int(pattern.match(os.path.basename(f)).group(1)))
        
        if not self.filenames:
           raise FileNotFoundError(f'No files matching counts_k<k>.log found in {self.folderpath}')
        
        self.n = len(self.filenames) + 2
        # transposition (0, 1) is diagonal; SUNED does not output anything for it
        # for an irrep with n boxes, there are n-1 adjacent transpositions
        
        # extract number of processes
        tmp = np.loadtxt(self.filenames[0])
        self.nprocs = tmp.shape[0]
        
        
        # for each file, extract the matrix, and the common max across all files
        self.counts = []
        self.local_counts = []
        self.remote_counts = []
        self.common_vmax = 0
        
        # add 0 values for transposition (0, 1)
        self.counts.append(np.zeros(shape=(self.nprocs, self.nprocs), dtype=int))
        self.local_counts.append(0)
        self.remote_counts.append(0)
        
        for i, filename in enumerate(self.filenames):
            
            counts = np.loadtxt(filename)
            
            local  = np.trace(counts)         # local pairs
            remote = (counts.sum() - local)/2 # remote pairs - divide by 2 to avoid double-counting with friend rank
            
            self.counts.append(counts)
            self.local_counts.append(local)
            self.remote_counts.append(remote)
            self.common_vmax = max(self.common_vmax, counts.max())
        
        self.local_counts = np.array(self.local_counts)
        self.remote_counts = np.array(self.remote_counts)
        
        return
    
    
    def plot(self, k, **kwargs):
        """
        Plot the communication pattern matrix for transposition (k, k+1)
        
        Parameters
        ----------
        k : int
            transposition (k, k+1)
        show_values : bool [optional][default: False]
            write values in matrix cells
        one_based : bool [optional][default: False]
            if True, the first transposition is k=1. Otherwise, it is k=0
        log10 : bool [optional][default: False]
            if True, plot the base-10 logarithm of the counts
        vmax : float [optional][default: None]
            vmax for color scheme
        ax : axis handle [optional][default: None]
            axis handle
        savefig : bool [optional][default: False]
            if True, save figure to disk
        
        Returns
        -------
        fig, ax : figure handles
        
        """
        
        one_based = kwargs.get('one_based', False)
        if one_based == True:
            index = k - 1
            if k==0:
                raise ValueError('For one-based transposition indexing, k must be strictly greater than 0.')
        else:
            index = k
        
        if index >= self.n-1:
            raise ValueError(f'Adjacent transposition ({k}, {k+1}) is invalid.')
        
        show_values = kwargs.get('show_values', False)
        log10 = kwargs.get('log10', False)
        vmax = kwargs.get('vmax', None)
        ax = kwargs.get('ax', None)
        do_save = kwargs.get('savefig', False)
        
        pixels_per_cell = int(8)
        pixels = self.nprocs * pixels_per_cell
        dpi = self.nprocs
        figsize = pixels / dpi
        
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=(figsize, figsize))
        
        counts = self.counts[index]
        if vmax == None:
            vmax = counts.max()
        
        if log10:
            counts = np.log10(counts)
            vmax = np.log10(vmax)
        
        # Custom colormap: green (0) to dark red (max)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'green_to_darkred', ['green', 'yellow', 'darkred']
        )
        
        im = ax.imshow(counts, 
                       cmap=cmap, 
                       vmin=0, 
                       vmax=vmax, 
                       aspect="equal",
                       interpolation='nearest')
        
        if show_values:
            for i in range(self.nprocs):
                for j in range(self.nprocs):
                    ax.text(j, i, 
                            str(int(counts[i, j])),
                            ha='center', 
                            va='center', 
                            fontsize=7, 
                            color='black')
        
        
        '''
        # Grid lines to separate cells
        ax.set_xticks(np.arange(-0.5, self.nprocs, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.nprocs, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.05)
        ax.tick_params(which='minor', bottom=False, left=False)
        '''
        
        lab = 'Off-diagonal pair count'
        if log10:
            lab += ' ($\log_{10}$)'
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, 
                     label=lab)
        
        ax.set_title(r'$\tau_{' + f'{k}, {k+1}' + '}$ ' + f'(world size: {self.nprocs})')
        ax.set_xlabel('MPI process')
        ax.set_ylabel('MPI process')
        
        plt.tight_layout()
        if do_save == True:
            plt.savefig(f'counts_k{k}.pdf', 
                        dpi=dpi, 
                        format='pdf')
        
        return fig, ax
    
    
    def plot_all(self, **kwargs):
        """
        Plot the communication pattern matrix for all transpositions
        
        Parameters
        ----------
        show_values : bool [optional][default: False]
            write values in matrix cells
        one_based : bool [optional][default: False]
            if True, the first transposition is k=1. Otherwise, it is k=0
        log10 : bool [optional][default: False]
            if True, plot the base-10 logarithm of the counts
        savefig : bool [optional][default: False]
            if True, save figure to disk
        
        """
        
        show_values = kwargs.get('show_values', False)
        one_based = kwargs.get('one_based', False)
        log10 = kwargs.get('log10', False)
        savefig = kwargs.get('savefig', False)
        
        if one_based:
            startk = int(1)
            endk = self.n # excluded
        else:
            startk = int(0)
            endk = self.n - 1 # excluded
        
        for k in range(startk, endk):
            fig, ax = self.plot(k,
                                one_based=one_based,
                                log10=log10,
                                vmax=self.common_vmax,
                                show_values=show_values,
                                savefig=savefig)
        
        return
    
    
    def plot_pairs_bar(self, **kwargs):
        """
        Plot a stacked bar chart of local and remote pairs for each transposition k
        
        Parameters
        ----------
        show_values : bool [optional][default: False]
            if True, print the exact value on each bar segment
        show_title : bool [optional][default: True]
            if True, print figure title
        one_based : bool [optional][default: False]
            if True, the first transposition is k=1. Otherwise, it is k=0
        ax : axes [optional][default: None]
            axes handles
        savefig : bool [optional][default: False]
            if True, save figure to disk
        """
        
        show_values = kwargs.get('show_values', False)
        show_title = kwargs.get('show_title', True)
        one_based = kwargs.get('one_based', False)
        do_save = kwargs.get('savefig', False)
        ax = kwargs.get('ax', None)
        
        if one_based:
            k_values = np.arange(1, self.n)
        else:
            k_values = np.arange(0, self.n-1)
        
        k_values_ticks = np.arange(k_values[0], k_values[-1]+1, 2)
        
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=(12, 6))
    
        ax.bar(k_values, self.local_counts,
               label='Local pairs',
               color='steelblue')
        ax.bar(k_values, self.remote_counts, 
               bottom=self.local_counts,
               label='Remote pairs',
               color='tomato')
    
        if show_values:
            for i, k in enumerate(k_values):
                # local value
                ax.text(k, self.local_counts[i] / 2,
                        str(int(self.local_counts[i])),
                        ha='center', va='center', fontsize=7, color='black')
                # remote value
                ax.text(k, self.local_counts[i] + self.remote_counts[i] / 2,
                        str(int(self.remote_counts[i])),
                        ha='center', va='center', fontsize=7, color='black')
    
        ax.set_xlabel('$k$')
        ax.set_ylabel('Off-diagonal pair count')
        if show_title:
            ax.set_title(r'Local and remote pairs per transposition $\tau_{k, k+1}$')
        ax.set_xticks(k_values_ticks)
        ax.legend(loc='lower left')
        
        if do_save:
            fig.savefig(os.path.join(self.folderpath, 'pairs_bar.png'), 
                        format='png',
                        dpi=150)
        
        return fig, ax

