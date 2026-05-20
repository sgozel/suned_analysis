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
        
        # for each file, extract the matrix, and the common max across all files
        self.counts = []
        self.common_vmax = 0
        
        for i, filename in enumerate(self.filenames):
            self.counts.append( np.loadtxt(filename) )
            self.common_vmax = max(self.common_vmax, self.counts[i].max())
        
        self.nprocs = self.counts[0].shape[0]
        
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
        
        if k==0:
            raise ValueError('Adjacent transposition (0, 1) is diagonal. Communication pattern is trivial.')
        index = k - 1
        if k >= self.n-1:
            raise ValueError(f'Adjacent transposition ({k}, {k+1}) is invalid.')
        
        show_values = kwargs.get('show_values', False)
        vmax = kwargs.get('vmax', None)
        ax = kwargs.get('ax', None)
        do_save = kwargs.get('savefig', False)
        
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=(8, 8))
        
        counts = self.counts[index]
        
        if vmax == None:
            vmax = counts.max()
        
        # Custom colormap: green (0) to dark red (max)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "green_to_darkred", ["green", "yellow", "darkred"]
        )
        
        im = ax.imshow(counts, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        
        if show_values:
            for i in range(self.nprocs):
                for j in range(self.nprocs):
                    ax.text(j, i, 
                            str(int(counts[i, j])),
                            ha='center', 
                            va='center', 
                            fontsize=7, 
                            color='black')
        
        # Grid lines to separate cells
        ax.set_xticks(np.arange(-0.5, self.nprocs, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.nprocs, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        
        # Major ticks: rank labels
        ax.set_xticks(np.arange(self.nprocs))
        ax.set_yticks(np.arange(self.nprocs))
        ax.set_xticklabels([f'rank {i}' for i in range(self.nprocs)], rotation=45, ha='right')
        ax.set_yticklabels([f'rank {i}' for i in range(self.nprocs)])
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pair count')
        
        ax.set_title(f'Pair counts: $k = {k}$ (world size: {self.nprocs})')
        
        plt.tight_layout()
        if do_save == True:
            plt.savefig(f'counts_k{k}.png',
                        dpi=150,
                        format='png')
        
        return fig, ax
    
    
    def plot_all(self, **kwargs):
        """
        Plot the communication pattern matrix for all transpositions
        
        Parameters
        ----------
        show_values : bool [optional][default: False]
            write values in matrix cells
        savefig : bool [optional][default: False]
            if True, save figure to disk
        
        """
        
        show_values = kwargs.get('show_values', False)
        savefig = kwargs.get('savefig', False)
        
        for k in range(0, len(self.filenames)):
            fig, ax = self.plot(k+1,
                                vmax=self.common_vmax,
                                show_values=show_values,
                                savefig=savefig)
        
        return

