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



class RuntimeLog:
    """
    Class to load and plot the runtime of ajacent transpositions and bonds
    """
    
    def __init__(self, folderpath):
        """
        Constructor
        """
        if not os.path.isdir(folderpath):
            raise FileNotFoundError(f'Folder {folderpath} does not exist.')
        self.folderpath = folderpath
        
        self.filepath_transpos = os.path.join(self.folderpath, 'runtime_transpos.log')
        
        # parse ragged file: each line has a variable number of whitespace-separated floats
        self.transpos_runtimes = []
        with open(self.filepath_transpos, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = [float(x) for x in line.split()]
                self.transpos_runtimes.append(np.array(values))

        if not self.transpos_runtimes:
            raise ValueError(f'No data found in {self.filepath_transpos}')
        
        self.ntranspos = len(self.transpos_runtimes)
        self.k_values = np.arange(0, self.ntranspos)
        self.transpos_means = [np.mean(r)  for r in self.transpos_runtimes]
        self.transpos_stds  = [np.std(r, ddof=1) if len(r) > 1 else 0.0 for r in self.transpos_runtimes]
        self.transpos_means = np.array(self.transpos_means)
        self.transpos_stds = np.array(self.transpos_stds)
        
        # read bonds runtimes
        self.filepath_bonds = os.path.join(self.folderpath, 'runtime_bonds.log')
        
        tmp = np.loadtxt(self.filepath_bonds)
        self.bonds_runtime = {}
        self.nbonds = tmp.shape[0]
        self.bonds_means = []
        self.bonds_stds = []
        for i in range(0, self.nbonds):
            self.bonds_runtime[i] = tmp[i, :]
            self.bonds_means.append( np.mean(tmp[i, :]) )
            self.bonds_stds.append( np.std(tmp[i, :], ddof=1) )
        self.bonds_means = np.array(self.bonds_means)
        self.bonds_stds = np.array(self.bonds_stds)
        
        return
    
    
    def plot_transpos_runtime(self, **kwargs):
        """
        Plot transposition runtime

        Parameters
        ----------
        show_title : bool [optional][default: True]
            if True, print figure title
        ax : axes [optional][default: None]
            axes handles
        do_save : bool [optional][default: False]
            if True, save figure to disk

        Returns
        -------
        fig, ax : figure handles

        """
        
        show_title = kwargs.get('show_title', True)
        do_save = kwargs.get('savefig', False)
        ax = kwargs.get('ax', None)
        
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=(12, 6))

        ax.errorbar(self.k_values, self.transpos_means, 
                    yerr=self.transpos_stds,
                    fmt='o',
                    color='black',
                    ecolor='black',
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    linestyle='none',
                    linewidth=1.5,
                    markersize=5,
                    zorder=10,
                    label='Mean runtime')
        
        ax.set_xlabel('$k$')
        ax.set_ylabel('Runtime (s)')
        if show_title:
            ax.set_title('Runtime per transposition $(k, k+1)$')
        ax.set_xticks(self.k_values)
        ax.legend(loc='upper left')
        
        if do_save:
            fig.savefig(os.path.join(os.path.join(self.folderpath), 'runtime_transpos.png'),
                        format='png',
                        dpi=150)
        
        return fig, ax
    
    
    def plot_bonds_runtime(self, **kwargs):
        """
        Plot bonds runtime

        Parameters
        ----------
        show_title : bool [optional][default: True]
            if True, print figure title
        ax : axes [optional][default: None]
            axes handles
        do_save : bool [optional][default: False]
            if True, save figure to disk
        color : string
            marker color
        
        Returns
        -------
        fig, ax : figure handles

        """
        
        show_title = kwargs.get('show_title', True)
        do_save = kwargs.get('savefig', False)
        ax = kwargs.get('ax', None)
        col = kwargs.get('col', 'k')
        
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=(12, 6))
        
        ax.errorbar(np.arange(0, self.nbonds), 
                    self.bonds_means, 
                    yerr=self.bonds_stds,
                    fmt='o',
                    color=col,
                    ecolor=col,
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    linestyle='none',
                    linewidth=1.5,
                    markersize=5,
                    zorder=10,
                    label='Mean runtime')
        
        ax.set_xlabel('Bond')
        ax.set_ylabel('Runtime (s)')
        if show_title:
            ax.set_title('Runtime per bond')
        ax.legend(loc='upper left')
        
        if do_save:
            fig.savefig(os.path.join(os.path.join(self.folderpath), 'runtime_bonds.png'),
                        format='png',
                        dpi=150)
        
        return fig, ax
    
    
    def plot_bonds_runtime_v2(self, outputlog, **kwargs):
        """
        Plot the bond runtime versus number of adjacent transposition in each 
        bond
        
        Parameters
        ----------
        outputlog : OutputLog
            outputlog object which contains the bonds information
        color : string
            marker color
        """
        
        show_title = kwargs.get('show_title', True)
        do_save = kwargs.get('savefig', False)
        ax = kwargs.get('ax', None)
        col = kwargs.get('col', 'k')
        
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=(12, 6))
        
        xvals = outputlog.df_bonds.NOPS.to_numpy()
        
        '''
        ax.errorbar(xvals, 
                    self.bonds_means, 
                    yerr=self.bonds_stds,
                    fmt='o',
                    color='black',
                    ecolor='black',
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    linestyle='none',
                    linewidth=1.5,
                    markersize=5,
                    zorder=10,
                    label='Mean runtime')
        '''
        ax.scatter(xvals,
                   self.bonds_means,
                   marker='o',
                   facecolors='none',
                   edgecolors=col,
                   color=col,
                   s=100,
                   label='_'
                   )
        
        ax.set_xlabel('Number of adjacent transpositions')
        ax.set_ylabel('Runtime (s)')
        if show_title:
            ax.set_title('Runtime per bond versus number of adjacent transpositions in bond')
        
        if do_save:
            fig.savefig(os.path.join(os.path.join(self.folderpath), 'runtime_bonds_v2.png'),
                        format='png',
                        dpi=150)
        return fig, ax
    
    
    def plot_bonds_runtime_v3(self, outputlog, **kwargs):
        """
        Plot the bond runtime versus number of adjacent transposition in each 
        bond
        
        Parameters
        ----------
        outputlog : OutputLog
            outputlog object which contains the bonds information
        color : string
            marker color
        """
        
        show_title = kwargs.get('show_title', True)
        do_save = kwargs.get('savefig', False)
        ax = kwargs.get('ax', None)
        col = kwargs.get('col', 'k')
        
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=(12, 6))
        
        xvals = np.arange(0, self.nbonds)
        yvals = outputlog.df_bonds.NOPS.to_numpy()
        
        maxbondmean = np.max(self.bonds_means)
        
        ax.scatter(xvals,
                   yvals,
                   s=self.bonds_means / maxbondmean * 400.0,
                   marker='o',
                   facecolors='none',
                   edgecolors=col,
                   color=col,
                   label='_'
                   )
        
        ax.set_xlabel('Bond')
        ax.set_ylabel('Number of adjacent transpositions')
        if show_title:
            ax.set_title('Runtime per bond versus number of adjacent transpositions in bond')
        
        if do_save:
            fig.savefig(os.path.join(os.path.join(self.folderpath), 'runtime_bonds_v2.png'),
                        format='png',
                        dpi=150)
        return fig, ax
    

