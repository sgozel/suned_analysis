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


class OutputLogPlotter:
    """
    Class to plot the data stored in an OutputLog instance
    """
    
    def __init__(self, outputlog):
        """
        Constructor
        """
        self.outputlog = outputlog
        return
    
    def plot_lattice(self, **kwargs):
        """
        Plot the lattice sites
        
        Parameters
        ----------
        simu_torus : dict [optional]
            simulation torus, expressed in units of the Bravais lattice vectors. 
            Must contain 2 keys, 't1' and 't2'
        bravais_vecs : dict [optional]
            dict with the two bravais lattice vectors (keys 'a1' and 'a2')
        show_numbering : bool [optional][default: False]
            if True, show the site indices
        
        """
        
        show_numbering = kwargs.get('show_numbering', False)
        
        minX = self.outputlog.df_sites['X'].min()
        maxX = self.outputlog.df_sites['X'].max()
        minY = self.outputlog.df_sites['Y'].min()
        maxY = self.outputlog.df_sites['Y'].max()
        
        fig0, ax0 = plt.subplots(1, 1, figsize=(7, 8), constrained_layout=True)
        if 'simu_torus' in kwargs:
            if not 'bravais_vecs' in kwargs:
                raise RuntimeError('Missing Bravais lattice vectors for plotting the simulation torus.')
            simu_torus = kwargs['simu_torus']
            bravais_vecs = kwargs['bravais_vecs']
            t1 = simu_torus['t1'][0] * bravais_vecs['a1'] + simu_torus['t1'][1] * bravais_vecs['a2']
            t2 = simu_torus['t2'][0] * bravais_vecs['a1'] + simu_torus['t2'][1] * bravais_vecs['a2']
            origin = np.array([0.0, 0.0])
            minX = min(minX, t1[0])
            minX = min(minX, t2[0])
            maxX = max(maxX, t1[0])
            maxX = max(maxX, t2[0])
            minY = min(minY, t1[1])
            minY = min(minY, t2[1])
            maxY = max(maxY, t1[1])
            maxY = max(maxY, t2[1])
            ax0.quiver(*origin, *t1, angles='xy', scale_units='xy', scale=1, color='r')
            ax0.quiver(*origin, *t2, angles='xy', scale_units='xy', scale=1, color='r')
        # plot lattice points
        ax0.scatter(self.outputlog.df_sites['X'], self.outputlog.df_sites['Y'],
                    marker='o', 
                    s=32,
                    color='k')
        if show_numbering==True:
            for index, row in self.outputlog.df_sites.iterrows():
                x = row['X']
                y = row['Y']
                ax0.annotate(str(index), xy=(x, y),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=10)
        ax0.set_xlabel('$x$')
        ax0.set_ylabel('$y$')
        ax0.set_xlim(left=minX-0.2, right=maxX+0.2)
        ax0.set_ylim(bottom=minY-0.2, top=maxY+0.2)
        ax0.grid(True, linestyle='--', alpha=0.5)
        ax0.set_aspect('equal')
        return fig0, ax0
    
    def plot_correlations(self):
        """
        Plot the lattice sites with the correlation data
        """
        if self.outputlog.has_correlations==True:
            fig0, ax0 = plt.subplots(1, 1, figsize=(7, 8), constrained_layout=True)
            # plot lattice points
            ax0.scatter(self.outputlog.df_sites['X'], self.outputlog.df_sites['Y'],
                        marker='o', 
                        s=32,
                        color='k')
            ax0.set_xlabel('$x$')
            ax0.set_ylabel('$y$')
            
            # plot correlations
            sorted_C = np.sort(self.outputlog.correlations)
            max_C = sorted_C[-2]
            suN = self.outputlog.metadata['N']
            for i, C in enumerate(self.outputlog.correlations):
                C -= 1/suN
                x_i = self.outputlog.df_sites.iloc[i]['X']
                y_i = self.outputlog.df_sites.iloc[i]['Y']
                if C>0:
                    col_i = 'b'
                else:
                    col_i = 'r'
                if (abs(C + 1/suN - 1) > 1.0e-12):
                    ax0.scatter(
                            x_i, y_i,
                            marker='o',
                            s=abs(C)/max_C*200,
                            color=col_i,
                            facecolor=col_i,
                            edgecolors='face'
                    )
            
            titstr = '$'
            for couplingName in self.outputlog.couplings.keys():
                titstr += couplingName + ' = ' + '{:.2f}'.format(self.outputlog.couplings[couplingName]) + '$, $'
            titstr = titstr[:-3]
            
            ax0.set_title(titstr)
            ax0.grid(True, linestyle='--', alpha=0.5)
            ax0.set_aspect('equal')
        else:
            raise ValueError('Missing correlations data.')
        return fig0, ax0
    
    def plot_mvm_time(self):
        """
        Plot MVM time as a scatter plot
        
        data is in self.outputlog.df_mvm_times as a Pandas dataframe with a 
        single column MVM_TIME
        """
        df = self.outputlog.df_mvm_times
        
        fig0, ax0 = plt.subplots(1, 1, figsize=(7, 8), constrained_layout=True)
        ax0.scatter(df.index, df['MVM_TIME'], 
                    color='b', 
                    marker='o',
                    label='_nolegend_')
        
        if len(df) > 2:
            mean = self.outputlog.get_mean_mvm_time()
            std = self.outputlog.get_std_mvm_time()
            ax0.axhline(mean, color='r', linestyle='--', label=f'Mean: {mean:.3f}s')
            ax0.axhspan(mean - std, mean + std, color='r', alpha=0.15, label=f'Std: {std:.3f}s')
            ax0.set_ylim(bottom=mean-2*std, top=mean+2*std)
            ax0.legend(loc='upper right')
        
        ax0.set_xlabel('Lanczos iteration')
        ax0.set_ylabel('MVM time (s)')
        ax0.grid(True, linestyle='--', alpha=0.5)
        
        return fig0, ax0
