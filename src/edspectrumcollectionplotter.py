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

from edspectrumplotter import EDSpectrumPlotter

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


class EDSpectrumCollectionPlotter:
    """
    Class to plot the data stored in an EDSpectrumCollection instance
    """
    
    def __init__(self, spectrumcoll):
        """
        Constructor
        """
        self.spectrumcoll = spectrumcoll
        
        # generate plotters for each spectrum in the collection
        self.specplotters = []
        for i in range(0, len(self.spectrumcoll.edspectra)):
            self.specplotters.append( EDSpectrumPlotter(self.spectrumcoll.edspectra[i]) )
        return
    
    def plot_lanczos_iterations(self, k=1, **kwargs):
        """
        Plot the Lanczos iterations for the k smallest eigenvalues, for all
        elements matching the given kwargs filters.
        """
        if k < 1:
            raise ValueError('k must be greater or equal than 1.')
        
        for i, data in enumerate(self.spectrumcoll.data):
            # validate kwargs keys
            for key in kwargs:
                if key not in data:
                    raise KeyError(f"Filter key '{key}' not found in data keys: {list(data.keys())}")
            
            # check if all kwargs match
            match = all(
                abs(data[key] - val) < 1.0e-12 if isinstance(val, float) else data[key] == val
                for key, val in kwargs.items()
            )
            
            if match:
                print('='*10)
                print(', '.join(f'{key} = {data[key]}' for key in kwargs))
                print('jobid = ', data['jobid'])
                print('='*10)
                for u in range(k):
                    self.specplotters[i].plot_eigenvalue(k=u)
                    self.specplotters[i].plot_successive_differences(k=u)
        return
    
    def plot_energy_vs_Jp(self, irrep_key):
        """
        Plot spectrum versus Jp coupling for a specified irrep
        """
        
        # extract all Jp values
        Jpvec = []
        for i, data in enumerate(self.spectrumcoll.data):
            if data['irrepkey'] == irrep_key:
                Jpvec.append( data['Jp'] )
        Jpvec = np.array(Jpvec)
        Jpvec = np.sort(np.unique(Jpvec))
        
        # collect and arrange data
        Nk = int(9)
        Nk_eff = int(9)
        
        energies = np.zeros(shape=(Nk, len(Jpvec)), dtype=float)
        
        for i, data in enumerate(self.spectrumcoll.data):
            if (data['irrepkey'] == irrep_key):
                Jpi = data['Jp']
                ind = np.argwhere(abs(Jpvec - Jpi) < 1.0e-12).flatten()[0]                
                nk = len(self.spectrumcoll.edspectra[i].unique_eigvals)                
                Nk_eff = min(nk, Nk_eff)                
                energies[:Nk_eff, ind] = self.spectrumcoll.edspectra[i].unique_eigvals[:Nk_eff]
                
        # plot
        fig0, ax0 = plt.subplots(figsize=(12, 8))        
        for k in range(0, Nk_eff):
            ax0.plot(Jpvec, energies[k],
                     color='k',
                     marker='o')
        
        ax0.set_xlabel('$J_2$')
        ax0.set_ylabel('Energy')
        ax0.set_title(irrep_key)
        ax0.grid(True, linestyle='--', alpha=0.5)
        
        return fig0, ax0
    
    def plot_gaps_vs_Jp(self, irrep_key):
        """
        Plot spectrum gaps versus Jp coupling for a specified irrep
        """
        
        # extract all Jp values
        Jpvec = []
        for i, data in enumerate(self.spectrumcoll.data):
            if data['irrepkey'] == irrep_key:
                Jpvec.append( data['Jp'] )
        Jpvec = np.array(Jpvec)
        Jpvec = np.sort(np.unique(Jpvec))
        
        # collect and arrange data
        Nk = int(8)
        Nk_eff = int(8)
        
        gaps = np.zeros(shape=(Nk, len(Jpvec)), dtype=float)
        
        for i, data in enumerate(self.spectrumcoll.data):
            if (data['irrepkey'] == irrep_key):
                Jpi = data['Jp']
                ind = np.argwhere(abs(Jpvec - Jpi) < 1.0e-12).flatten()[0]                
                nk = len(self.spectrumcoll.edspectra[i].gaps)                
                Nk_eff = min(nk, Nk_eff)                
                gaps[:Nk_eff, ind] = self.spectrumcoll.edspectra[i].gaps[:Nk_eff]
                
        # plot
        fig0, ax0 = plt.subplots(figsize=(12, 8))        
        for k in range(0, Nk_eff):
            ax0.plot(Jpvec, gaps[k],
                     color='k',
                     marker='o')
        
        ax0.set_xlabel('$J_2$')
        ax0.set_ylabel('Gap')
        ax0.set_title(irrep_key)
        ax0.grid(True, linestyle='--', alpha=0.5)
        return fig0, ax0
    
    def plot_gaps_all_irreps_vs_Jp(self):
        """
        Plot energy gaps of all found irreps vs Jp
        """
        
        # extract all irreps and Jp values
        Jpvec_all = []
        irreps = []
        JpByirrep = {}
        irreps_str = {}
        
        for i, data in enumerate(self.spectrumcoll.data):
            Jpi = data['Jp']
            irrepkey = data['irrepkey']
            irrepstr = data['irrep_str']
            Jpvec_all.append( Jpi )
            if not irrepkey in irreps:
                irreps.append( data['irrepkey'] )
                JpByirrep[irrepkey] = [Jpi]
                irreps_str[irrepkey] = irrepstr
            else:
                JpByirrep[irrepkey].append( Jpi )
        
        Jpvec_all = np.sort(np.unique(np.array(Jpvec_all)))
        for irrepi in irreps:
            JpByirrep[irrepi] = np.sort(np.unique(np.array(JpByirrep[irrepi])))
        
        # extract the gaps for each irrep and Jp value
        gaps = {}
        for irrepkey in irreps:
            
            Njp = len(JpByirrep[irrepkey])
            Nk = int(20)
            Nk_eff = int(20)
            
            gaps[irrepkey] = np.zeros(shape=(Nk, Njp), dtype=float)            
            
            for j, Jp in enumerate(JpByirrep[irrepkey]):
                for i, data in enumerate(self.spectrumcoll.data):
                    if ((data['irrepkey']==irrepkey) & (abs(Jp - data['Jp']) < 1.0e-12)):
                        nk = len(self.spectrumcoll.edspectra[i].gaps)
                        Nk_eff = min(nk, Nk_eff)
                        gaptemp = self.spectrumcoll.edspectra[i].gaps[:Nk_eff]
                        '''
                        # remove gap values corresponding to the upper part of the spectrum
                        for u in range(2, Nk_eff):
                            if (gaptemp[u]>10*gaptemp[u-1]):
                                gaptemp = gaptemp[:u]
                                Nk_eff = u
                                break
                        '''
                        gaps[irrepkey][:Nk_eff, j] = gaptemp
            gaps[irrepkey] = gaps[irrepkey][:Nk_eff, :]
        
        # plot
        
        cols = ['k', 'b', 'r', 'g', 'm']
        mks = ['o', 's', 'v', 'D', '^', '<', '>']
        ms = 80
        
        fig0, ax0 = plt.subplots(figsize=(12, 8))
        for i, irrepkey in enumerate(irreps):
            x = JpByirrep[irrepkey]
            y = gaps[irrepkey]
            ax0.scatter(x, y[0, :],
                        s=ms,
                        marker=mks[i],
                        color=cols[i],
                        facecolors='none',
                        label=irreps_str[irrepkey])
            for k in range(1, y.shape[0]):
                ax0.scatter(x, y[k, :],
                            s=ms,
                            marker=mks[i],
                            color=cols[i],
                            facecolors='none')
        ax0.legend()
        ax0.set_xlabel('$J_2$')
        ax0.set_ylabel('Gap')
        #ax0.set_title()
        ax0.grid(True, linestyle='--', alpha=0.5)
        return fig0, ax0
    
    '''
    def plot_tower_of_states_base(self, **kwargs):
        """
        Plot the tower of states
        
        Parameters
        ----------
        figtitle : string
            figure title
        show_slope : bool [optional][default: False]
            plot the slope of the tower of states versus Casimir
        
        Returns
        -------
        fig, ax : figure handle and axes
            output figure
        """
        
        show_slope = kwargs.get('show_slope', False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = iter(cm.rainbow(np.linspace(0, 1, 10)))
        
        casimirs = {}
        casimirs_vec = []
        
        for i, data in enumerate(self.spectrumcoll.data):
            print('Found irrep: ', data['irrep_str'])
            c = next(colors)
            gaps_temp = self.spectrumcoll.edspectra[i].gaps
            lgaps = len(gaps_temp)
            
            csi = data['casimir']
            casimirs_vec.append(csi)
            if csi in casimirs.keys():
                casimirs[csi] += int(1)
            else:
                casimirs[csi] = int(0)
            
            x_temp = (csi + casimirs[csi]*0.05) * np.full(shape=(lgaps,), fill_value=1, dtype=float)
            
            ax.scatter(x_temp, gaps_temp, 
                       marker='o',
                       color=c)
        
        casimirs_vec = np.array(casimirs_vec)
        
        if show_slope:
            
            ind1 = np.argwhere(casimirs_vec > 0).flatten()
            ind1_temp = np.argmin(casimirs_vec[ind1])
            ind1 = ind1[ind1_temp]
            
            c1 = casimirs_vec[ind1]
            gap1 = self.spectrumcoll.edspectra[ind1].gaps[0]
            slope = gap1 / c1
            
            csmax = np.max(casimirs_vec)
            
            xfine = np.linspace(0.0, csmax, 300)
            yfine = slope * xfine
            
            ax.plot(xfine, yfine,
                    linestyle='--',
                    color='k',
                    label='Slope: ' + '{:.2f}'.format(slope))
        
        ax.set_xlabel('Casimir')
        ax.set_ylabel('Gap')
        ax.set_ylim(bottom=-0.1)
        if 'figtitle' in kwargs:
            ax.set_title(kwargs['figtitle'])
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig, ax
    
    def plot_tower_of_states(self, target_Jp, **kwargs):
        """
        Plot the tower of states for a given Jp value
        
        X axis: quadratic Casimir
        Y axis: Energy gaps
        
        Parameters
        ----------
        target_Jp : float
            target value of Jp parameter
        figtitle : string [optional][default: None]
            figure title
        show_slope : bool [optional][default: False]
            plot the slope of the tower of states versus Casimir
        
        Returns
        -------
        fig, ax : figure handle and axes
            output figure
        """
        
        show_slope = kwargs.get('show_slope', False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = iter(cm.rainbow(np.linspace(0, 1, 10)))
        
        inds = []
        casimirs = {}
        casimirs_vec = []
        
        print('='*10)
        print('Jp = ', target_Jp)
        print('='*10)
        
        for i, data in enumerate(self.spectrumcoll.data):
            Jpi = data['Jp']
            if (abs(Jpi - target_Jp) < 1.0e-12):
                print('Found irrep: ', data['irrep_str'])
                
                inds.append(i)
                csi = data['casimir']
                casimirs_vec.append(csi)
                
                c = next(colors)
                gaps_temp = self.spectrumcoll.edspectra[i].gaps
                lgaps = len(gaps_temp)
                
                if csi in casimirs.keys():
                    casimirs[csi] += int(1)
                else:
                    casimirs[csi] = int(0)
                
                x_temp = (csi + casimirs[csi]*0.05) * np.full(shape=(lgaps,), fill_value=1, dtype=float)
                
                ax.scatter(x_temp, gaps_temp, 
                           marker='o',
                           color=c)
        
        inds = np.array(inds)
        casimirs_vec = np.array(casimirs_vec)
        
        if show_slope:
            ind1 = np.argwhere(casimirs_vec > 0).flatten()
            ind1_temp = np.argmin(casimirs_vec[ind1])
            ind1l = ind1[ind1_temp]
            ind1g = inds[ind1][ind1_temp]
            
            c1 = casimirs_vec[ind1l]
            gap1 = self.spectrumcoll.edspectra[ind1g].gaps[0]
            slope = gap1 / c1
            
            csmax = np.max(casimirs_vec)
            
            xfine = np.linspace(0.0, csmax, 300)
            yfine = slope * xfine
            
            ax.plot(xfine, yfine,
                    linestyle='--',
                    color='k',
                    label='Slope: ' + '{:.2f}'.format(slope))
        
        
        ax.set_xlabel('Casimir')
        ax.set_ylabel('Gap')
        if 'figtitle' in kwargs:
            ax.set_title(kwargs['figtitle'])
        ax.set_ylim(bottom=-0.1)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig, ax
    '''
    
    def plot_tower_of_states(self, **kwargs):
        """
        Plot the tower of states for a given Jp value
        
        X axis: quadratic Casimir
        Y axis: Energy gaps
        
        Parameters
        ----------
        selector : dict
            selector of data to plot
        figtitle : string [optional][default: None]
            figure title
        show_slope : bool [optional][default: False]
            plot the slope of the tower of states versus Casimir
        
        Returns
        -------
        fig, ax : figure handle and axes
            output figure
        """
        
        selector = kwargs.get('selector', {})
        show_slope = kwargs.get('show_slope', False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = iter(cm.rainbow(np.linspace(0, 1, 10)))
        
        inds = []
        casimirs = {}
        casimirs_vec = []
        
        print('='*10)
        print(selector)
        print('='*10)
        
        tol = 1.0e-12
        
        for i, data in enumerate(self.spectrumcoll.data):
            
            match = True
            for key in selector.keys():
                kval = data[key]
                if isinstance(kval, float):
                    if abs( kval - selector[key] ) > tol:
                        match = False
                        break
                else:
                    if not kval == selector[key]:
                        match = False
                        break
            
            if match:
                print('Found irrep: ', data['irrep_str'])
                
                inds.append(i)
                csi = data['casimir']
                casimirs_vec.append(csi)
                
                c = next(colors)
                gaps_temp = self.spectrumcoll.edspectra[i].gaps
                lgaps = len(gaps_temp)
                
                if csi in casimirs.keys():
                    casimirs[csi] += int(1)
                else:
                    casimirs[csi] = int(0)
                
                x_temp = (csi + casimirs[csi]*0.05) * np.full(shape=(lgaps,), fill_value=1, dtype=float)
                
                ax.scatter(x_temp, gaps_temp, 
                           marker='o',
                           color=c)
        
        inds = np.array(inds)
        casimirs_vec = np.array(casimirs_vec)
        
        if show_slope:
            ind1 = np.argwhere(casimirs_vec > 0).flatten()
            ind1_temp = np.argmin(casimirs_vec[ind1])
            ind1l = ind1[ind1_temp]
            ind1g = inds[ind1][ind1_temp]
            
            c1 = casimirs_vec[ind1l]
            gap1 = self.spectrumcoll.edspectra[ind1g].gaps[0]
            slope = gap1 / c1
            
            csmax = np.max(casimirs_vec)
            
            xfine = np.linspace(0.0, csmax, 300)
            yfine = slope * xfine
            
            ax.plot(xfine, yfine,
                    linestyle='--',
                    color='k',
                    label='Slope: ' + '{:.2f}'.format(slope))
        
        
        ax.set_xlabel('Casimir')
        ax.set_ylabel('Gap')
        if 'figtitle' in kwargs:
            ax.set_title(kwargs['figtitle'])
        ax.set_ylim(bottom=-0.1)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig, ax
