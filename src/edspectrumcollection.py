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

from edspectrum import EDSpectrum
from outputlog import OutputLog


class EDSpectrumCollection:
    """
    Class to represent an ensemble of ED spectra
    """
        
    def __init__(self, data):
        """
        Constructor
        
        Parameters
        ----------
        data : list of dicts
            descriptive data set
        """
        self.data = data
        self.edspectra = []
        self.outputlogs = []
        for i in range(0, len(self.data)):
            print('='*20)
            self.edspectra.append(EDSpectrum(data[i]['folderpath']))
            self.outputlogs.append(OutputLog(data[i]['outputfile']))
        return
    
    def _collect_Jpvec(self):
        """
        Find the unique Jp values
        """
        
        # extract all Jp values
        Jpvec = []
        for i, data in enumerate(self.data):
            Jpvec.append( data['Jp'] )
        Jpvec = np.array(Jpvec)
        Jpvec = np.sort(np.unique(Jpvec))
        return Jpvec
    
    def compute_gaps_across_Jp(self):
        """
        Search for minimum eigenvalue across all irreps
        """
        
        Jpvec = self._collect_Jpvec()
        
        # search minimum value of energy, for each Jp, across all irreps
        Emin = np.zeros(shape=(len(Jpvec),), dtype=float)
        
        for j in range(0, len(Jpvec)):
            Jpj = Jpvec[j]
            print('--------------------')
            print(f'Jp = {Jpj:.2f}')
            print('--------------------')
            # collect all lowest energies for Jp = Jpvec[j] accross all irreps
            temp_irrep = []
            temp_emin = []
            
            for i, data in enumerate(self.data):
                if (abs(data['Jp'] - Jpj) < 1.0e-12):
                    temp_irrep.append( data['irrep_str'] )
                    temp_emin.append( self.edspectra[i].unique_eigvals[0] )
            
            temp_emin = np.array(temp_emin)
        
            # search min across all irreps
            ind = np.argmin(temp_emin)
            emin = temp_emin[ind]
            Emin[j] = emin
            
            print(f'For Jp = {Jpj:.2f}, the minimal eigenvalue is in irrep: {temp_irrep[ind]}')
            print(f'The min value is: {Emin[j]}')
        
        # Compute gaps
        for i, data in enumerate(self.data):
            Jpi = data['Jp']
            ind = np.argwhere(abs(Jpvec - Jpi) < 1.0e-12).flatten()[0]
            self.edspectra[i].compute_gaps(Emin[ind])
        
        return
    
