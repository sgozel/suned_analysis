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

import os
import sys
import re
import numpy as np



class EDSpectrum:
    """
    Class to store ED spectrum data of a single calculation
    """
    
    def __init__(self, folderpath: str):
        """
        Constructor
        """
        self.folderpath = folderpath
        
        # This will store the eigenvalues written in eigval_k.log files
        self.eigvalkbyit = {} # dict: k -> {iterate: eigenvalue}
        
        # This will store the eigenvalues written in the file alleigvals.log
        self.all_eigvalsbyit = {} # dict: iterate --> np.array(eigenvalues)
        self.all_eigvalsbyk = {}  # dict: k --> np.array(eigenvalues)
        self.all_iterates = {}    # dict: k --> np.array(iterates)
        
        # This will store the actual number of Lanczos iterations
        self.num_iterates = int(1e4)
        
        # This will store the unique, converged eigenvalues at the last Lanczos iteration
        self._load_all_eigvalk()
        self._load_logalleivals()
        
        # check consistency between data in eigval_k.log and alleigvals.log
        self._check_two_sets()
        
        # extract unique, converged eigenvalues
        self.unique_eigvals = self.get_unique_eigvals()
        
        return
    
    def _check_two_sets(self):
        """
        Check consistency of two set of eigenvalues:
            - obtained from alleigvals.log
            - obtained from all files eigval_k.log
        """
        print('Check two sets ...')        
        if self.has_full_spectrum:
            
            for k_key in self.eigvalkbyit.keys():
                # k_key is an eigenvalue index
                for it_key in self.eigvalkbyit[k_key].keys():
                    # it_key is a Lanczos iteration index                
                    if (abs(self.eigvalkbyit[k_key][it_key] - self.all_eigvalsbyit[it_key][k_key]) > 1.0e-15):
                        print('Problem A: missmatch between eigenvalues extracted in two ways')
                        print(f'From eigval_{k_key}.log : at iterate {it_key} : value = {self.eigvalkbyit[k_key[it_key]]}')
                        print(f'From alleigvals.log : at iterate {it_key} : value = {self.all_eigvalsbyit[it_key][k_key]}')
                        sys.exit()
            
            # now do the same, but with all_eigvalsbyk to verify that "transposition"
            # was done correctly
            for k_key in self.eigvalkbyit.keys():
                # k_key is an eigenvalue index
                for it_key in self.eigvalkbyit[k_key].keys():
                    # it_key is a Lanczos iteration index
                    
                    ind = np.argwhere(self.all_iterates[k_key] == it_key).flatten()
                    
                    if (len(ind)>0):
                        ind = ind[0]
                        if (abs(self.eigvalkbyit[k_key][it_key] - self.all_eigvalsbyk[k_key][ind]) > 1.0e-15):
                            print('Problem B: missmatch between eigenvalues extracted in two ways')
                            print(f'From eigval_{k_key}.log : at iterate {it_key} : value = {self.eigvalkbyit[k_key[it_key]]}')
                            print(f'From alleigvals.log (transposed) : at iterate {it_key} : value = {self.all_eigvalsbyk[k_key][ind]}')
                            sys.exit()
        print('Check passed.')
        return
    
    
    def _load_eigvalk(self, filepath: str, k: int):
        """
        Parse a single eigval_k.log file.
        """
        data = {}
        with open(filepath, 'r') as f:
            print('Reading eigenvalue file: ', filepath)
            for line in f:
                line = line.strip()
                match = re.match(r'^(\d+):\s+([-+]?\d+\.\d+(?:[eEdD][+-]?\d+)?)$', line)
                if match:
                    iterate = int(match.group(1))
                    eigval  = float(match.group(2))
                    data[iterate] = eigval
        self.eigvalkbyit[k] = data
        self.num_iterates = min(self.num_iterates, max(data.keys()))
        return
    
    
    def _load_all_eigvalk(self):
        """
        Scan folderpath for all eigval_k.log files and load them.
        """
        pattern = re.compile(r'^eigval_(\d+)\.log$')
        found = []
        for filename in os.listdir(self.folderpath):
            m = pattern.match(filename)
            if m:
                k = int(m.group(1))
                found.append((k, filename))

        if not found:
            raise FileNotFoundError(
                f"No eigval_k.log files found in '{self.folderpath}'"
            )

        found.sort(key=lambda x: x[0])
        for k, filename in found:
            filepath = os.path.join(self.folderpath, filename)
            self._load_eigvalk(filepath, k)

        print(f"Loaded {len(self.eigvalkbyit)} eigenvalue file(s): "
              f"indices {sorted(self.eigvalkbyit.keys())}")
        
        self.num_eigvalk = len(self.eigvalkbyit)
        return
    
    
    def _load_logalleivals(self):
        """
        Load alleigvals.log
        """
        filename = 'alleigvals.log'
        filepath = os.path.join(self.folderpath, filename)
        
        float_pat = r'[-+]?\d+\.\d+(?:[eEdD][+-]?\d+)?'
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                print('Reading full specrum from: ', filepath)
                for line in f:
                    line = line.strip()
                    match = re.match(rf'^(\d+):\s+({float_pat}(?:\s+{float_pat})*)$', line)
                    if match:
                        iterate = int(match.group(1))
                        eigvals = [float(v) for v in match.group(2).split()]
                        self.all_eigvalsbyit[iterate] = np.array(eigvals)
            self.num_iterates = max(self.all_eigvalsbyit.keys())
            self.has_full_spectrum = True
            self._compute_transposed()
        else:
            self.has_full_spectrum = False
        
        return
    
    
    def _compute_transposed(self):
        """
        Build all_eigvalsbyk and all_iterates from all_eigvalsbyit.
        
        Recall:
            all_eigvalsbyit : dict iterate --> np.array(eigenvalues)
        
        Goal:
            all_eigvalsbyk : dict k --> np.array(eigenvalues)
            where k is an eigenvalue index
        
        all_eigvalsbyk[k] = numpy array of k-th eigenvalue iterates
        all_iterates[k]    = numpy array of k-th eigenvalue Lanczos iterate indices
        """
    
        for iterate, eigvals in self.all_eigvalsbyit.items():
            for k, val in enumerate(eigvals):
                if k not in self.all_eigvalsbyk:
                    self.all_eigvalsbyk[k] = []
                    self.all_iterates[k] = []
                self.all_eigvalsbyk[k].append(val)
                self.all_iterates[k].append(iterate)
        
        for k in self.all_eigvalsbyk.keys():
            self.all_eigvalsbyk[k] = np.array(self.all_eigvalsbyk[k])
            self.all_iterates[k] = np.array(self.all_iterates[k])
        
        return
    
    def compute_gaps(self, E0):
        """
        Compute gaps, based on GS value E0
        """
        self.gaps = self.unique_eigvals - E0
        return
        
    
    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def iterates(self, k: int) -> np.ndarray:
        """Return array of iterate indices for eigenvalue k."""
        return np.array(sorted(self.eigvalkbyit[k].keys()))

    def values(self, k: int) -> np.ndarray:
        """Return eigenvalue array for eigenvalue k, sorted by iterate."""
        its = self.iterates(k)
        return np.array([self.eigvalkbyit[k][i] for i in its])
    
    # ------------------------------------------------------------------
    # Extraction of eigenvalues at a given iterate
    # ------------------------------------------------------------------
    
    def extract_at_iterate(self, **kwargs):
        """
        Extract individual eigenvalues at a given iteration point
        
        Eigenvalues must be separated by at least <tol> to be considered different
        
        This is basically a ghost-removal
        
        Parameters
        ----------
        it : int [optional][default: max]
            Lanczos iteration where to extract unique eigenvalues
        tol : float [optional][default: 1.0e-12]
            tolerance to separate 2 different eigenvalues
        """
        
        it = kwargs.get('it', self.num_iterates)
        tol = kwargs.get('tol', 1.0e-12)
        
        # convergence criteria : the same eigenvalue must be constant across
        # Lanczos iterations (to discard an eigenvalue which is moving down
        # as a ghost to its lower neighbor)
        tolc = 1.0e-13
        
        eigvals = []
        
        for k in self.eigvalkbyit.keys():
            its = self.iterates(k)
            values = self.values(k)
            
            ind = np.argwhere(its == it).flatten()[0]
            val = values[ind]
            
            # first check that this eigenvalue is stable
            if (abs(val - values[ind-1]) < tolc):
                if len(eigvals) == 0:
                    eigvals.append(val)
                else:
                    if (abs(val - eigvals[-1]) > tol):
                        eigvals.append(val)
            
        eigvals = np.array(eigvals)
        return eigvals        
    
    
    def get_unique_eigvals(self, **kwargs):
        """
        Extract the unique (ghost removed) eigenvalues at a specified Lanczos
        iteration step
        
        Parameters
        ----------
        it : int [optional][default: max]
            Lanczos iteration step where to extract the eigenvalues
        tol : float [optional][default: 1.0e-12]
            tolerance criteria to differentiate 2 eigenvalues
        """
        
        it = kwargs.get('it', self.num_iterates)
        tol = kwargs.get('tol', 1.0e-11)
        
        # convergence criteria : the same eigenvalue must be constant across
        # Lanczos iterations (to discard an eigenvalue which is moving down
        # as a ghost to its lower neighbor)
        tolc = 1.0e-11
        
        eigvals = []
        
        for k in self.all_eigvalsbyk.keys():
            its = self.all_iterates[k]
            values = self.all_eigvalsbyk[k]
            
            ind = np.argwhere(its == it).flatten()
            
            if len(ind) == 1:
                ind = ind[0]
                val = values[ind]
                if (abs(val - values[ind-1]) < tolc):
                    if len(eigvals) == 0:
                        eigvals.append(val)
                    else:
                        if (abs(val - eigvals[-1]) > tol):
                            eigvals.append(val)
        eigvals = np.array(eigvals)
        return eigvals
