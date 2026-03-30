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
from pathlib import Path
import numpy as np
import pandas as pd



class OutputLog:
    """
    Class to parse the log output file of an ED calculation
    """
    
    def __init__(self, filepath):
        """
        Constructor
        """
        self.filepath = filepath
        
        self.parse()
        return
    
    def parse(self):
        """
        Parse a SUNED log file containing one or more simulations.
    
        Returns a list of dicts, one per simulation. If only one simulation
        is found, a list of length 1 is still returned for consistency.
        """
        self._read_lines()
        
        self._extract_metadata()
        self.df_sites = self._extract_sites()
        self.df_bonds = self._extract_bonds()
        self.df_mvm_times = self._extract_mvm_time()
        self.energies = self._extract_energies()
        self.correlations = self._extract_correlations()
        
        return
    
    def _read_lines(self):
        """
        Read all lines of the log file
        """
        with open(self.filepath, encoding="utf-8", errors="replace") as f:
            print('Reading log file: ', self.filepath)
            self.lines =  [line.rstrip("\n") for line in f]
        return
    
    @staticmethod
    def extract_float(pattern, line):
        """
        Return the first float captured by <pattern> in <line>, or None.
        """
        m = re.search(pattern, line)
        return float(m.group(1)) if m else None
    
    def _extract_ns(self):
        """
        Extract the number of sites Ns from the header
        """
        pattern = r"^Ns\s*=\s*(\d+)"
        for line in self.lines:
            m = re.match(pattern, line.strip())
            if m:
                return int(m.group(1))
        raise ValueError("Could not find 'Ns' in the log file.")
        return
    
    def _extract_nbonds(self):
        """
        Extract the number of bonds from the line 'Parsed bonds: '
        """
        pattern = r"^Parsed bonds:\s*(\d+)"
        for line in self.lines:
            m = re.match(pattern, line.strip())
            if m:
                return int(m.group(1))
        raise ValueError("Could not find 'Parsed bonds' in the log file.")
        return
    
    def _extract_n(self):
        """
        Extract N of SU(N) from the line 'N = ...'.
        """
        pattern = r"^N\s*=\s*(\d+)"
        for line in self.lines:
            m = re.match(pattern, line.strip())
            if m:
                return int(m.group(1))
        raise ValueError("Could not find 'N' in the log file.")
        return
    
    def _extract_target_irrep(self):
        """
        Extract target irrep as a numpy array.
        The label 'Target irrep:' appears on one line, the value '[a, b, c, ...]'
        on the next line.
        """
        for i, line in enumerate(self.lines):
            if re.search(r"Target irrep:", line):
                next_line = self.lines[i + 1].strip()
                m = re.match(r"^\[([^\]]+)\]", next_line)
                if m:
                    values = [int(x.strip()) for x in m.group(1).split(",")]
                    return np.array(values, dtype=int)
        raise ValueError("Could not find 'Target irrep' in the log file.")
        return
    
    def _extract_dimension(self):
        """
        Extract dimension (first occurrence) from 'dimension = ...'.
        """
        pattern = r"^dimension\s*=\s*(\d+)"
        for line in self.lines:
            m = re.match(pattern, line.strip())
            if m:
                return int(m.group(1))
        raise ValueError("Could not find 'dimension' in the log file.")
        return
    
    def _extract_mpi_world_size(self):
        """
        Extract mpi_world_size_ from the line 'mpi_world_size_ = N'.
        """
        pattern = r"^mpi_world_size_\s*=\s*(\d+)"
        for line in self.lines:
            m = re.match(pattern, line.strip())
            if m:
                return int(m.group(1))
        raise ValueError("Could not find 'mpi_world_size_' in the log file.")
        return
    
    def _extract_job_id(self):
        """
        Extract job ID (last integer before .log or .out) from the filename.
        The job ID is preceded by either '_' or '-'.
        """
        name = Path(self.filepath).name
        m = re.search(r"[_-](\d+)\.(log|out)$", name)
        if m:
            return int(m.group(1))
        raise ValueError("Could not extract job ID from filename: {}".format(name))
        return
    
    def _extract_metadata(self):
        """
        Returns a dict with scalar metadata extracted from the header:
            N         : int
            Ns        : int
            alpha     : numpy array of ints (target irrep)
            JOB_ID    : int (from filename)
            dimension : int (first occurrence)
        """
        self.metadata = {
            "N":              self._extract_n(),
            "Ns":             self._extract_ns(),
            "alpha":          self._extract_target_irrep(),
            "JOB_ID":         self._extract_job_id(),
            "dimension":      self._extract_dimension(),
            "mpi_world_size": self._extract_mpi_world_size(),
        }
        return
    
    def _extract_sites(self):
        """
        Extracts site coordinates from the 'Parsed sites' section.
    
        Example lines:
            Parsed sites: 24
            0 (2.500, 2.598, 0.000)
            1 (5.000, 0.000, 0.000)
    
        Columns:
            X : float
            Y : float
            Z : float
        """
        site_pattern = re.compile(r"^\s*(\d+)\s*\(([\d.+-]+),\s*([\d.+-]+),\s*([\d.+-]+)\)")
    
        rows = []
        for line in self.lines:
            m = site_pattern.match(line)
            if m:
                rows.append({
                    "X": float(m.group(2)),
                    "Y": float(m.group(3)),
                    "Z": float(m.group(4)),
                })
    
        df = pd.DataFrame(rows, columns=["X", "Y", "Z"])
        df.index = range(len(df))
        return df
    
    
    def _extract_bonds(self):
        """
        Extracts bond properties from the 'Parsed bonds' section.
    
        Example line:
            16) [   769] J=1: [3] : (1, 3) = (1, 2)(2, 3)(1, 2)
    
        Columns:
            BOND      : tuple of two ints, e.g. (1, 3)
            COUPLING  : float, the coupling constant value (e.g. J=1 -> 1.0)
            NOPS      : int, value in second square brackets
            DECOMP    : list of tuples of two ints, e.g. [(1,2),(2,3),(1,2)]
        """
        bond_pattern = re.compile(
            r"^\s*(\d+)\)\s*"              # bond id
            r"\[.*?\]\s*"                  # first bracket (ignored)
            r"(\w+)=([^:]+):\s*"           # coupling name and value
            r"\[(\d+)\]\s*:\s*"            # NOPS
            r"\((\d+),\s*(\d+)\)\s*=\s*"   # BOND (x, y)
            r"(.+)$"                       # DECOMP string
        )
        pair_pattern = re.compile(r"\((\d+),\s*(\d+)\)")
        
        self.couplings = {}
        
        rows = []
        for line in self.lines:
            m = bond_pattern.match(line)
            if m:
                couplingName = m.group(2).strip()
                couplingValue = float(m.group(3).strip())
                nops          = int(m.group(4))
                bond          = (int(m.group(5)), int(m.group(6)))
                decomp        = [(int(a), int(b)) for a, b in pair_pattern.findall(m.group(7))]
                
                if couplingName in self.couplings.keys():
                    if abs(couplingValue - self.couplings[couplingName])>1.0e-13:
                        raise ValueError('Problem: two bonds with same coupling name, but different coupling values.')
                else:
                    self.couplings[couplingName] = couplingValue
                
                rows.append({
                    "BOND":          bond,
                    "COUPLINGNAME":  couplingName,
                    "COUPLINGVALUE": couplingValue,
                    "NOPS":          nops,
                    "DECOMP":        decomp,
                })
        df = pd.DataFrame(rows, columns=["BOND", "COUPLINGNAME", "COUPLINGVALUE", "NOPS", "DECOMP"])
        df.index = range(len(df))
        return df
    
    def _extract_mvm_time(self):
        """
        One row per completed Lanczos iteration.
        Matches lines of the form:
            multiply time =   2498.09 s
        """
        pattern = r"^multiply time\s*=\s*([\d.]+)\s*s"
        values = []
        for line in self.lines:
            val = OutputLog.extract_float(pattern, line)
            if val is not None:
                values.append(val)
        return pd.DataFrame({"MVM_TIME": values}, dtype=float)
    
    def get_mean_mvm_time(self):
        """
        Compute mean MVM time across all MVMs
        """
        return self.df_mvm_times['MVM_TIME'].mean()        
    
    def get_std_mvm_time(self):
        """
        Compute standard deviation on MVM time across all MVMs
        """
        return self.df_mvm_times['MVM_TIME'].std()
    
    def _extract_energies(self):
        """
        Extract energy eigenvalues
        """
        e_pattern = re.compile(r"Eigenvalue\[(\d+)\]\s*=\s*([+-]?\d+\.\d+)")
        
        values = {}
        for line in self.lines:
            m = e_pattern.match(line.strip())
            if m:
                values[int(m.group(1))] = float(m.group(2))
        
        if not values:
            self.has_energy = False
            return None
        
        max_idx = max(values)
        arr = np.zeros(max_idx + 1)
        for idx, val in values.items():
            arr[idx] = val
        self.has_energy = True
        return arr
    
    def _extract_correlations(self):
        """
        Extracts spin correlations C[i] if present in the file.
        Returns a numpy array of floats indexed 0..Ns-1, or None if not found.
        """
        c_pattern = re.compile(r"^C\[(\d+)\]\s*=\s*([+-]?[\d.]+)")
    
        values = {}
        for line in self.lines:
            m = c_pattern.match(line.strip())
            if m:
                values[int(m.group(1))] = float(m.group(2))
    
        if not values:
            self.has_correlations = False
            return None
    
        max_idx = max(values)
        arr = np.zeros(max_idx + 1)
        for idx, val in values.items():
            arr[idx] = val
        self.has_correlations = True
        return arr



class SimulationSplitter:
    """
    Split a single .log/.out output file in as many actual simulations it contains
    """
    
    def __init__(self, filepath):
        """
        Constructor
        """
        self.filepath = filepath
        
        with open(self.filepath, encoding="utf-8", errors="replace") as f:
            self.alllines =  [line.rstrip("\n") for line in f]
        
        chunks = self._split()
        
        if (len(chunks) > 1):    
            rootname, jobid, ext = self._extract_parts()    
            for i, chunk in enumerate(chunks):
                print('-'*20)
                print('Chunk ', i)
                print('-'*20)
                filename = rootname + f'_chunk{i}_' + str(jobid) + ext
                print('Write in:', filename)
                with open(filename, 'w') as f:
                    f.writelines(line + '\n' for line in chunk)
        return
    
    def _split(self):
        """
        Splits a log file containing multiple concatenated simulations into
        a list of line-lists, one per simulation.
    
        A new simulation is detected when the 4-line header sequence appears:
            N = ...
            Ns = ...
            Target irrep:
            [...]
        """
        header_patterns = [
            re.compile(r"^\s*N\s*=\s*\d+"),
            re.compile(r"^\s*Ns\s*=\s*\d+"),
            re.compile(r"^\s*Target irrep:"),
            re.compile(r"^\s*\["),
        ]
        n = len(header_patterns)
    
        # Find all line indices where a 4-line header sequence starts
        start_indices = []
        for i in range(len(self.alllines) - n + 1):
            if all(header_patterns[j].match(self.alllines[i + j]) for j in range(n)):
                start_indices.append(i)
    
        if not start_indices:
            return [self.alllines]  # single simulation, no splitting needed
        
        # lines before the first simulation, to be replicated in each output instance
        execution_pattern = re.compile(r"^\s*Execution\s*-+\s*:")
        execution_idx = next(
            (i for i, line in enumerate(self.alllines) if execution_pattern.match(line)),
            start_indices[0]  # fallback
        )
        
        header = self.alllines[:execution_idx + 2]  # +1 for the line after, +1 for slice exclusion
        
        # Slice lines between consecutive start indices
        chunks = []
        
        for k, start in enumerate(start_indices):
            end = start_indices[k + 1] if k + 1 < len(start_indices) else len(self.alllines)
            chunks.append(header + self.alllines[start:end])
    
        return chunks
    
    def _extract_parts(self):
        """
        Split filepath into (rootname, jobid, extension)
        """
        name = Path(self.filepath).stem # name without the .extension
        ext  = Path(self.filepath).suffix # .log or .out or ...
        m = re.search(r"^(.*)[_-](\d+)$", name)
        if not m:
            raise ValueError(f"Could not extract job ID from filename: {self.filepath}")
        
        rootname = m.group(1)
        jobid = int(m.group(2))
        
        return rootname, jobid, ext
    
