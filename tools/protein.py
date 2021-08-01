import numpy as np
from typing import List

class ProteinSpectrumData:
    def __init__(
        self,
        nsaf_ppm,
        empai_ppm,
        nsaf_fmol,
        empai_fmol,
        mw_kda
    ):
        self.nsaf_ppm = nsaf_ppm
        self.empai_ppm = empai_ppm
        self.nsaf_fmol = nsaf_fmol
        self.empai_fmol = empai_fmol
        self.mw_kda = mw_kda
    
    def asnp(self):
        return np.array([[
            self.nsaf_ppm, 
            self.empai_ppm,
            self.nsaf_fmol,
            self.empai_fmol,
            self.mw_kda
        ]])