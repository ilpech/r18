import numpy as np

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
        

# add time of report
class SearchGUIProtein:    
    def __init__(
        self,
        protein_group,
        confidence,
        chromosome,
        description,
        sec_description,
        secondary_accessions,
        peptides_n,
        uniq_peptides_n,
        confident_coverage,
        nsaf_ppm,
        empai_ppm,
        nsaf_fmol,
        empai_fmol,
        mw_kda
    ):
        self.protein_group = protein_group
        self.description = description
        self.conf = confidence
        self.chromosome = chromosome
        self.description = description
        self.sec_description = sec_description
        self.secondary_accessions = secondary_accessions
        self.peptides_n = peptides_n
        self.uniq_peptides_n = uniq_peptides_n
        self.confident_coverage = confident_coverage
        self.nsaf_ppm = nsaf_ppm
        self.empai_ppm = empai_ppm
        self.nsaf_fmol = nsaf_fmol
        self.empai_fmol = empai_fmol
        self.mw_kda = mw_kda
        self.spectrum_data = ProteinSpectrumData(
            nsaf_ppm,
            empai_ppm,
            nsaf_fmol,
            empai_fmol,
            mw_kda
        )