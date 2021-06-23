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
        uniq_peptides_n
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