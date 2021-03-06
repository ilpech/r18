import numpy as np
import os
from typing import List
from tools import readSearchXlsxReport
from uniprot_api import getGeneFromApi, sequence

class GenesMapping:
    def __init__(self, readpath=''):
        self.__mapping = {}
        if len(readpath):
            self.read(readpath)
        
    def mapping(self):
        return self.__mapping
    
    def add(self, uniprot_id, ensg_id):
        self.mapping()[uniprot_id] = self.GeneMapping(uniprot_id, ensg_id)
    
    def write(self, outpath):
        if not os.path.isdir(os.path.split(outpath)[0]):
            os.makedirs(os.path.split(outpath)[0])
        with open(outpath, 'w') as f:
            for k,v in self.mapping().items():
                f.write('{} {}\n'.format(k,v.ensg_id))
        print('{} genes mapping written to {}'.format(len(self.mapping()), outpath))
        
    def read(self, inpath):
        with open(inpath, 'r') as f:
            data = f.readlines()
            print('{} mapping lines read from {}'.format(len(data), inpath))
            for d in data:
                d = d.replace('\n', '')
                splt = d.split(' ')
                self.mapping()[splt[0]] = self.GeneMapping(splt[0], splt[1])

    class GeneMapping:
        def __init__(
            self,
            uniprot_id,
            ensg_id
        ):
            self.ensg_id = ensg_id 
            self.uniprot_id = uniprot_id
    

class Gene:
    def __init__(
        self,
        uniprot_id,
        only_w_values=False
    ):
        self.id_uniprot: str = uniprot_id
        # print('created gene: ', self.id_uniprot)
        # init by addProteinAbundanceReport
        self.ratio: float  = None #Ratio (Spep/Slab)
        self.c_fmol_mkg: float = None 
        self.c_fmol_cell: float = None 
        self.c_mol_l: float = None
        self.protein_copies_per_cell_1D: float = None
        self.protein_copies_per_cell_2D: float = None
        self.protein_data = {}
        self.rna_measurements = {}
        self.protein_measurements = {}
        # init by ion data
        self.compound: str = None
        # add other info from file
        # init by rna
        self.gene_name = None
        self.protein_name = None
        self.chromosome: int = None # number
        self.chromosome_pos: str = None # locus
        self.nextprot_status = None
        self.peptide_seq = None
        self.only_w_values = only_w_values
        self.id_ensg = None
        
        self.mapping : GeneMapping = None
        
    def id(self):
        return self.id_uniprot
        
    def addProteinAbundanceReport(
        self,
        prot1D,
        prot2D
    ):
        prot_1d_names = prot1D[' PI (Uniprot)']
        prot_2d_names = prot2D['PI (Uniprot)']
        copies_column = 'Copies of protein per cell'
        prot_1d_id = [i for i in range(len(prot_1d_names)) if prot_1d_names[i] == self.id_uniprot]
        prot_2d_id = [i for i in range(len(prot_1d_names)) if prot_2d_names[i] == self.id_uniprot]
        if not len(prot_1d_id):
            if self.only_w_values:
                raise Exception('gene(only_w_values)::error while trying to find protein1D with id {}'.format(self.id_uniprot))
            self.protein_copies_per_cell_2D = -1.0
        else:
            self.protein_copies_per_cell_1D = float(prot1D[copies_column][prot_1d_id[0]])
        if not len(prot_2d_id):
            if self.only_w_values:
                raise Exception('gene(only_w_values)::error! no protein2D with id {}'.format(self.id_uniprot))
            self.protein_copies_per_cell_1D = -1.0
        else:
            self.protein_copies_per_cell_2D = float(prot2D[copies_column][prot_2d_id[0]])
        self.protein_measurements['protein_copies_per_cell_1D'] = self.protein_copies_per_cell_1D 
        self.protein_measurements['protein_copies_per_cell_2D'] = self.protein_copies_per_cell_2D 
    
    def addRNAReport(self, rna_data):
        gene_names = rna_data['Uniprot AC']
        gene_id = [i for i in range(len(gene_names)) if gene_names[i] == self.id_uniprot]
        if not len(gene_id):
            raise Exception('error while trying to find rna with id {}'.format(self.id_uniprot))
        gene_id = gene_id[0]
        self.gene_name = gene_names[gene_id]
        self.protein_name = rna_data['Protein name'][gene_id]
        self.nextprot_status = rna_data['2017_NextProt status'][gene_id]
        self.peptide_seq = rna_data['Peptide seq'][gene_id]
        self.chromosome = 18
        self.chromosome_pos = rna_data['Chr_Pos'][gene_id]
        search_sustrs = [
            'qPCR',
            'ONT',
            'HepG2',
            'Liv20',
            'Liver'
        ]
        for k,_ in rna_data.items():
            for s in search_sustrs:
                if s in k:
                    m_out = rna_data[k][gene_id]
                    if isinstance(m_out, str):
                        if self.only_w_values:
                            raise Exception('gene(only_w_values)::error! RNA={} with id {}'.format(
                                m_out,
                                self.id_uniprot
                            ))
                        m_out = -1.0
                    self.rna_measurements[k] = m_out
                    break
    
    @staticmethod
    def proteinAminoAcidsAlphabet():
        return sorted([
            'A', 'C', 'D', 'E', 'F', 
            'G', 'H', 'I', 'K', 'L', 
            'M', 'N', 'P', 'Q', 'R', 
            'S', 'T', 'V', 'W', 'Y'
        ])
    
    def apiData(self):
        return getGeneFromApi(self.id())
    
    def apiSequence(self):
        return sequence(self.id())
    
    def apiSeqOneHot(self):
        return Gene.seq2oneHot(self.apiSequence())
    
    @staticmethod
    def seq2oneHot(seq):
        alphabet = Gene.proteinAminoAcidsAlphabet()
        set_seq = set(seq)
        if len(set_seq) > len(alphabet):
            raise Exception('seq2oneHot:: seq len {} > alph len {}'.format(
                len(set_seq), 
                len(alphabet)
            ))
        onehot = [[0] * len(alphabet)] * len(seq)
        onehot = np.array(onehot)
        for i in range(len(seq)):
            pos = [j for j in range(len(alphabet)) if alphabet[j] == seq[i]][0]
            onehot[i][pos] = 1 #alphabet[j]
        return onehot
        

if __name__ == '__main__':
    data_path = '../data/liver_hepg2'
    rna_file_path = 'rna_liver_hepg2_13_20_no_header.xlsx'
    prot_1D = 'prot_1D_analysis.xlsx'
    prot_2D = 'prot_2D_analysis.xlsx'
    rna_data = readSearchXlsxReport(
        os.path.join(data_path, rna_file_path),
        'Chr18_data'
    )
    prot_1D = readSearchXlsxReport(
        os.path.join(data_path, prot_1D),
        '????????1'
    )
    prot_2D = readSearchXlsxReport(
        os.path.join(data_path, prot_2D),
        '????????1'
    )
    print(len(rna_data), len(prot_1D), len(prot_2D))
    uniprot_ac = rna_data['Uniprot AC']
    genes : List[Gene] = []
    for gene_id in uniprot_ac:
        gene = Gene(gene_id)
        gene.addRNAReport(rna_data)
        gene.addProteinAbundanceReport(
            prot_1D,
            prot_2D
        )
        genes.append(gene)
    for gene in genes:
        print('============', gene.id_uniprot)
        print(gene.protein_name)
        print(gene.nextprot_status)
        print(gene.peptide_seq)
        print(gene.chromosome_pos)
        print('rna experiments:: len::', len(gene.rna_measurements))
        for m,v in gene.rna_measurements.items():
            print(m, '::', v)
        print('gene.protein_copies_per_cell_1D', gene.protein_copies_per_cell_1D)
        print('gene.protein_copies_per_cell_2D', gene.protein_copies_per_cell_2D)
    
    print('\n============ OVERALL')
    print('genes', len(genes))
    with_prot1D = len([x for x in genes if x.protein_copies_per_cell_1D is not None])
    with_prot2D = len([x for x in genes if x.protein_copies_per_cell_2D is not None])
    with_prot1D2D = len([x for x in genes if x.protein_copies_per_cell_1D is not None and 
                         x.protein_copies_per_cell_2D is not None])
    print('with_prot1D', with_prot1D)
    print('with_prot2D', with_prot2D)
    print('with_prot1D2D', with_prot1D2D)

            
        