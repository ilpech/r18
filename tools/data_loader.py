import os
import mxnet as mx
import numpy as np
import argparse
from tools import boolean_string, readSearchXlsxReport
from gene_mapping import mapping2dict, uniprot_mapping_header
from typing import List
from gene import Gene

class DataLoader:
    """
    data loader for liver_hepg2 r18 proein abundance experiment
    """
    def _get_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
                            '--rna', type=str,
                            help='path to transcriptom out'
                            )
        parser.add_argument(
                            '--prot1D', type=str,
                            help='path to prot1D out'
                            )
        parser.add_argument(
                            '--prot2D', type=str,
                            help='path to prot2D out'
                            )
        parser.add_argument(
                            '--ionData', type=str, default='',
                            help='path to file with ion data on different compounds'
                            )
        parser.add_argument(
                            '--geneMapping', type=str,
                            help='path to gene ids description'
                            )
        parser.add_argument(
                            '--skipMissing', type=boolean_string, default=True,
                            help='path to gene ids description'
                            )
        return parser.parse_args()
    
    def __init__(self):
        opt = self._get_argparse()
        self.prot1D_path = opt.prot1D
        self.prot2D_path = opt.prot2D
        self.rna_path = opt.rna
        self.ion_path = opt.ionData
        self.gene_mapping_path = opt.geneMapping
        print('reading rna ', self.rna_path)
        self.rna_data = readSearchXlsxReport(
            self.rna_path,
            'Chr18_data'
        )
        print('reading prot1D', self.prot1D_path)
        self.prot1D_data = readSearchXlsxReport(
            self.prot1D_path,
            'Лист1'
        )
        print('reading prot2D', self.prot2D_path)
        self.prot2D_data = readSearchXlsxReport(
            self.prot2D_path,
            'Лист1'
        )
        print('reading mapping', self.gene_mapping_path)
        self.mapping = mapping2dict(self.gene_mapping_path)
        self.gene_ids = [x for x in self.rna_data['Uniprot AC']]
        self.genes: List[Gene] = []
        for gene_id in self.gene_ids:
            gene = Gene(gene_id)
            gene.addRNAReport(self.rna_data)
            gene.addProteinAbundanceReport(
                self.prot1D_data,
                self.prot2D_data
            )
            print('gene ', gene.id_uniprot, ' added')
            self.genes.append(gene)
    
    def info(self):
        for gene in self.genes:
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
            print('GO keywords::')
            print(self.mapping[gene.id()]['GO'])
            print('Ensembl keywords::')
            print(self.mapping[gene.id()]['Ensembl'])

if __name__ == '__main__':
    dataloader = DataLoader()
    dataloader.info()
        