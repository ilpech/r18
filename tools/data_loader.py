# python3 data_loader.py --rna ../data/liver_hepg2/rna_liver_hepg2_13_20_no_header.xlsx --prot1D ../data/liver_hepg2/prot_1D_analysis.xlsx --prot2D ../data/liver_hepg2/prot_2D_analysis.xlsx --geneMapping ../data/liver_hepg2/human_18chr_gene_mapping.tab --ionData ../data/liver_hepg2/prot_ion_data.xlsx

import os
import mxnet as mx
import numpy as np
import argparse
from tools import str2bool, readSearchXlsxReport
from gene_mapping import mapping2dict, uniprot_mapping_header
from uniprot_api import getGeneFromApi, sequence
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
                            '--skipMissing', type=str2bool, default=True,
                            help='skip genes with nd'
                            )
        return parser.parse_args()
    
    @staticmethod
    def opt_from_config(config_path):
        import yaml
        with open(config_path) as c:
            return yaml.load(c)['data']
    
    def __init__(self, config_path):
        self.config_path = config_path
        if len(self.config_path) == 0:
            opt = self._get_argparse()
            self.prot1D_path = opt.prot1D
            self.prot2D_path = opt.prot2D
            self.rna_path = opt.rna
            self.ion_path = opt.ionData
            self.gene_mapping_path = opt.geneMapping
        else:
            opt = DataLoader.opt_from_config(self.config_path)
            self.prot1D_path = opt['prot1D']
            self.prot2D_path = opt['prot2D']
            self.rna_path = opt['rna']
            self.ion_path = opt['ionData']
            self.gene_mapping_path = opt['geneMapping']
        self.genes_mapping_databases = uniprot_mapping_header()
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
        self.gene_ids = [x for x in self.rna_data['Uniprot AC'] if isinstance(x, str)]
        self.genes: List[Gene] = []
        print('creating genes')
        for gene_id in self.gene_ids:
            gene = Gene(gene_id)
            gene.addRNAReport(self.rna_data)
            gene.addProteinAbundanceReport(
                self.prot1D_data,
                self.prot2D_data
            )
            self.genes.append(gene)
        print('{} genes created'.format(len(self.genes)))
    
    def dataFromMappingDatabase(self, db_name, gene_name):
        '''
        db_name should exitst in self.genes_mapping_databases
        '''
        return self.mapping[gene_name][db_name]
    
    def geneMappingDatabases(self, gene_name):
        return self.geneMappingDatabases[gene_name]
    
    def mappingDatabaseToOneHot(self, db_name):
        uniq_data = []
        genes_data = []
        for i in range(len(self.genes)):
            db_gene_data = self.dataFromMappingDatabase(db_name, self.genes[i].id_uniprot)
            genes_data.append(db_gene_data)
            for data in db_gene_data:
                if data not in uniq_data:
                    uniq_data.append(data)
        genes_onehot_vectors = [[0] * len(uniq_data)] * len(self.genes)
        npa = np.array(genes_onehot_vectors)
        sorted_uniq = sorted(uniq_data)
        for i in range(len(genes_data)):
            for j in range(len(sorted_uniq)):
                found = [x for x in genes_data[i] if x == sorted_uniq[j]]
                if len(found) > 0:
                    npa[i][j] = 1
        print('mappingDatabaseToOneHot::{}::found shape {}'.format(db_name, npa.shape))
        return npa

    def sequencesAnalys(self):
        max_seq = None
        max_set_seq = None
        for gene in self.genes:
            seq = sequence(gene.id())
            onehot = gene.apiSeqOneHot()
            set_seq = set(seq)
            if max_seq == None or len(seq) > len(max_seq):
                max_seq = seq
            if max_set_seq == None or len(set_seq) > len(max_set_seq):
                max_set_seq = set_seq
        print('max seq', max_seq, len(max_seq))
        print('max set seq', max_set_seq, len(max_set_seq))
    
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
    dataloader = DataLoader('../config/dense_train.yaml')
    dataloader.mappingDatabaseToOneHot('GI')
    dataloader.mappingDatabaseToOneHot('GO')
    dataloader.mappingDatabaseToOneHot('PubMed')
    dataloader.mappingDatabaseToOneHot('MIM')
    dataloader.mappingDatabaseToOneHot('RefSeq')
    dataloader.mappingDatabaseToOneHot('Ensembl')
    dataloader.sequencesAnalys()
    
        