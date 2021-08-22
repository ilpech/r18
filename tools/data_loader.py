# python3 data_loader.py --rna ../data/liver_hepg2/rna_liver_hepg2_13_20_no_header.xlsx --prot1D ../data/liver_hepg2/prot_1D_analysis.xlsx --prot2D ../data/liver_hepg2/prot_2D_analysis.xlsx --geneMapping ../data/liver_hepg2/human_18chr_gene_mapping.tab --ionData ../data/liver_hepg2/prot_ion_data.xlsx
# todo
# data stored in shape [
    # dataset_size, 
    # useful_sheets_size(like_channels), 
    # amino_acids_size(20), 
    # max_aplhabet_size
# ]
# max 3349500 database Additional PubMed matrix (275, 609, 20)
# useful_sheets_size contains of:
#   - channel full of rna expetiment value
#   - channel for rna experiment_id(?)
#   - channel for amino acids sequence coding (...*20)
#   OR - channels for rna experiments(?)
#   - channels for each needed database with size <= max_database.shape[1]
# 22 databases now
# 3 channels listed above
# 275 genes for 18 chr
# 275*(22+3)*20*609
# >>> sample_size = 25*20*609
# >>> sample_size
# 304500
# like 24.78 pictures 64*64*3 => ~3 samples in batch
# dataset of 275 weight like 6814.5 pictures 64*64*3
import os
import mxnet as mx
import numpy as np
import argparse
from tools import (
    debug,
    str2bool, 
    readSearchXlsxReport, 
    roundUp,
    norm_shifted_log,
    denorm_shifted_log,
    list2nonEmptyIds,
    listfind
)
from gene_mapping import (
    mapping2dict, 
    uniprot_mapping_header, 
    uniq_nonempty_uniprot_mapping_header
)
from uniprot_api import getGeneFromApi, sequence
from typing import List
from gene import Gene

class DataLoader:
    """
    data loader for liver_hepg2 r18 protein abundance experiment
    """
    class magic_consts:
        '''
        just previously counted values for default setup
        '''
        # def __init__(self):
        #     # DATABASE:: Additional PubMed
        #     # db alph size 12180
        #     # db aplh size converted to amino acids:: 609.0*20
        max_database_alph_size = 12180
        protein_amino_acids_size = 20

    @staticmethod
    def max_db2acids_size():
        return DataLoader.magic_consts.max_database_alph_size/ \
               DataLoader.magic_consts.protein_amino_acids_size 
        
    
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
    
    def gene(self, upirot_gene_id):
        gene_ids = [i for i in range(len(self.genes)) if self.genes[i].id() == upirot_gene_id]
        if not gene_ids:
           raise Exception('gene::gene {} not found'.format(upirot_gene_id))
        return self.genes[gene_ids[0]], gene_ids[0]
    
    def maxRnaMeasurementsInData(self):
        m = 0
        for g in self.genes:
            exps = len(g.rna_measurements)
            if exps > m:
                m = exps
        return m

    def gene2sampleExperimentHasId(
        self, 
        uniprot_gene_id, 
        databases_gene_data,
        databases2use,
        max_measures=None
    ):
        if not max_measures:
            max_measures = self.maxRnaMeasurementsInData()
        gene, gene_idx = self.gene(uniprot_gene_id)
        experiments_size = len(gene.rna_measurements)
        variable_length_layer_size = int(DataLoader.max_db2acids_size())
        batch = np.zeros(
            (
                (len(databases2use)+3), 
                # 3 channels::
                #   - channel full of rna expetiment value
                #   - channel for rna experiment_id(?)
                #   - channel for amino acids sequence coding (...*20)
                variable_length_layer_size,
                int(DataLoader.magic_consts.protein_amino_acids_size)
            )
        )
        gene_experiments_batches = [batch] * experiments_size
        experiments_alph = sorted([e for e,_ in gene.rna_measurements.items()])
        gene_seq_onehot = gene.apiSeqOneHot()
        onehot_rows = gene_seq_onehot.shape[0]
        if onehot_rows > variable_length_layer_size:
           onehot_rows = variable_length_layer_size
        api_seq = gene.apiSequence()
        for i in range(experiments_size):
            exp = experiments_alph[i]
            value = gene.rna_measurements[exp]
            norm_value = norm_shifted_log(value)
            # first channel is fullfilled with rna experiment value
            gene_experiments_batches[i][0].fill(norm_value)
            # second channel is fullfilled with rna experiment id
            gene_experiments_batches[i][1].fill(i)
            # third channel is filled by gene seq in onehot representation
            if onehot_rows:
                gene_experiments_batches[i][2][:onehot_rows] = gene_seq_onehot[:onehot_rows]
            # !debug
            # for l,row in enumerate(gene_experiments_batches[i][2]):
            #     id_ = list2nonEmptyIds(row)
            #     if not len(id_):
            #         # empty row detected
            #         break
            #     id_ = id_[0]
            #     debug(Gene.proteinAminoAcidsAlphabet()[id_])
            #     # print(gene_seq_onehot[l][id_])
            #     print('from api', api_seq[l])
            #     print(row)
            # debug(np.sum(gene_seq_onehot))
            # debug(np.sum(gene_experiments_batches[i][2]))
            for j, database in enumerate(databases2use):
                id2fill = j + 3
                len_filled = databases_gene_data[j].shape[0]
                gene_experiments_batches[i][id2fill][:len_filled] = databases_gene_data[j]
        print(
            'created batch {} for gene {}'.format(  
                np.array(gene_experiments_batches).shape,
                gene.id()
            ) 
        )
        return gene_experiments_batches            
    
    def dataFromMappingDatabase(self, db_name, gene_name):
        '''
        db_name should exist in self.genes_mapping_databases
        '''
        return self.mapping[gene_name][db_name]
    
    def geneMappingDatabase(self, gene_name):
        return self.geneMappingDatabases[gene_name]
    
    def mappingDatabaseAlphabet(self, db_name):
        uniq_data = []
        for i in range(len(self.genes)):
            db_gene_data = self.dataFromMappingDatabase(db_name, self.genes[i].id_uniprot)
            for data in db_gene_data:
                if data not in uniq_data:
                    uniq_data.append(data) 
        return uniq_data

    def mappingDatabaseAplhabetSize(self, db_name):
        return len(self.mappingDatabaseAlphabet(db_name))
    
    def mappingDatabase2matrix(self, db_name, cols=20):
        onehot = self.mappingDatabase2oneHot(db_name)
        uniq_size = roundUp(onehot.shape[1]/float(cols))
        reshape = np.zeros((onehot.shape[0], uniq_size, cols)).flatten()
        for gene_id in range(len(onehot)):
            for value_id in range(len(onehot[gene_id])):
                reshape[len(onehot[gene_id])*gene_id + value_id] = onehot[gene_id][value_id]
        reshape = np.reshape(reshape, (onehot.shape[0], uniq_size, cols))
        return reshape
    
    def mappingDatabase2oneHot(self, db_name):
        '''
        returns [db_mapping_alphabet, onehotvector]
        '''
        genes_data = []
        uniq_data = []
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
        print('mappingDatabase2oneHot::{}::found shape {}'.format(db_name, npa.shape))
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
    dataloader = DataLoader('../config/train.yaml')
    # gi_onehot = dataloader.mappingDatabase2oneHot('GI')
    # go_onehot = dataloader.mappingDatabase2oneHot('GO')
    # pubmed_onehot = dataloader.mappingDatabase2oneHot('PubMed')
    # mim_onehot = dataloader.mappingDatabase2oneHot('MIM')
    # refseq_onehot = dataloader.mappingDatabase2oneHot('RefSeq')
    # ensembl_onehot = dataloader.mappingDatabase2oneHot('Ensembl')
    max_measures=dataloader.maxRnaMeasurementsInData()
    databases = uniq_nonempty_uniprot_mapping_header()
    databases_data = []
    databases2use =[]
    for x in databases:
        mtrx = dataloader.mappingDatabase2matrix(x)
        if not mtrx.shape[1]:
            continue
        databases_data.append(mtrx)
        databases2use.append(x)
    # exit(0)
    for i,gene in enumerate(dataloader.genes):
        all_databases_gene_data = [x[i] for x in databases_data]
        print('gene {} of {}'.format(i, len(dataloader.genes)))
        gene_batch_exps = dataloader.gene2sampleExperimentHasId(
            gene.id_uniprot, 
            all_databases_gene_data,
            databases2use,
            max_measures
        )
        debug(gene_batch_exps[0].shape)
        exit(0)
    max_shape = 0
    max_shape_orig = (0,0)
    max_shape_name = ''
    for db_name in databases:
        onehot_m = dataloader.mappingDatabase2matrix(db_name)
        shape_s = onehot_m.flatten().shape[0]
        if shape_s > max_shape:
            max_shape = shape_s
            max_shape_name = db_name
            max_shape_orig = onehot_m.shape
        onehot = dataloader.mappingDatabase2oneHot(db_name)
        alph = dataloader.mappingDatabaseAlphabet(db_name)
        print('=====\nDATABASE::',db_name)
        alph_size = len(alph)
        print('db alph size', alph_size)
        print('db aplh size converted to amino acids::', alph_size/20.0)
        if alph_size:
            print('example::')
            print(alph[:5])
    print('max', max_shape, 'database', max_shape_name, 'matrix', max_shape_orig)
    dataloader.sequencesAnalys()
    
        