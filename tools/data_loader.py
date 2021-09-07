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
    listfind,
    is_number,
    setindxs,
    shuffle,
    RnaProtAbundance
)
from gene_mapping import (
    mapping2dict, 
    uniprot_mapping_header, 
    uniq_nonempty_uniprot_mapping_header,
    rewrite_mapping_with_ids
)
from uniprot_api import getGeneFromApi, sequence
from gene import Gene, GenesMapping
import csv

from typing import List

# isdebug = True
isdebug = False

class DataLoader:
    """
    data loader for liver_hepg2 r18 protein abundance experiment
    """
    class magic_consts:
        '''
        just previously counted values for default setup
        '''
        protein_amino_acids_size = 20

    # @staticmethod
    # def max_db2acids_size():
    #     return roundUp(
    #         DataLoader.magic_consts.max_database_alph_size/ \
    #         DataLoader.magic_consts.protein_amino_acids_size
    #     )
        
    
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
    
    def __init__(self, config_path, only_genes_with_value=True):
        self.config_path = config_path
        self.only_genes_with_value = only_genes_with_value
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
        self.databases_alphs = {}
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
        engs2uniprot_file = '../data/mapping_out/engs2uniprot.txt'
        full_genome_mappping_path = '../data/liver_hepg2/HUMAN_9606_idmapping_selected.tab'
        print('reading ensg2uniprot_mapping')
        if not os.path.isfile(engs2uniprot_file):
            self.ensg2uniprot_mapping: GenesMapping = DataLoader.ensg2uniprot(
                full_genome_mappping_path,
                engs2uniprot_file 
            )
        else:
            self.ensg2uniprot_mapping = GenesMapping(engs2uniprot_file)
        self.gene_ids = [x for x in self.rna_data['Uniprot AC'] if isinstance(x, str)]
        self.__genes: List[Gene] = []
        print('creating genes')
        for gene_id in self.gene_ids:
            try:
                gene = Gene(
                    gene_id,
                    only_w_values=self.only_genes_with_value
                )
                gene.addRNAReport(self.rna_data)
                gene.addProteinAbundanceReport(
                    self.prot1D_data,
                    self.prot2D_data
                )
                self.genes().append(gene)
            except Exception as e:
                print(e)
                pass
        print('{} genes created'.format(len(self.genes())))
    
    def genes(self):
        return self.__genes
    
    def gene(self, upirot_gene_id):
        gene_ids = [i for i in range(len(self.genes())) if self.genes()[i].id() == upirot_gene_id]
        if not gene_ids:
           raise Exception('gene::gene {} not found'.format(upirot_gene_id))
        return self.genes()[gene_ids[0]], gene_ids[0]
    
    def rnaExperimentsCount(self):
        return sum([
            len(x.rna_measurements) for x in self.genes()
        ])
        
    def proteinExperimentsCount(self):
        return sum([
            len(x.protein_measurements) for x in self.genes()
        ])
    
    def rnaMeasurementsAlphabet(self):
        out = []
        for g in self.genes():
            for k, v in g.rna_measurements.items():
                if k not in out:
                    out.append(k)
        out = set(out)
        return sorted(out)
    
    def proteinMeasurementsAlphabet(self):
        out = []
        for g in self.genes():
            for k, v in g.protein_measurements.items():
                if k not in out:
                    out.append(k)
        out = set(out)
        return sorted(out)
    
    def maxRnaMeasurementsInData(self):
        m = 0
        for g in self.genes():
            exps = len(g.rna_measurements)
            if exps > m:
                m = exps
        return m

    def gene2sampleExperimentHasId(
        self, 
        uniprot_gene_id, 
        databases_gene_data,
        databases2use,
        max_db_data,
        rna_exps_alphabet=None,
        protein_exps_alphabet=None
    ):
        if not rna_exps_alphabet:
            rna_exps_alphabet = self.rnaMeasurementsAlphabet()
        if not protein_exps_alphabet:
            protein_exps_alphabet = self.proteinMeasurementsAlphabet()
        gene, gene_idx = self.gene(uniprot_gene_id)
        rna_experiments_size = len(rna_exps_alphabet)
        variable_length_layer_size = max_db_data
        batch = np.zeros(
            (
                (len(databases2use)+4), 
                # 4 channels::
                #   - channel full of rna expetiment value
                #   - channel for rna experiment_id
                #   - channel for prot label experiment_id
                #   - channel for amino acids sequence coding (...*20)
                variable_length_layer_size,
                int(DataLoader.magic_consts.protein_amino_acids_size)
            )
        )
        gene_experiments_batches = [None] * (len(rna_exps_alphabet)*len(protein_exps_alphabet))
        labels2add_prot = ['protein_copies_per_cell_1D', 'protein_copies_per_cell_2D']
        gene_seq_onehot = gene.apiSeqOneHot()
        onehot_rows = gene_seq_onehot.shape[0]
        if onehot_rows > variable_length_layer_size:
           onehot_rows = variable_length_layer_size
        last_filled = 0
        for i in range(rna_experiments_size):
            experiment = rna_exps_alphabet[i]
            try:
                value = gene.rna_measurements[experiment]
            except:
                continue
            if 'tissue29_' in experiment:
                ids2fill_prot_exp = [j for j in range(len(protein_exps_alphabet)) if protein_exps_alphabet[j] == experiment]
            else:
                ids2fill_prot_exp = [j for j in range(len(protein_exps_alphabet)) if protein_exps_alphabet[j] in labels2add_prot]
            for j in range(len(ids2fill_prot_exp)): 
                norm_value = norm_shifted_log(value)
                gene_experiments_batches[last_filled] = batch.copy() 
                # first channel is fullfilled with rna experiment value
                gene_experiments_batches[last_filled][0].fill(norm_value)
                # second channel is fullfilled with rna experiment id
                id2fill_rna_exp = i
                gene_experiments_batches[last_filled][1].fill(id2fill_rna_exp)
                # third channel is full filled with prot experiment id
                id2fill_prot_exp = ids2fill_prot_exp[j]
                gene_experiments_batches[last_filled][2].fill(id2fill_prot_exp)
                # 4-nd channel is filled by gene seq in onehot representation
                if onehot_rows:
                    gene_experiments_batches[last_filled][3][:onehot_rows] = gene_seq_onehot[:onehot_rows]
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
                for k, database in enumerate(databases2use):
                    id2fill = k + 4
                    len_filled = databases_gene_data[k].shape[0]
                    gene_experiments_batches[last_filled][id2fill][:len_filled] = databases_gene_data[k]
                last_filled += 1
        gene_experiments_batches = np.array([x for x in gene_experiments_batches if x is not None])
        if not len(gene_experiments_batches):
            return None 
        # !debug
        # print(
        #     'created batch {} for gene {}. last_layer_id_filled={}'.format(  
        #         gene_experiments_batches.shape,
        #         gene.id(),
        #         last_filled
        #     ) 
        # )
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
        for i in range(len(self.genes())):
            db_gene_data = self.dataFromMappingDatabase(db_name, self.genes()[i].id_uniprot)
            for data in db_gene_data:
                if data not in uniq_data:
                    uniq_data.append(data) 
        return uniq_data

    def mappingDatabaseAplhabetSize(self, db_name):
        return len(self.mappingDatabaseAlphabet(db_name))
    
    def mappingDatabase2matrix(
        self, 
        db_name,
        db_alphabet=None, 
        cols=20
    ):
        '''
        returns onehotmatrix, db_mapping_alphabet
        '''
        onehot, alph = self.mappingDatabase2oneHot(
            db_name, 
            db_alphabet
        )
        uniq_size = roundUp(onehot.shape[1]/float(cols))
        reshape = np.zeros((onehot.shape[0], uniq_size, cols)).flatten()
        for gene_id in range(len(onehot)):
            for value_id in range(len(onehot[gene_id])):
                reshape[len(onehot[gene_id])*gene_id + value_id] = onehot[gene_id][value_id]
        reshape = np.reshape(reshape, (onehot.shape[0], uniq_size, cols))
        return reshape, alph
    
    def mappingDatabase2oneHot(
        self, 
        db_name,
        db_alphabet=None,
        outpath='../data/mapping_out'
    ):
        '''
        returns onehotvector, db_mapping_alphabet
        '''
        genes_data = []
        uniq_data = db_alphabet
        if uniq_data is None:
            uniq_data = []
            for i in range(len(self.genes())):
                db_gene_data = self.dataFromMappingDatabase(
                    db_name, 
                    self.genes()[i].id_uniprot
                )
                genes_data.append(db_gene_data)
                for data in db_gene_data:
                    if data not in uniq_data:
                        uniq_data.append(data) 
        npa = np.zeros(shape=(len(self.genes()), len(uniq_data)))
        sorted_uniq = sorted(uniq_data)
        for i in range(len(genes_data)):
            for j in range(len(sorted_uniq)):
                found = [x for x in genes_data[i] if x == sorted_uniq[j]]
                if len(found) > 0:
                    npa[i][j] = 1
        print('mappingDatabase2oneHot::{}::found shape {}'.format(db_name, npa.shape))
        return npa, sorted_uniq

    def sequencesAnalys(self):
        max_seq = None
        max_set_seq = None
        for gene in self.genes():
            seq = sequence(gene.id())
            onehot = gene.apiSeqOneHot()
            set_seq = set(seq)
            if max_seq == None or len(seq) > len(max_seq):
                max_seq = seq
            if max_set_seq == None or len(set_seq) > len(max_set_seq):
                max_set_seq = set_seq
        print('max seq', max_seq, len(max_seq))
        print('max set seq', max_set_seq, len(max_set_seq))
    
    def data(self, isdebug=False):
        db_lim_ifdebug = 3
        lim_ifdebug = 100
        rna_exps_alphabet = self.rnaMeasurementsAlphabet()
        protein_exps_alphabet = self.proteinMeasurementsAlphabet()
        gene_exp_data2train = []
        databases = uniq_nonempty_uniprot_mapping_header()
        if isdebug:
            databases = databases[:lim_ifdebug]
        databases_data = []
        databases2use =[]
        max_db_alph = 0
        for x in databases:
            mtrx, alph = self.mappingDatabase2matrix(x)
            self.databases_alphs[x] = alph
            if not mtrx.shape[1]:
                continue
            l_alph = roundUp(len(alph)/20)
            if l_alph > max_db_alph:
                max_db_alph = l_alph
            databases_data.append(mtrx)
            databases2use.append(x)
        genes_exps_batches = []
        for j, gene in enumerate(self.genes()):
            if isdebug:
                if j >= lim_ifdebug:
                    break
            print('gene {} of {}'.format(j, len(self.genes())))
            all_databases_gene_data = [x[j] for x in databases_data]
            o = self.gene2sampleExperimentHasId(
                gene.id_uniprot, 
                all_databases_gene_data,
                databases2use,
                max_db_alph,
                rna_exps_alphabet,
                protein_exps_alphabet
            )
            if o is not None:
                genes_exps_batches.append(o)
        data = []
        labels = []
        for gene_id in range(len(genes_exps_batches)):
            gene = self.genes()[gene_id] # проверить точно ли правильная индексация?
            for exp_id in range(len(genes_exps_batches[gene_id])):
                rna_exp_id = int(np.mean(genes_exps_batches[gene_id][exp_id][1]))
                prot_exp_id = int(np.mean(genes_exps_batches[gene_id][exp_id][2]))
                rna_experiment = rna_exps_alphabet[rna_exp_id]
                prot_experiment = protein_exps_alphabet[prot_exp_id]
                try:
                    exp_v = float(gene.protein_measurements[prot_experiment]) 
                    # if not is_number(gene.protein_measurements[experiment]):
                        # continue
                    data.append(genes_exps_batches[gene_id][exp_id].astype('float32'))
                    labels.append(norm_shifted_log(exp_v))
                except:
                    pass
        labels, shuffle_indxs = shuffle(labels)
        data = setindxs(data, shuffle_indxs)
        # !error handling
        # for d in data:
        #     debug(d.shape)
        #     # debug(d[0][0])
        #     # debug(d[0][0].shape)
        #     debug(np.mean(d[0]))
        #     debug(np.mean(d[1]))
        #     debug(np.sum(d[2]))
        #     exit(0)
        #     # debug(d[2][0])
        return data, labels
    
    def info(self):
        for gene in self.genes():
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
    
    @staticmethod 
    def ensg2uniprot(
        mapping_path='../data/liver_hepg2/HUMAN_9606_idmapping_selected.tab',
        out_path='../data/mapping_out/engs2uniprot.txt'
    ):
        print('ensg2uniprot::reading {}'.format(mapping_path))
        mapping = mapping2dict(mapping_path)
        mapping_size = len(mapping)
        ensg_db = 'Ensembl'
        uniprot_db = 'UniProtKB-ID'
        out = GenesMapping()
        i = 0
        for gene_name, value in mapping.items():
            if not i % 5000:
                print('{} of {}'.format(i, mapping_size))
            if len(value[ensg_db]):
                out.add(gene_name, value[ensg_db][0])
            i += 1
        if len(out_path):
            out.write(out_path)
        return out
    
    def uniprot2ensg(self, uniprot_gene_id):
        return self.dataFromMappingDatabase('Ensembl', uniprot_gene_id)
    
    def uniprot2db(self, uniprot_gene_id, db2convert_name):
        try:
            return self.dataFromMappingDatabase(db2convert_name, uniprot_gene_id)
        except:
            raise Exception('error finding mapping Uniprot::{} ==> {} database'.format(
                uniprot_gene_id,
                db2convert_name
            ))
    
    def loadTissue29data2genes(
        self, 
        rna_path, 
        prot_path,
        create_new_genes=True,
        isdebug=False
    ):
        rna_tissues = []
        prot_tissues = []
        rna_header = []
        prot_header = []
        rna_data = []
        rna_ensg_ids = []
        rna_id_col_name = None
        prot_data = []
        prot_ensg_ids = []
        prot_id_col_name = None
        ensg2uniprot = self.ensg2uniprot_mapping.mapping()
        with open(rna_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            if not len(rna_header):
                for rs in reader:
                    for r in rs:
                        rna_header.append(r)
                    break
                rna_id_col_name = rna_header[0]
                rna_tissues = rna_header[1:]
            for ord_dict in reader:
                rna_ensg_ids.append(ord_dict[rna_id_col_name]) 
                rna_data.append(ord_dict)
        print('rna data loaded {} from file {}'.format(
            len(rna_ensg_ids),
            rna_path
        ))
        with open(prot_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            if not len(prot_header):
                for rs in reader:
                    for r in rs:
                        prot_header.append(r)
                    break
                prot_id_col_name = prot_header[0]
                prot_tissues = prot_header[1:]
            for ord_dict in reader:
                prot_ensg_ids.append(ord_dict[prot_id_col_name]) 
                prot_data.append(ord_dict)
        print('prot data loaded {} from file {}'.format(
            len(prot_ensg_ids),
            prot_path
        ))
        if rna_tissues != prot_tissues:
            raise Exception('rna_tissues != prot_tissues')
        tissues = rna_tissues
        good_data = 0
        for i in range(len(prot_ensg_ids)):
            if isdebug and good_data > 500:
                break
            ensg_id = prot_ensg_ids[i]
            uniprot_id = [x.uniprot_id for k, x in ensg2uniprot.items() if x.ensg_id == ensg_id]
            if not len(uniprot_id):
                print('error while searching ensg {} in uniprot'.format(ensg_id))
                continue
            uniprot_id = uniprot_id[0]
            is_new_gene = False
            gene = [i for i in range(len(self.genes())) if self.genes()[i].id() == uniprot_id]
            if not len(gene):
                if not create_new_genes:
                    continue
                gene = Gene(uniprot_id, only_w_values=True)
                is_new_gene = True
            else:
                gene = self.genes()[gene[0]]
                print('tissue29 data would be addect to currently existing gene {}'.format(gene.id()))
            gene.id_ensg = ensg_id
            is_any_found = 0
            for t in range(len(tissues)):
                tissue = tissues[t]
                prot_value = float(prot_data[i][tissue])
                rna_value = float(rna_data[i][tissue])
                if prot_value == 0:
                    continue
                if rna_value == 0:
                    continue
                is_any_found += 1
                good_data += 1
                measurement_name = 'tissue29_{}'.format(tissue)
                gene.rna_measurements[measurement_name] = rna_value
                gene.protein_measurements[measurement_name] = prot_value
            if not is_any_found:
                continue
            if is_new_gene:
                self.genes().append(gene)
        
        print('{} good genes experiments added'.format(good_data))
        print('{} genes now'.format(len(self.genes())))
        print('{} rna experiments in data'.format(self.rnaExperimentsCount()))
        print('{} protein experiments in data'.format(self.proteinExperimentsCount()))

if __name__ == '__main__':
    # DataLoader.ensg2uniprot(
    #     '../data/liver_hepg2/HUMAN_9606_idmapping_selected.tab'
    # )
    # exit()
    dataloader = DataLoader('../config/train.yaml')
    dataloader.loadTissue29data2genes(
        '../data/liver_hepg2/tissue29_rna.tsv',
        '../data/liver_hepg2/tissue29_prot.tsv',
        # '../data/liver_hepg2/tissue29.1k_rna.tsv',
        # '../data/liver_hepg2/tissue29.1k_prot.tsv',
        # '../data/liver_hepg2/tissue29.05k_rna.tsv',
        # '../data/liver_hepg2/tissue29.05k_prot.tsv',
        create_new_genes=True,
        isdebug=isdebug
    )
    databases = uniq_nonempty_uniprot_mapping_header()
    for db_name in databases:
        onehot_m, alph = dataloader.mappingDatabase2matrix(db_name)
        shape_s = onehot_m.flatten().shape[0]
        # if shape_s > max_shape:
        #     max_shape = shape_s
        #     max_shape_name = db_name
        #     max_shape_orig = onehot_m.shape
        onehot, alph = dataloader.mappingDatabase2oneHot(db_name)
        print('=====\nDATABASE::',db_name)
        alph_size = len(alph)
        print('db alph size', alph_size)
        print('db aplh size converted to amino acids::', alph_size/20.0)
        if alph_size:
            print('example::')
            print(alph[:5])
    # print('max', max_shape, 'database', max_shape_name, 'matrix', max_shape_orig)
    dataloader.sequencesAnalys()
    
        