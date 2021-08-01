import os 
import gzip

def uniprot_mapping_header():
    return [
        'UniProtKB-AC',
        'UniProtKB-ID',
        'GeneID (EntrezGene)',
        'RefSeq',
        'GI',
        'PDB',
        'GO',
        'UniRef100',
        'UniRef90',
        'UniRef50',
        'UniParc',
        'PIR',
        'NCBI-taxon',
        'MIM',
        'UniGene',
        'PubMed',
        'EMBL',
        'EMBL-CDS',
        'Ensembl',
        'Ensembl_TRS',
        'Ensembl_PRO',
        'Additional PubMed'
    ]

def mapping2dict(path):
    with gzip.open(path, 'rt') as f:
        data = f.readlines()
    databases = uniprot_mapping_header()
    out = {}
    for d in data:
        splt = d.split('\t')
        data = {}
        for i in range(len(databases)):
            database_splt = splt[i].replace(';', '').split()
            data[databases[i]] = database_splt 
            if i == 0:
                main_name = database_splt[0]
        out[main_name] = data
    return out


if __name__ == '__main__':
    out = mapping2dict('../data/liver_hepg2/HUMAN_9606_idmapping_selected.tab.gz')
    for gene, databases in out.items():
        print('gene name', gene)
        for db, values in databases.items():
            print('DATABASE::', db)
            print(values)
        print('============')