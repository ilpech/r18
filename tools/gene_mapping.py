import os 
import gzip

def uniq_nonempty_uniprot_mapping_header():
    return [
        'GO',
        'RefSeq',
        'PIR',
        'MIM',
        'PubMed',
        'Ensembl_PRO',
    ]
    
def uniprot_mapping_header():
    return [
        'UniProtKB-AC',
        'UniProtKB-ID',
        'GeneID (EntrezGene)',
        'RefSeq',
        'GI', # too big
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
    with open(path, 'r') as f:
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

def rewrite_mapping_with_ids(
    path, 
    uniprot_ids,
    outpath
):
    with gzip.open(path, 'rt') as f:
        data = f.readlines()
    lines_ids2remain = []
    databases = uniprot_mapping_header()
    for line_id in range(len(data)):
        splt = data[line_id].split('\t')
        for i in range(len(databases)):
            database_splt = splt[i].replace(';', '').split()
            if i == 0:
                main_name = database_splt[0]
                if main_name in uniprot_ids:
                    lines_ids2remain.append(line_id)
    new_data = [data[i] for i in range(len(data)) if i in lines_ids2remain]
    with open(outpath, 'w') as f:
        f.writelines(new_data)
    print('remapping written in ', outpath)
                    


if __name__ == '__main__':
    out = mapping2dict('../data/liver_hepg2/human_18chr_gene_mapping.tab')
    for gene, databases in out.items():
        print('gene name', gene)
        for db, values in databases.items():
            print('DATABASE::', db)
            print(values)
        print('============')