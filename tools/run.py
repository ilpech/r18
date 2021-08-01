import os
from tools import readSearchXlsxReport
from gene import Gene
from typing import List
from gene_mapping import uniprot_mapping_header, mapping2dict
# уточнить ссылки на статьи
# 1. rna – https://www.biorxiv.org/content/10.1101/2020.11.04.358739v2.full.pdf
# 2. prot – https://pubs.acs.org/doi/10.1021/acs.jproteome.0c00856
#    supporting_info(data): https://pubs.acs.org/doi/10.1021/acs.jproteome.0c00856?goto=supporting-info
# 3. попросить аккаунт с доступом к статьям от нии

print('reading mapping...')
genes_mapping = mapping2dict('../data/liver_hepg2/HUMAN_9606_idmapping_selected.tab.gz')
print('...reading mapping')
data_path = '../data/liver_hepg2'
rna_file_path = 'rna_liver_hepg2_13_20_no_header.xlsx'
prot_1D_n = 'prot_1D_analysis.xlsx'
prot_2D_n = 'prot_2D_analysis.xlsx'
rna_data = readSearchXlsxReport(
    os.path.join(data_path, rna_file_path),
    'Chr18_data'
)
prot_1D = readSearchXlsxReport(
    os.path.join(data_path, prot_1D_n),
    'Лист1'
)
prot_2D = readSearchXlsxReport(
    os.path.join(data_path, prot_2D_n),
    'Лист1'
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
    # print('rna experiments:: len::', len(gene.rna_measurements))
    # for m,v in gene.rna_measurements.items():
    #     print(m, '::', v)
    # print('gene.protein_copies_per_cell_1D', gene.protein_copies_per_cell_1D)
    # print('gene.protein_copies_per_cell_2D', gene.protein_copies_per_cell_2D)
    print('GO keywords::')
    print(genes_mapping[gene.id()]['GO'])
    print('Ensembl keywords::')
    print(genes_mapping[gene.id()]['Ensembl'])

print('\n============ OVERALL')
print('genes', len(genes))
with_prot1D = len([x for x in genes if x.protein_copies_per_cell_1D is not None])
with_prot2D = len([x for x in genes if x.protein_copies_per_cell_2D is not None])
with_prot1D2D = len([x for x in genes if x.protein_copies_per_cell_1D is not None and 
                        x.protein_copies_per_cell_2D is not None])
print('with_prot1D', with_prot1D)
print('with_prot2D', with_prot2D)
print('with_prot1D2D', with_prot1D2D)

        
    