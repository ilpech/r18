import os
from tools import readSearchXlsxReport
# TODO
# написать Екатерине Ильгисонис, почему в табличка разные идентификаторы UniprotAC
# уточнить ссылки на статьи
# 1. rna – https://www.biorxiv.org/content/10.1101/2020.11.04.358739v2.supplementary-material
#          https://www.biorxiv.org/content/10.1101/2020.11.04.358739v2.full.pdf
# 2. prot – https://pubs.acs.org/doi/10.1021/acs.jproteome.0c00856
#           https://pubmed.ncbi.nlm.nih.gov/33202127/
#    supporting_info(data): https://pubs.acs.org/doi/10.1021/acs.jproteome.0c00856?goto=supporting-info
# 3. попросить аккаунт с доступом к статьям от нии


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
    'Лист1'
)
prot_2D = readSearchXlsxReport(
    os.path.join(data_path, prot_2D),
    'Лист1'
)
print(len(rna_data), len(prot_1D), len(prot_2D))

uniprot_kb = 'Q8TDN4'
prot_1d_names = prot_1D[' PI (Uniprot)']
prot_2d_names = prot_2D['PI (Uniprot)']
copies_column = 'Copies of protein per cell'
prot_1d_id = [i for i in range(len(prot_1d_names)) if prot_1d_names[i] == uniprot_kb][0]
prot_2d_id = [i for i in range(len(prot_1d_names)) if prot_2d_names[i] == uniprot_kb][0]
prot_1d_copies = prot_1D[copies_column][prot_1d_id]
prot_2d_copies = prot_2D[copies_column][prot_2d_id]
prot_name = prot_1d_names[prot_1d_id]

gene_name = 'TMEM200C'

gene_names = rna_data['GeneNames']
protein_names = rna_data['Protein name']
qPCR_20_d1 = rna_data['qPCR_Liv20.d1 (copies cDNA per cell)']
qPCR_20_d3 = rna_data['qPCR_Liv20.d5 (copies cDNA per cell)']
qPCR_20_d5 = rna_data['qPCR_Liv20.d3 (copies cDNA per cell)']
ont_20_liv = rna_data['ONT_Liv20.d1_TPM _gencode.v32']
ont_20_hepg2 = rna_data['ONT_HepG2_20_ TMP_gencode.v32']
uniprot_ac = rna_data['Uniprot AC']

tmem_id = [i for i in range(len(gene_names)) if gene_names[i] == gene_name][0]
data = [
    gene_names[tmem_id],
    protein_names[tmem_id],
    qPCR_20_d1[tmem_id],
    qPCR_20_d3[tmem_id],
    qPCR_20_d5[tmem_id],
    ont_20_liv[tmem_id],
    ont_20_hepg2[tmem_id],
    'uniprot_ac::{}'.format(uniprot_ac[tmem_id])
]
print('rna\n',data)
prot = [
    prot_name,
    prot_1d_copies,
    prot_2d_copies
]
print('protein\n', prot)
exit(0) 
