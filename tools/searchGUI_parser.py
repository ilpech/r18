import pandas as pd
import os

from searchGUI_protein import SearchGUIProtein

class SearchGUIParser:
    @staticmethod
    def readSearchGUIReport(file_path, sheet_name=''):
        file_abs_path = os.path.abspath(file_path)
        print('readSearchGUIReport::reading data at::', file_abs_path)
        df_sheet_all = pd.read_excel(file_path, sheet_name=None)
        print('readSearchGUIReport::sheets in file::')
        [print(x) for x in df_sheet_all]
        if len(sheet_name) != 0:
            try:
                return df_sheet_all[sheet_name]
            except KeyError as e:
                print(e)
                raise Exception(
                    'readSearchGUIReport::selected sheet {} is not accessible for file {}'.format(
                        sheet_name, file_path
                    )
                )
        selected_sheet_name = list(df_sheet_all.keys())[0]
        if len(df_sheet_all) > 1:
            selected_sheet_name = [x for x in df_sheet_all if 'Total_Report' in x]
            if len(selected_sheet_name) == 0:
                raise Exception('more than one sheet and no one with Total_Report')
            selected_sheet_name = selected_sheet_name[0]
        print('readSearchGUIReport::selected_sheet_name::', selected_sheet_name)
        return df_sheet_all[selected_sheet_name]
    
    @staticmethod
    def parseSearchGUIReport(df_sheet):
        if (len(df_sheet)) == 0:
            raise Exception('check you searchGUI report sheet, it is empty now')
        proteins = []
        try:
            prots_group_data = df_sheet['Protein Group']
            prots_confidence_data = df_sheet['Confidence [%]']
            prots_chromosome_data = df_sheet['Chromosome']
            prots_descr_data = df_sheet['Description']
            prots_descrs_data = df_sheet['Descriptions']
            prots_sec_accs_data = df_sheet['Secondary Accessions']
            prots_peptides_data = df_sheet['#Validated Peptides']
            prots_uniq_peptides_data = df_sheet['#Validated Unique Peptides']
            prots_inference_data = df_sheet['Protein Inference']
            confident_coverage_data = df_sheet['Confident Coverage [%]']
            nsaf_ppm_data = df_sheet['Spectrum Counting NSAF [ppm]']
            empai_ppm_data = df_sheet['Spectrum Counting emPAI [ppm]']
            nsaf_fmol_data = df_sheet['Spectrum Counting NSAF [fmol]']
            empai_fmol_data = df_sheet['Spectrum Counting emPAI [fmol]']
            mw_kda_data = df_sheet['MW [kDa]']
        except KeyError as e:
            print(e)
            raise Exception('check you searchGUI report')
        ids2del = []
        for i in range(len(df_sheet)):
            if i in ids2del:
                continue
            if 'Related' in prots_inference_data[i]:
                ids2del.append(i)
                continue
            if prots_confidence_data[i] < 10:
                ids2del.append(i)
                continue
        for i in range(len(df_sheet)):
            if i in ids2del:
                continue
            proteins.append(SearchGUIProtein(
                    prots_group_data[i],
                    prots_confidence_data[i],
                    prots_chromosome_data[i],
                    prots_descr_data[i],
                    prots_descrs_data[i],
                    prots_sec_accs_data[i],
                    prots_peptides_data[i],
                    prots_uniq_peptides_data[i],
                    confident_coverage_data[i],
                    nsaf_ppm_data[i],
                    empai_ppm_data[i],
                    nsaf_fmol_data[i],
                    empai_fmol_data[i],
                    mw_kda_data[i]
                )
            )

            
        for prot in proteins:
            print()
            print(prot.protein_group)
            print(prot.spectrum_data.asnp())
            print()
        print(len(proteins))

if __name__ == '__main__':
    file_path = '/Users/ilpech/repositories/r18/data/searchGUI/Stem_cells_Control_in-gel_12.12.2019.xlsx'
    selected_sheet = SearchGUIParser.readSearchGUIReport(file_path)
    prots = SearchGUIParser.parseSearchGUIReport(selected_sheet)