import os
import pandas as pd

def readSearchXlsxReport(file_path, sheet_name=''):
    file_abs_path = os.path.abspath(file_path)
    print('readSearchGUIReport::reading data at::', file_abs_path)
    df_sheet_all = pd.read_excel(file_path, sheet_name=None)
    if len(sheet_name) != 0:
        try:
            return df_sheet_all[sheet_name]
        except KeyError as e:
            print(e)
            print('readSearchGUIReport::sheets in file::')
            [print(x) for x in df_sheet_all]
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