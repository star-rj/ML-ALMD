import pandas as pd
from sklearn.model_selection import train_test_split

excel_path = "/Users/zhangxiwen/project/patient_classification/features/20240227_RJ_M_POX_CE.xlsx"
split_nums = 10
origin_sheet = 'Sheet1'

df = pd.read_excel(excel_path, sheet_name=origin_sheet)
ALLs = []
M2s = []
M3s = []
M4s = []
M5s = []
for index, row in df.iterrows():
    a = row.iloc[0]
    if a.startswith('ALL'):
        ALLs.append(row.to_list())
    elif a.startswith('M2'):
        M2s.append(row.to_list())
    elif a.startswith('M3'):
        M3s.append(row.to_list())
    elif a.startswith('M4'):
        M4s.append(row.to_list())
    else:
        M5s.append(row.to_list())
ALLs = pd.DataFrame(ALLs)
M2s = pd.DataFrame(M2s)
M3s = pd.DataFrame(M3s)
M4s = pd.DataFrame(M4s)
M5s = pd.DataFrame(M5s)

result_df = pd.DataFrame(columns=df.columns)
for i in range(split_nums):
    train_data_ALL, test_data_ALL = train_test_split(ALLs, test_size=0.3, random_state=i)
    train_data_M2, test_data_M2 = train_test_split(M2s, test_size=0.3, random_state=i)
    train_data_M3, test_data_M3 = train_test_split(M3s, test_size=0.3, random_state=i)
    train_data_M4, test_data_M4 = train_test_split(M4s, test_size=0.3, random_state=i)
    train_data_M5, test_data_M5 = train_test_split(M5s, test_size=0.3, random_state=i)
    train_data = pd.concat([train_data_ALL, train_data_M2, train_data_M3, train_data_M4, train_data_M5])
    test_data = pd.concat([test_data_ALL, test_data_M2, test_data_M3, test_data_M4, test_data_M5])
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        train_sheet_name = f"train_{i}"
        train_data.to_excel(writer, sheet_name=train_sheet_name, index=False)
        test_sheet_name = f"test_{i}"
        test_data.to_excel(writer, sheet_name=test_sheet_name, index=False)