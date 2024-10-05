import pandas as pd

# 读取三个Excel文件
excel1 = pd.read_csv('csv/20240312_holdout_M.csv', index_col=0, header=0)
# excel2 = pd.read_csv('csv/20240312_holdout_POX.csv', index_col=0, header=0)
# excel3 = pd.read_csv('csv/20240312_holdout_CE.csv', index_col=0, header=0)

# # 合并三个DataFrame，按照行名进行合并
merged_df = pd.concat([excel1], axis=1)

# # 将合并后的DataFrame保存为新的Excel文件
merged_df.to_excel('csv/20240312_holdout_M.xlsx')

###############################################
# AE
# excel1 = pd.read_excel('20231212_all.xlsx', sheet_name='origin', index_col=0, header=0)
# excel2 = pd.read_excel('output.xlsx', index_col=0, header=0)

# 合并三个DataFrame，按照行名进行合并
# merged_df = pd.concat([excel1, excel2], axis=1)

# 将合并后的DataFrame保存为新的Excel文件
# merged_df.to_excel('csv/AE.xlsx')