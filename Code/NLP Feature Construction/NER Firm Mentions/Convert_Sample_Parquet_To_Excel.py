# Small script to convert firm mentions parquet file to excel file
import pandas as pd

# Load parquet file
df = pd.read_parquet('../../../Data/Company_Mentions/Company_Mentions_Sample.parquet')

# Print df
print(df)

# Save to excel
df.to_excel('../../../Data/Company_Mentions/Company_Mentions_Sample.xlsx', index=False)
