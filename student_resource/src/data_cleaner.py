import pandas as pd
import easyocr

reader = easyocr.Reader(['en'])

df = pd.read_csv('./checkpoints/full_test_output.csv', index_col=0)
test_df = pd.read_csv('./dataset/test.csv', index_col=0)

total_in_data = len(df.iloc[:])
total_in_test = len(test_df.iloc[:])

df = df.reindex(range(total_in_test))

missing = []
for i in range(total_in_test):
    val = df.at[i, 'e']
    if(pd.isna(val)):
        print(df.iloc[i])
        missing.append(i)


print(len(missing))

df.to_csv('./cleaned_data.csv')
print("written")
