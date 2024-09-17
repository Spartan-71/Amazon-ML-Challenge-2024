import pandas as pd

df = pd.read_csv('./checkpoints/full_test_output.csv', index_col=0)
test_df = pd.read_csv('./dataset/test.csv')

total_in_data = len(df.iloc[:])
total_in_test = test_df.iloc[-1]['index'] + 1
print(total_in_test)

df = df.reindex(range(total_in_test))

missing = []
for i in range(total_in_test):
    val = df.at[i, 'e']
    if(pd.isna(val)):
        # print(df.iloc[i])
        missing.append(i)


print(len(missing))

df.to_csv('./cleaned_data.csv')
print("written")
