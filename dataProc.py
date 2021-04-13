import pandas as pd

preFix = 'cb-devices-main('
postFix = ').csv'
files = []
dfList = []

for i in range(1, 11):
    filename = preFix + str(i) + postFix
    df = pd.read_csv(filename)
    dfList.append(df)

df = pd.concat(dfList, axis=0, ignore_index=True)

print(df.describe())
