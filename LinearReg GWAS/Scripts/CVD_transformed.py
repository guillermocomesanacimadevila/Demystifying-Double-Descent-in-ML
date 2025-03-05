import os
import pandas as pd

def read_file(location):
    with open(os.path.expanduser(location)) as file:
        return pd.read_csv(file, sep="\t")

def parse_csv(location):
    with open(os.path.expanduser(location)) as file:
        return pd.read_csv(file, sep=",")

df = read_file("/Users/guillermocomesanacimadevila/Desktop/CVD_GWAS.tsv")
# print(df.shape)
# print(df.columns)

# Narrow it down to cols of interest
df = df[["riskAllele", "pValue", "riskFrequency","beta", "ci", "locations", "mappedGenes"]]
df = df.rename(columns={"riskAllele": "SNP", "riskFrequency": "MAF","beta": "Beta", "ci": "CI", "locations": "Position", "mappedGenes": "Mapped Genes"})

print(df.columns)
print(df.shape)
print(df.isna().sum()) # 13 missing values for MAF
# Clean "Beta"

df = df[df["Beta"] != "-"]
df.loc[:, "Beta"] = df["Beta"].str.replace(r"(\d+).*", r"\1", regex=True)
print(len([i for i in df["Beta"] if i == "-"])) # 8680
print(df["Beta"].head(n=5))

# Change MAF, pValue and Beta to FLOAT, change Position to NUMERIC
df["Beta"] = df["Beta"].astype(float)
df["pValue"] = df["pValue"].astype(float)
# df["MAF"] = df["MAF"].astype(float)

# Remove where pVal = 0
df = df[df["pValue"] != 0]

# print(df.dtypes)
print(len([i for i in df["pValue"] if i == 0])) # 0 - all good
print(len([i for i in df["MAF"] if i == "NR"])) # 2052

print(df.head(n=5))
print(df.shape)
# Now export new df
df.to_csv("CVD_cleaned.csv", header=True, sep=",", index=False)