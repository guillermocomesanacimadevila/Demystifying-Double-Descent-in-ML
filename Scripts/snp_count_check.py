import os
import pandas as pd

def parse_csv(location):
    with open(os.path.expanduser(location), "r") as file:
        return pd.read_csv(file ,sep=",", index_col=0)

vcf_example = parse_csv("/Users/guillermocomesanacimadevila/Desktop/CRyPTIC_cleaning/vcf_sample.csv")
# print(vcf_example.head(n=5))
print(len([x for x in vcf_example["ALT"]])) # 1104

# Check SNPs from initial VCF
vcf_uncleaned = parse_csv("/Users/guillermocomesanacimadevila/Desktop/CRyPTIC_cleaning/uncleaned_vcf.csv")
print(len([x for x in vcf_uncleaned["ALT"]])) #  1470

# Final sum - Kinda uncecesary
print(f"Total SNPs removed = {1470 - 1104} ")