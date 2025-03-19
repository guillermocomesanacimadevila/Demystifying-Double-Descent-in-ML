import os
import pandas as pd

def count_snps(directory):
    snps = []
    
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            snps.append((file, len(df)))
    return snps

def average_snps(snp_list):
    if not snp_list:
        return 0
    total_snps = sum(snp for _, snp in snp_list)
    return total_snps / len(snp_list)

resistant_dir = count_snps("/home/jovyan/DOUBLE_DESCENT/Data/downloaded_resistant/gunziped_resistant/csv_converted/post_QC_resistant")
susceptible_dir = count_snps("/home/jovyan/DOUBLE_DESCENT/Data/downloaded_susceptible/gunziped_susceptible/csv_susceptible/post_QC_susceptible")

# for file, snp in resistant_dir:
#     print(f"{file} has {snp} SNPs")

# for file, snp in susceptible_dir:
#     print(f"{file} has {snp} SNPs")
    
avg_resistant = average_snps(resistant_dir)
avg_susceptible = average_snps(susceptible_dir)

print(f"Average SNPs per sample in Susceptible Dir is {round(avg_susceptible, 0)}")
print(f"Average SNPs per sample in Resistant Dir is {round(avg_resistant, 0)}")
print(f"Overall average = {round(((avg_resistant + avg_susceptible) / 2), 0)}") # 1548

# Now check for before removing uninformative SNPs

resistant_dir_pre = count_snps("/home/jovyan/DOUBLE_DESCENT/Data/downloaded_resistant/gunziped_resistant/csv_converted")
susceptible_dir_pre = count_snps("/home/jovyan/DOUBLE_DESCENT/Data/downloaded_susceptible/gunziped_susceptible/csv_susceptible")

avg_resistant_pre = average_snps(resistant_dir_pre)
avg_susceptible_pre = average_snps(susceptible_dir_pre)

print(f"PRE - Average SNPs per sample in Susceptible Dir is {round(avg_susceptible_pre, 0)}")
print(f"PRE - Average SNPs per sample in Resistant Dir is {round(avg_resistant_pre, 0)}")
print(f"PRE - Overall average = {round(((avg_resistant_pre + avg_susceptible_pre) / 2), 0)}")
print(f"Average SNPs removed per sample = {round((((avg_resistant_pre + avg_susceptible_pre) / 2) - ((avg_resistant + avg_susceptible) / 2)), 0)}")