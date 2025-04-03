import os
import csv
import pandas as pd

def open_file(location):
    file = open(os.path.expanduser(location), "r")
    return pd.read_csv(file, sep="\t", header=None)

def vcf_to_csv(vcf_file_path, csv_file_path):
    with open(vcf_file_path, "r") as vcf_file:
        with open(csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            for line in vcf_file:
                if line.startswith("##"):
                    continue
                elif line.startswith("#"):
                    header = line.strip("#").strip().split("\t")
                    csv_writer.writerow(header)
                else:
                    data = line.strip().split("\t")
                    csv_writer.writerow(data)

def convert_all_vcfs(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".vcf"):
            vcf_path = os.path.join(source_dir, filename)
            csv_filename = os.path.splitext(filename)[0] + ".csv"
            csv_path = os.path.join(target_dir, csv_filename)
            vcf_to_csv(vcf_path, csv_path)
            print(f"Converted: {vcf_path} -> {csv_path}")

# === Resistant VCFs  === #
input_resistant = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_resistant/gunziped_resistant"
output_resistant = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_resistant/gunziped_resistant/csv_converted"

input_susceptible = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_susceptible/gunziped_susceptible"
output_susceptible = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_susceptible/gunziped_susceptible/csv_susceptible"

# Function execution
convert_all_vcfs(input_resistant, output_resistant)
convert_all_vcfs(input_susceptible, output_susceptible)