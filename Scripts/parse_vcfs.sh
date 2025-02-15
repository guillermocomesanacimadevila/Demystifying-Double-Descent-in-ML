#!bin/bash

# Base URL for the VCF files
BASE_URL="https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/"

# CSV file containing the VCF column
CSV_FILE="vcf_drug_final.csv"

# Extract VCF filenames and download each file
while IFS=',' read -r _ _ VCF _; do
    # Skip header
    if [[ "$VCF" != "VCF" ]]; then
        FILE_URL="$BASE_URL$VCF"
        echo "Downloading: $FILE_URL"
        wget -q "$FILE_URL" -P downloaded_files/
    fi
done < <(tail -n +2 "$CSV_FILE")

echo "Download complete."

