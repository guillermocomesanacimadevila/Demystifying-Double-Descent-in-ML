#!bin/bash

# Just wget -> EBI URL + Sample tables (.csv)
wget https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/CRyPTIC_reuse_table_20221019.csv
echo "First table = downloaded"

wget https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/CRyPTIC_reuse_table_20231027.csv
echo "Second table = downloaded"

wget https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/CRyPTIC_reuse_table_20231107.csv
echo "Third table = downloaded"

wget https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/CRyPTIC_reuse_table_20231208.csv
echo "Fourth table = downloaded"

wget https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/CRyPTIC_reuse_table_20240917.csv
echo "Fifth table = downloaded"

echo "Data = downloaded!"