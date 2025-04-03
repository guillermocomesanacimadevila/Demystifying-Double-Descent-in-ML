## Evaluating Double Descent in Machine Learning

** Workflow conducted in the Cloud Infrastructure for Microbial Bioinformatics (CLIMB) **

** All Bash scripts have been written in a UNIX-based system ** 

** Python3 (> v.3.12) and a UNIX-based terminal are required for immediate execution ** 

** Execute Bash scripts -> chmod +x file.sh && ./file.sh ** 

## What is Double Descent? 
<img width="390" alt="image" src="https://github.com/user-attachments/assets/72c0717a-0e50-4e0b-a1a4-0509d05c1dba" />


## Pipeline Overview
![image](https://github.com/user-attachments/assets/d70a4fe1-68fa-4541-bceb-c507b3434616)

### Pipeline Execution (Chronollogically)



- 01_get_metadata.sh (** Get CSV metadata - "CRyPTIC_reuse_table_20221019.csv" ** )
- 02_exploratory_analysis.ipynb (** Split Resistant & Susceptible **)
- 03_parse_resistant_vcfs.sh (** Get Resistant VCFs from sub-sampled metadata **)
- 04_parse_susceptible_vcfs.sh (** Get Susceptible VCFs from sub-sampled metadata ** )
- 05_gunzip_resistant.sh (** Gunzip - vcf.gz -> .vcf - Resistant samples **)
- 06_gunzip_susceptible.sh (** Gunzip - vcf.gz -> .vcf - Susceptible samples **)
- 07_vcf_to_csv.py (** Convert all VCFs into CSVs **)
- 08_QC_resistant.py (** QC - Remove INDELs and Empty loci - Resistant samples **)
- 09_QC_susceptible.py (** QC - Remove INDELs and Empty loci - Susceptible samples **)
- 10_Confirm_QC.py (** Confirm QC has worked - Pre-QC vs. Post-QC comparison **)
#### Machine Learning (CRyPTIC)
-  dt_rf_CRyPTIC.py (** Decision Tree and Ensemble Experiments CRyPTIC **)
-  gboost_CRyPTIC.py (** Boosting Rounds and Ensemble Experiments CRyPTIC **)
#### Machine Learning (Synthetic)
-  dt_rf_simulation.py (** Decision Tree and Ensemble Experiments Synthetic **)
-  gboost_simulation.py (** Boosting Rounds and Ensemble Experiments Synthetic **)

## References
CRyPTIC, 2022. A data compendium associating the genomes of 12,289 Mycobacterium tuberculosis isolates with quantitative resistance phenotypes to 13 antibiotics [Online]. PLOS Biology, 20(8), p.e3001721. Available from: https://doi.org/10.1371/journal.pbio.3001721.

Curth, A., Jeffares, A. and van, 2023. A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning [Online]. arXiv.org. Available from: https://arxiv.org/abs/2310.18988 [Accessed 16 March 2025].
