# 📉 Evaluating Double Descent in Machine Learning

**Project conducted on the [CLIMB](https://www.climb.ac.uk/) Cloud Infrastructure for Microbial Bioinformatics**

This repository investigates the phenomenon of **double descent** in decision tree-based and gradient boosting models, using both **simulated** and **real-world genomic data** from the CRyPTIC consortium. The pipeline is fully reproducible and executable on UNIX-based systems.

---

## 🧠 What is Double Descent?

Double descent describes a curious U-turn in the test error of overparameterized models: after the classical bias-variance tradeoff peak, error decreases again with increasing model complexity. We evaluate this using decision trees, random forests, and gradient boosting.

<p align="center">
  <img width="400" alt="Double Descent" src="https://github.com/user-attachments/assets/72c0717a-0e50-4e0b-a1a4-0509d05c1dba" />
</p>

---

## ⚙️ Requirements

- **Python 3.12+**
- **UNIX-based terminal** (e.g., Linux/macOS)
- `wget`, `bash`, and Python libraries from `requirements.txt`

> 💡 Make Bash scripts executable with:  
> `chmod +x script.sh && ./script.sh`

---

## 🧬 Dataset

We use high-quality phenotype-genotype data from:

> **CRyPTIC Consortium**, 2022.  
> *A data compendium associating the genomes of 12,289 Mycobacterium tuberculosis isolates with quantitative resistance phenotypes to 13 antibiotics.*  
> [PLOS Biology](https://doi.org/10.1371/journal.pbio.3001721)

---

## 📊 Pipeline Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/d70a4fe1-68fa-4541-bceb-c507b3434616" alt="Pipeline Diagram"/>
</p>

---

## 🧪 Execution Workflow

This section outlines the complete chronological pipeline to evaluate double descent using CRyPTIC data and simulated models.

### 🔹 Step 1: Download Metadata Tables

```bash
chmod +x 01_get_metadata.sh
./01_get_metadata.sh
```

### 🔹 Step 2: Subsample and Clean Metadata

```bash
Run the notebook 02_exploratory_analysis.ipynb
```

### 🔹 Step 3: Parse VCF Files

```bash
chmod +x 03_parse_resistant_vcfs.sh && ./03_parse_resistant_vcfs.sh
chmod +x 04_parse_susceptible_vcfs.sh && ./04_parse_susceptible_vcfs.sh
```

### 🔹 Step 4: Unzip .vcf.gz Files

```bash
chmod +x 05_gunzip_resistant.sh && ./05_gunzip_resistant.sh
chmod +x 06_gunzip_susceptible.sh && ./06_gunzip_susceptible.sh
```

### 🔹 Step 5: Convert VCF to CSV

```bash
python 07_vcf_to_csv.py
```

### 🔹 Step 6: Quality Control

```bash
python 08_QC_resistant.py
python 09_QC_susceptible.py
python 10_Confirm_QC.py
```

### 🔹 Step 7: Machine Learning Experiments (Synthetic) 

```bash
python dt_rf_simulation.py       # Decision Tree & Random Forest
python gboost_simulation.py      # Gradient Boosting composite
```

### 🔹 Step 8: Machine Learning Experiments (CRyPTIC) 

```bash
python dt_rf_CRyPTIC.py          # Double descent with tree models
python gboost_CRyPTIC.py         # Gradient boosting curves
```

---

## 📚 References

CRyPTIC, 2022. A data compendium associating the genomes of 12,289 Mycobacterium tuberculosis isolates with quantitative resistance phenotypes to 13 antibiotics [Online]. PLOS Biology, 20(8), p.e3001721. Available from: https://doi.org/10.1371/journal.pbio.3001721.

Curth, A., Jeffares, A. and van, 2023. A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning [Online]. arXiv.org. Available from: https://arxiv.org/abs/2310.18988 [Accessed 16 March 2025].
