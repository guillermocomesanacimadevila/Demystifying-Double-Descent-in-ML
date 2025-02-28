-- serum_fe
-- ida

SELECT * FROM serum_fe;

SELECT * FROM ida;

-- ensure that they look all good
SELECT *
FROM serum_fe
WHERE rsids IS NOT NULL
    AND nearest_genes IS NOT NULL
    AND `annotation.GENOME_enrichment_nfe` IS NOT NULL
ORDER BY chrom;  

-- do the same with ida
SELECT *
FROM ida
WHERE rsids IS NOT NULL
    AND nearest_genes IS NOT NULL
    AND `annotation.GENOME_enrichment_nfe` IS NOT NULL
ORDER BY chrom;  

-- Now load the new CSV files
-- Merge both tables but with these SNPs
-- 'rs199598395', 'rs572071331', 'rs117725035', 'rs184867000'
-- cleaned_fe
-- cleaned_ida

SELECT * FROM cleaned_fe;

SELECT * FROM cleaned_ida;

-- Merge
SELECT *
FROM cleaned_fe fe
JOIN cleaned_ida ida
ON fe.rsids = ida.rsids
WHERE fe.rsids IN ('rs199598395', 'rs572071331', 'rs117725035', 'rs184867000');

-- Now select the columns that we care about
-- merged_snps

SELECT * FROM merged_snps;

SELECT 
    `rsids`, 
    `chrom`, 
    `pos`, 
    `ref`, 
    `alt`, 
    `maf`, 
    `beta` AS `beta_fe`, 
    `beta_[0]` AS `beta_ida`, 
    `pval` AS `pval_fe`, 
    `pval_[0]` AS `pval_ida`
FROM merged_snps;

-- Second dataset - SNPs -> Chol_T (GWAS)
-- beta_snp.csv
-- beta_snp

SELECT * FROM beta_snp;

SELECT riskAllele, beta,
       CAST(REGEXP_REPLACE(beta, ' unit (increase|decrease)', '') AS FLOAT) AS beta_cleaned
FROM beta_snp
WHERE beta NOT LIKE '%unit decrease%'
   OR beta IS NULL;




