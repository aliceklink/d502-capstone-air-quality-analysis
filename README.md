# D502 Capstone: PM2.5 Air Pollution and Asthma Prevalence Analysis

**Author:** Alice Klink  
**Institution:** Western Governors University  
**Program:** Data Analytics  
**Course:** D502 Data Analytics Capstone  
**Date:** May 2025

## Project Overview

This capstone project investigates the statistical relationship between PM2.5 air pollution concentrations and asthma prevalence rates across US counties. Using data from the EPA Air Quality System and CDC PLACES, this analysis employs rigorous statistical methods to examine potential correlations between environmental air quality and respiratory health outcomes at the population level.

### Research Question
**Is there a statistically significant correlation between PM2.5 air pollution concentrations and asthma prevalence rates in US counties?**

## Key Findings

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Dataset** | 607 US counties | All 50 states represented |
| **Correlation Coefficient (r)** | -0.025 | Negligible negative correlation |
| **Statistical Significance** | p = 0.539 | Not statistically significant |
| **Effect Size (r²)** | 0.0006 | 0.1% variance explained |
| **Conclusion** | **No significant relationship** | County-level PM2.5 does not correlate with asthma prevalence |

## Methodology

### Data Sources
- **EPA Air Quality System (AQS):** PM2.5 annual mean concentrations by county (2023)
- **CDC PLACES:** County-level age-adjusted asthma prevalence data (2023)

### Statistical Analysis
- **Exploratory Data Analysis:** Distribution analysis, outlier detection using IQR method
- **Pearson Correlation Analysis:** Linear relationship assessment with 95% confidence intervals
- **Two-Sample T-Tests:** Group comparison between high and low PM2.5 counties
- **Assumption Testing:** Normality (D'Agostino-Pearson) and equal variance (Levene) validation

### Data Quality Assurance
- 97.4% data retention after outlier removal (16 outliers removed from 623 counties)
- No missing values in final analytical dataset
- Comprehensive geographic coverage across all 50 US states
- Minimum 30 PM2.5 measurements per county for reliable annual averages

## Repository Structure

```
d502-capstone-air-quality-analysis/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── src/                              # Source code
│   ├── data_collection.py            # EPA/CDC data collection pipeline
│   ├── main_analysis.py              # Complete statistical analysis
│   └── statistical_analysis.py       # Additional statistical methods
├── data/processed/                   # Clean, analysis-ready datasets
│   ├── merged_pm25_asthma.csv        # Final analysis dataset (607 counties)
│   ├── epa_pm25_annual.csv          # EPA county-level PM2.5 data
│   └── cdc_asthma_prevalence.csv     # CDC asthma prevalence data
└── results/                          # Analysis outputs
    ├── figures/                      # Publication-quality visualizations
    │   ├── exploratory_analysis.png   # EDA plots and distributions
    │   ├── correlation_analysis.png   # Correlation and regression analysis
    │   └── hypothesis_testing.png     # Group comparison visualizations
    └── tables/                       # Statistical results and datasets
        ├── analysis_results_summary.csv # Key statistical findings
        └── final_analysis_dataset.csv   # Complete cleaned dataset
```

## How to Reproduce This Analysis

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy requests openpyxl statsmodels
```

### Running the Analysis
```bash
# 1. Collect and process EPA/CDC data
python src/data_collection.py

# 2. Perform complete statistical analysis
python src/main_analysis.py
```

### Expected Runtime
- **Data collection:** 5-10 minutes (depending on download speeds)
- **Statistical analysis:** 2-3 minutes
- **Total:** ~15 minutes for complete reproduction

## Detailed Statistical Results

### Dataset Characteristics
- **Sample Size:** 607 US counties after quality control
- **Geographic Coverage:** All 50 US states
- **PM2.5 Range:** 3.60 - 13.38 μg/m³ (Mean: 8.51, SD: 1.87)
- **Asthma Prevalence Range:** 8.20 - 13.30% (Mean: 10.74%, SD: 0.94%)

### Correlation Analysis
- **Pearson r:** -0.025 (95% CI: -0.104, 0.055)
- **P-value:** 0.539 (not significant at α = 0.05)
- **Effect Size:** Trivial (Cohen's interpretation)
- **Variance Explained:** 0.1% (r² = 0.0006)

### Hypothesis Testing
- **Groups:** High PM2.5 (>8.63 μg/m³, n=303) vs. Low PM2.5 (≤8.63 μg/m³, n=304)
- **Mean Difference:** -0.062% (High: 10.71%, Low: 10.77%)
- **T-statistic:** -0.816 (df = 605)
- **P-value:** 0.415 (not significant)
- **Effect Size:** Negligible (Cohen's d = -0.066)

## Scientific Conclusions

### Primary Finding
**No statistically significant correlation exists between county-level PM2.5 concentrations and asthma prevalence rates** in this analysis of 607 US counties. This null finding provides important evidence for understanding air quality-health relationships at the ecological level.

### Implications
1. **Public Health Policy:** County-level PM2.5 may not be the primary driver of geographic asthma prevalence patterns
2. **Research Focus:** Other factors (socioeconomic, healthcare access, allergens) may be more influential
3. **Methodological Insights:** Individual-level studies may be needed to detect health effects
4. **Environmental Regulation:** PM2.5 standards may be effectively controlling harmful exposures

## Study Limitations

### Design Limitations
- **Ecological Fallacy:** County-level associations may not reflect individual-level relationships
- **Cross-sectional Design:** Cannot establish causation or temporal relationships
- **Temporal Mismatch:** EPA and CDC data collection periods may differ

### Potential Confounders
- Socioeconomic status and income inequality
- Healthcare access and quality
- Other air pollutants (ozone, NO₂, particulate composition)
- Environmental allergens and indoor air quality
- Population demographics and lifestyle factors

### Data Limitations
- County-level aggregation may mask local variations
- Single-year analysis may not capture long-term trends
- Self-reported asthma prevalence from CDC PLACES

## Future Research Directions

1. **Individual-Level Studies:** Personal exposure monitoring with health outcomes
2. **Longitudinal Analysis:** Multi-year trends and lagged health effects
3. **Multivariate Modeling:** Control for socioeconomic and demographic confounders
4. **Multi-Pollutant Analysis:** Combined effects of various air pollutants
5. **Spatial Analysis:** Geographic clustering and regional environmental factors
6. **Vulnerable Populations:** Age-stratified and high-risk group analyses

##  Data Sources and References

### Primary Data Sources
- **EPA Air Quality System (AQS):** https://www.epa.gov/aqs
- **CDC PLACES Dataset:** https://www.cdc.gov/places/

### Statistical Methodology References
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.)
- Field, A. (2013). *Discovering Statistics Using IBM SPSS Statistics* (4th ed.)

### Scientific Literature
- Guarnieri, M., & Balmes, J. R. (2014). Outdoor air pollution and asthma. *The Lancet*, 383(9928), 1581-1592.
- Khreis, H., et al. (2017). Exposure to traffic-related air pollution and risk of development of childhood asthma. *Environment International*, 100, 1-31.
- Orellano, P., et al. (2017). Effect of outdoor air pollution on asthma exacerbations in children and adults. *PLOS ONE*, 12(3), e0174050.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information

**Alice Klink**  
Western Governors University  
Data Analytics Program  
Email: aklink2@wgu.edu
GitHub: [@aliceklink](https://github.com/aliceklink)





