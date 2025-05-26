#!/usr/bin/env python3
"""
FIXED Data Collection Script for D502 Capstone
Addresses column name issues in EPA and CDC data

Author: Alice Klink
Date: May 2025
"""

import pandas as pd
import requests
import os
from pathlib import Path
import zipfile
import io
import time
import numpy as np
from datetime import datetime

def setup_directories():
    """Create necessary directories for data storage"""
    directories = [
        'data/raw',
        'data/processed',
        'results/figures',
        'results/tables'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def print_separator(title):
    """Print a formatted separator with title"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def inspect_epa_columns(filepath):
    """Inspect EPA data columns to find the right names"""
    print("Inspecting EPA data structure...")
    
    # Read just the first few rows to check columns
    sample_data = pd.read_csv(filepath, nrows=5)
    print("   Available EPA columns:")
    for i, col in enumerate(sample_data.columns):
        print(f"     {i}: {col}")
    
    # Look for measurement-related columns
    measurement_cols = [col for col in sample_data.columns if 'measure' in col.lower() or 'value' in col.lower() or 'concentration' in col.lower()]
    print(f"   Potential measurement columns: {measurement_cols}")
    
    return sample_data.columns.tolist()

def process_epa_data_fixed(filepath):
    """
    Fixed EPA processing with correct column names
    """
    print("Processing EPA AQS data (FIXED VERSION)...")
    
    if not os.path.exists(filepath):
        print(f"EPA data file not found: {filepath}")
        return None
    
    try:
        # First, inspect the columns
        columns = inspect_epa_columns(filepath)
        
        # Read EPA data
        print("   Loading raw EPA data...")
        epa_raw = pd.read_csv(filepath, low_memory=False)
        print(f"   Raw EPA data loaded: {len(epa_raw):,} records")
        
        # Find the correct column names by checking what's actually in the data
        # Common variations of measurement columns in EPA data:
        measurement_col = None
        for col in epa_raw.columns:
            if any(term in col.lower() for term in ['arithmetic mean', 'sample measurement', 'concentration', 'observed values']):
                measurement_col = col
                break
        
        if measurement_col is None:
            # Try to find numeric columns that might be measurements
            numeric_cols = epa_raw.select_dtypes(include=[np.number]).columns
            print(f"   Available numeric columns: {list(numeric_cols)}")
            
            # Look for columns with reasonable PM2.5 values (0-200 range typically)
            for col in numeric_cols:
                if epa_raw[col].min() >= 0 and epa_raw[col].max() < 500:
                    sample_stats = epa_raw[col].describe()
                    print(f"   Column '{col}' stats: mean={sample_stats['mean']:.2f}, max={sample_stats['max']:.2f}")
                    if 1 < sample_stats['mean'] < 50:  # Reasonable PM2.5 range
                        measurement_col = col
                        break
        
        if measurement_col is None:
            print("Could not identify measurement column")
            print("   Please check the EPA data structure manually")
            return None
            
        print(f"   Using measurement column: '{measurement_col}'")
        
        # Display basic info about the dataset
        print(f"   Date range: {epa_raw['Date Local'].min()} to {epa_raw['Date Local'].max()}")
        
        # Create FIPS code
        epa_raw['FIPS'] = (epa_raw['State Code'].astype(str).str.zfill(2) + 
                          epa_raw['County Code'].astype(str).str.zfill(3))
        
        # Data quality filters using the correct column name
        print("   Applying data quality filters...")
        initial_count = len(epa_raw)
        
        epa_clean = epa_raw[
            (pd.notna(epa_raw[measurement_col])) &  # Not null
            (epa_raw[measurement_col] > 0) &        # Positive values
            (epa_raw[measurement_col] < 500)        # Remove extreme outliers
        ].copy()
        
        print(f"   Records after quality filters: {len(epa_clean):,} ({len(epa_clean)/initial_count*100:.1f}%)")
        
        # Convert date for temporal analysis
        epa_clean['Date Local'] = pd.to_datetime(epa_clean['Date Local'])
        
        # Calculate annual statistics by county
        print("   Calculating county-level annual statistics...")
        
        # Group by FIPS and calculate statistics
        grouping_cols = ['FIPS', 'County Name', 'State Name', 'State Code']
        epa_annual = epa_clean.groupby(grouping_cols).agg({
            measurement_col: ['mean', 'median', 'std', 'count', 'max', 'min']
        }).round(3)
        
        # Flatten column names
        epa_annual.columns = [
            'PM25_Annual_Mean', 'PM25_Annual_Median', 'PM25_Std', 
            'PM25_Sample_Count', 'PM25_Annual_Max', 'PM25_Annual_Min'
        ]
        
        epa_annual = epa_annual.reset_index()
        
        # Quality control: Filter counties with sufficient data
        min_samples = 30
        epa_final = epa_annual[epa_annual['PM25_Sample_Count'] >= min_samples].copy()
        
        print(f"   Counties with ≥{min_samples} measurements: {len(epa_final):,}")
        print(f"   PM2.5 range: {epa_final['PM25_Annual_Mean'].min():.1f} - {epa_final['PM25_Annual_Mean'].max():.1f} μg/m³")
        
        # Add data quality flags
        epa_final['Data_Quality'] = 'Good'
        epa_final.loc[epa_final['PM25_Sample_Count'] < 50, 'Data_Quality'] = 'Moderate'
        epa_final.loc[epa_final['PM25_Sample_Count'] >= 200, 'Data_Quality'] = 'Excellent'
        
        # Save processed data
        output_path = 'data/processed/epa_pm25_annual.csv'
        epa_final.to_csv(output_path, index=False)
        print(f"Processed EPA data saved to: {output_path}")
        
        return epa_final
        
    except Exception as e:
        print(f"Error processing EPA data: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_cdc_data_fixed(filepath):
    """
    Fixed CDC processing with correct column identification
    """
    print("Processing CDC PLACES data (FIXED VERSION)...")
    
    if not os.path.exists(filepath):
        print(f"CDC data file not found: {filepath}")
        return None
    
    try:
        # Read CDC data
        print("   Loading raw CDC data...")
        cdc_raw = pd.read_csv(filepath, low_memory=False)
        print(f"   Raw CDC data loaded: {len(cdc_raw):,} records")
        
        # Show available columns for debugging
        print("   Available CDC columns:")
        for col in cdc_raw.columns:
            print(f"     {col}")
        
        # Filter for asthma prevalence data
        asthma_data = cdc_raw[
            cdc_raw['Measure'].str.contains('asthma', case=False, na=False)
        ].copy()
        
        print(f"   Asthma records found: {len(asthma_data):,}")
        
        if len(asthma_data) == 0:
            print("No asthma records after filtering")
            return None
        
        # Clean and prepare asthma data using CORRECT column names
        print("   Cleaning asthma data...")
        
        # Based on the column list you provided, the correct mappings are:
        # FIPS: LocationID (not CountyFIPS)
        # Value: Data_Value (not DataValueTypeID)
        # Location: LocationName ✓
        # State: StateAbbr ✓
        
        asthma_clean = asthma_data[
            (pd.notna(asthma_data['Data_Value'])) &
            (asthma_data['Data_Value'] > 0) &
            (asthma_data['Data_Value'] < 50)  # Reasonable prevalence range
        ].copy()
        
        # Filter for age-adjusted prevalence if available
        if 'DataValueTypeID' in asthma_clean.columns:
            age_adj_data = asthma_clean[asthma_clean['DataValueTypeID'] == 'AgeAdjPrv']
            if len(age_adj_data) > 0:
                asthma_clean = age_adj_data
                print("   Using age-adjusted prevalence data")
        
        print(f"   Clean asthma records: {len(asthma_clean):,}")
        
        # Create standardized output DataFrame using CORRECT column names
        cdc_final = pd.DataFrame()
        
        # FIPS code - Use LocationID and ensure 5-digit format
        if 'LocationID' in asthma_clean.columns:
            cdc_final['FIPS'] = asthma_clean['LocationID'].astype(str).str.zfill(5)
        else:
            print("LocationID column not found")
            return None
        
        # Asthma prevalence - Use Data_Value
        cdc_final['Asthma_Prevalence'] = asthma_clean['Data_Value'].round(2)
        
        # Location and state info
        cdc_final['County_Name'] = asthma_clean['LocationName']
        cdc_final['State_Code'] = asthma_clean['StateAbbr']
        
        # Confidence intervals
        cdc_final['Asthma_CI_Lower'] = asthma_clean['Low_Confidence_Limit'].round(2)
        cdc_final['Asthma_CI_Upper'] = asthma_clean['High_Confidence_Limit'].round(2)
        
        # Population data
        if 'TotalPopulation' in asthma_clean.columns:
            cdc_final['Population'] = asthma_clean['TotalPopulation']
        
        # Remove duplicates (keep most recent/complete record)
        initial_count = len(cdc_final)
        cdc_final = cdc_final.drop_duplicates(subset=['FIPS'], keep='first')
        print(f"   Removed {initial_count - len(cdc_final)} duplicate FIPS codes")
        
        print(f"   Final CDC records: {len(cdc_final):,}")
        print(f"   Asthma prevalence range: {cdc_final['Asthma_Prevalence'].min():.1f}% - {cdc_final['Asthma_Prevalence'].max():.1f}%")
        
        # Save processed data
        output_path = 'data/processed/cdc_asthma_prevalence.csv'
        cdc_final.to_csv(output_path, index=False)
        print(f"Processed CDC data saved to: {output_path}")
        
        return cdc_final
        
    except Exception as e:
        print(f"Error processing CDC data: {e}")
        import traceback
        traceback.print_exc()
        return None

def merge_datasets():
    """
    Merge EPA and CDC datasets for analysis
    """
    print("Merging EPA and CDC datasets...")
    
    # Load processed datasets
    epa_path = 'data/processed/epa_pm25_annual.csv'
    cdc_path = 'data/processed/cdc_asthma_prevalence.csv'
    
    if not os.path.exists(epa_path):
        print(f"EPA processed data not found: {epa_path}")
        return None
        
    if not os.path.exists(cdc_path):
        print(f"CDC processed data not found: {cdc_path}")
        return None
    
    try:
        epa_data = pd.read_csv(epa_path)
        cdc_data = pd.read_csv(cdc_path)
        
        print(f"   EPA data: {len(epa_data):,} counties")
        print(f"   CDC data: {len(cdc_data):,} counties")
        
        # Ensure FIPS codes are strings and properly formatted
        epa_data['FIPS'] = epa_data['FIPS'].astype(str).str.zfill(5)
        cdc_data['FIPS'] = cdc_data['FIPS'].astype(str).str.zfill(5)
        
        # Check for common FIPS codes
        epa_fips = set(epa_data['FIPS'])
        cdc_fips = set(cdc_data['FIPS'])
        common_fips = epa_fips.intersection(cdc_fips)
        
        print(f"   Common FIPS codes: {len(common_fips):,}")
        
        # Merge on FIPS code
        merged_data = pd.merge(
            epa_data, 
            cdc_data, 
            on='FIPS', 
            how='inner',
            suffixes=('_EPA', '_CDC')
        )
        
        print(f"   Merged dataset: {len(merged_data):,} counties")
        
        if len(merged_data) == 0:
            print("No matching counties found")
            return None
        
        # Data quality summary
        print(f"\nData Quality Summary:")
        print(f"   PM2.5 Mean: {merged_data['PM25_Annual_Mean'].mean():.2f} μg/m³")
        print(f"   PM2.5 Range: {merged_data['PM25_Annual_Mean'].min():.2f} - {merged_data['PM25_Annual_Mean'].max():.2f} μg/m³")
        print(f"   Asthma Mean: {merged_data['Asthma_Prevalence'].mean():.2f}%")
        print(f"   Asthma Range: {merged_data['Asthma_Prevalence'].min():.2f}% - {merged_data['Asthma_Prevalence'].max():.2f}%")
        
        # Save merged dataset
        output_path = 'data/processed/merged_pm25_asthma.csv'
        merged_data.to_csv(output_path, index=False)
        print(f"Merged dataset saved to: {output_path}")
        
        return merged_data
        
    except Exception as e:
        print(f"Error merging datasets: {e}")
        import traceback
        traceback.print_exc()
        return None

def fix_existing_data():
    """
    Fix the data that was already downloaded but failed processing
    """
    print_separator("FIXING EXISTING DATA")
    
    epa_file = 'data/raw/daily_88101_2023.csv'
    cdc_file = 'data/raw/cdc_places_2023.csv'
    
    success = True
    
    # Process EPA data with fixes
    if os.path.exists(epa_file):
        epa_data = process_epa_data_fixed(epa_file)
        if epa_data is None:
            success = False
    else:
        print("EPA file not found")
        success = False
    
    # Process CDC data with fixes
    if os.path.exists(cdc_file):
        cdc_data = process_cdc_data_fixed(cdc_file)
        if cdc_data is None:
            success = False
    else:
        print("CDC file not found")
        success = False
    
    if success:
        # Merge datasets
        merged_data = merge_datasets()
        if merged_data is not None:
            print_separator("DATA PROCESSING COMPLETE")
            print("Data processing successful!")
            print(f"   Final dataset: {len(merged_data):,} counties")
            print("   Files created:")
            print("     • data/processed/merged_pm25_asthma.csv")
            print("     • data/processed/epa_pm25_annual.csv")
            print("     • data/processed/cdc_asthma_prevalence.csv")
            print("\nReady for statistical analysis!")
            print("   Next: Create and run main_analysis.py")
            return True
    
    return False

if __name__ == "__main__":
    print("="*60)
    print("D502 CAPSTONE: DATA PROCESSING FIX")
    print("="*60)
    
    try:
        success = fix_existing_data()
        
        if success:
            print("\nData processing completed successfully!")
        else:
            print("\nData processing failed.")
            print("Please check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n\n Processing interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()