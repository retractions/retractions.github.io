#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import os
import re
import sys
import json
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm


# PREFIX = '20-'
PREFIX = os.path.basename(__file__).split('-')[0] + '-'


dbfile = r'retractions - downloaded 2025-02-14.csv'
outfile = r'filtered.csv'

df = pd.read_csv(dbfile, encoding='iso-8859-1')

print(df.columns)

unique_publishers = df['Publisher'].unique()
filtered_publishers = [publ.lower() for publ in unique_publishers if pd.notna(publ)]

for publ in sorted(list(filtered_publishers)): print(publ)

# Major publishers to include in the analysis
# Top 9 by retraction count (with sub-brands) + ACM as focal publisher
publishers = [
	'Association for Computing Machinery (ACM)',
	'IEEE: Institute of Electrical and Electronics Engineers',
	'Elsevier',
	'Elsevier - Cell Press',
	'Springer',
	'Springer - Nature Publishing Group',
	'Springer - Biomed Central (BMC)',
	'Wiley',
	'Taylor and Francis',
	'Taylor and Francis - Dove Press',
	'SAGE Publications',
	'Hindawi',
	'IOS Press (bought by Sage November 2023)',
	'PLoS',
]

filter_df = df[df['Publisher'].isin(publishers)]

# Restrict to study window: 1997-2026 (ACM-DL launched October 1997)
filter_df['RetractionDate'] = pd.to_datetime(filter_df['RetractionDate'], errors='coerce')
filter_df = filter_df[filter_df['RetractionDate'] >= '1997-01-01']

print()
print(f"{len(filter_df.index)} total filtered entries (all types)")
print()

for pub in publishers:
	n = len(filter_df.loc[filter_df['Publisher'] == pub])
	print(f"{n:>6} {pub}")

# Save all-types data for retraction-nature analysis (Script 9)
all_types_file = 'filtered-all-types.csv'
filter_df.to_csv(all_types_file, index=False, encoding='iso-8859-1')
print()
print(f"Written {len(filter_df.index)} entries to {all_types_file}")

# Filter to retractions only
filter_df = filter_df[filter_df['RetractionNature'] == 'Retraction']

print()
print(f"{len(filter_df.index)} retractions only")
print()

for pub in publishers:
	n = len(filter_df.loc[filter_df['Publisher'] == pub])
	print(f"{n:>6} {pub}")

filter_df.to_csv(outfile, index=False, encoding='iso-8859-1')

print()
print(f"Written {len(filter_df.index)} retractions to {outfile}")
