#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import os, sys
import re
import sys
import json
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import seaborn as sns


# PREFIX = os.path.basename(__file__).split('-')[0] + '-' # Example: '20-'

colors = ['#eaeaea', '#ff2e63', '#08D9D6']  # Example neon colors
blue = '#1f77b4'
orange = '#ff7f0e'

AXIS_FONTSIZE = 16
TICK_FONTSIZE = 12
LABEL_FONTSIZE = 14

PUBLISHERS = {
    'ACM': ['Association for Computing Machinery (ACM)'],
    'IEEE': ['IEEE: Institute of Electrical and Electronics Engineers'],
    'Elsevier': ['Elsevier', 'Elsevier - Cell Press'],
    'Springer Nature': ['Springer', 'Springer - Nature Publishing Group', 'Springer - Biomed Central (BMC)'],
    'Wiley': ['Wiley'],
    'Taylor & Francis': ['Taylor and Francis', 'Taylor and Francis - Dove Press'],
    'SAGE': ['SAGE Publications'],
    'Hindawi': ['Hindawi'],
    'IOS Press': ['IOS Press (bought by Sage November 2023)'],
    'PLoS': ['PLoS'],
}

from plot_config import PUBLISHER_COLORS


def rewrite_category(cat):
  for k in substitutes.keys():
    if str(k) in str(cat): cat = cat.replace(str(k), substitutes[str(k)])
  return cat


def expand_countries(series):
    """Split semicolon-delimited country field, strip whitespace, drop blanks."""
    expanded = series.str.split(';', expand=True).stack().reset_index(level=1, drop=True)
    expanded = expanded.str.strip(' ;')
    return expanded[expanded != '']


inputfile = r'filtered.csv'

df = pd.read_csv(inputfile, encoding='iso-8859-1')

with open('substitutes.json', 'r') as f: substitutes = json.load(f)

df['Consolidated_Reason'] = df['Reason'].apply(rewrite_category)

# Global country statistics (across all publishers)
countries_expanded = expand_countries(df['Country'])
print(countries_expanded.describe())

print()
print(f"{len(countries_expanded)} total countries")
print(f"{len(set(countries_expanded))} unique countries")
print(f"{len(countries_expanded[countries_expanded == 'China'])} authors from China ({len(countries_expanded[countries_expanded == 'China']) / len(countries_expanded) * 100}%)")
print()


with open('countries_expanded_unique.json', 'w') as f: json.dump(list(set(countries_expanded)), f)
print(f'Written {len(list(set(countries_expanded)))} to countries_expanded_unique.json (including 1 \'Unknown\')')
num_countries = len(set(countries_expanded)) - 1
print(f"{num_countries} unique countries")
print()


top_country_list = countries_expanded.value_counts().head(10)
print(top_country_list)

# ---- Shared y-axis (same countries + same order for all subplots) -----------
TOPK_SHARED = 10
shared_countries = list(countries_expanded.value_counts().head(TOPK_SHARED).index)  # high -> low globally
shared_countries_rev = shared_countries[::-1]  # for barh so biggest appears at top

print()
for c,num in top_country_list.items():
	perc = num / len(countries_expanded) * 100
	print(c, f"{round(perc,2)}%")


#########################################################
# Figure 1: Absolute frequency — small multiples (2x4)
#########################################################

# fig, axes = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)

# for idx, (label, rw_name) in enumerate(PUBLISHERS.items()):
#     row, col = divmod(idx, 4)
#     ax = axes[row][col]
#     color = PUBLISHER_COLORS[label]

#     pub_df = df[df['Publisher'] == rw_name]
#     pub_countries = expand_countries(pub_df['Country'])
#     counts = pub_countries.value_counts()

#     top10 = counts.head(10)
#     x = top10.index[::-1]
#     y = top10.values[::-1]

#     ax.barh(x, y, color=color, alpha=0.9)
#     ax.set_title(label, fontsize=AXIS_FONTSIZE, fontweight='bold')
#     ax.tick_params(axis='both', labelsize=8)

#     if col == 0:
#         ax.set_ylabel('Author Country', fontsize=TICK_FONTSIZE)
#     if row == 1:
#         ax.set_xlabel('Abs. Frequency', fontsize=TICK_FONTSIZE)

# # Hide unused subplots
# for idx in range(len(PUBLISHERS), 2 * 4):
#     row, col = divmod(idx, 4)
#     axes[row][col].set_visible(False)

# fig.suptitle('Top 10 Author Countries per Publisher', fontsize=14, fontweight='bold', y=1.0)
# fig.tight_layout()
# plt.savefig('figures/author-top-countries-expanded.eps', format='eps', bbox_inches='tight')
# print('\nSaved figures/author-top-countries-expanded.eps')
# plt.show()
# plt.close()

#########################################################
# Figure 2: Relative frequency (%) — small multiples (2x4)
#########################################################

ncols = 5
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8), squeeze=False, sharey=True)

for idx, (label, rw_names) in enumerate(PUBLISHERS.items()):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    color = PUBLISHER_COLORS[label]

    pub_df = df[df['Publisher'].isin(rw_names)]
    pub_countries = expand_countries(pub_df['Country'])
    counts = pub_countries.value_counts()
    norm = len(pub_countries)

    # Align publisher counts to the shared country list so every subplot has the same y-axis
    aligned = counts.reindex(shared_countries).fillna(0).astype(int)
    x = shared_countries_rev
    y_abs = list(aligned.reindex(shared_countries_rev).values)
    y_pct = [val / norm * 100 if norm else 0 for val in y_abs]

    bars = ax.barh(x, y_pct, color=color, alpha=0.9)
    ax.set_title(label, fontsize=AXIS_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=LABEL_FONTSIZE)

    # Only show y tick labels on the leftmost column
    if col != 0:
        ax.tick_params(axis='y', labelleft=False, left=False)
    else:
        ax.tick_params(axis='y', labelleft=True, left=True)

    # Add Absolute Counts as text labels at the end of bars
    for bar, count in zip(bars, y_abs):
        if count <= 0:
            continue
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{count:,}",
            va='center', fontsize=TICK_FONTSIZE
        )

    ax.set_xlim(0, 105)
    # Expand x-axis limit so labels don't get cut off
    # ax.set_xlim(right=max(y_pct) * 1.15)

    formatter = FuncFormatter(lambda y, _: '{:.0f}'.format(y))
    ax.xaxis.set_major_formatter(formatter)

    if col == 0:
        ax.set_ylabel('Author Country', fontsize=AXIS_FONTSIZE)
    if row == nrows - 1:
        ax.set_xlabel('Rel. Frequency (%)', fontsize=AXIS_FONTSIZE)

# Hide unused subplots
for idx in range(len(PUBLISHERS), nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_visible(False)

# fig.suptitle('Top 10 Author Countries per Publisher (Relative)', fontsize=14, fontweight='bold', y=1.0)
fig.tight_layout()
plt.savefig('figures/author-countries-expanded-normalized.eps', format='eps', bbox_inches='tight')
print('Saved figures/author-countries-expanded-normalized.eps')
# plt.show()
plt.close()

#########################################################
