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


inputfile = r'filtered.csv'

df = pd.read_csv(inputfile, encoding='iso-8859-1')

# Ensure 'RetractionDate' is in datetime format
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], errors='coerce', infer_datetime_format=True)

# oldest entry
oldest_date = df['RetractionDate'].min()
print(f"Oldest retraction date: {oldest_date}")

# newest entry
newest_date = df['RetractionDate'].max()
print(f"Newest retraction date: {newest_date}")


def plot_small_multiples(df, freq, log_scale, output_file, figsize=(20, 8)):
    """Plot a 2x4 grid of bar charts, one per publisher.

    Parameters
    ----------
    df : DataFrame with 'Publisher' and 'RetractionDate' columns.
    freq : 'M' for monthly, 'Q' for quarterly.
    log_scale : bool — use logarithmic y-axis.
    output_file : str — path for the saved PDF figure.
    figsize : tuple — overall figure size.
    """
    # Compute period counts per publisher and determine global date range
    period_counts = {}
    all_min, all_max = None, None
    for label, rw_names in PUBLISHERS.items():
        pub_df = df[df['Publisher'].isin(rw_names)]
        counts = pub_df['RetractionDate'].dt.to_period(freq).value_counts().sort_index()
        period_counts[label] = counts
        if len(counts) > 0:
            pmin, pmax = counts.index.min(), counts.index.max()
            all_min = pmin if all_min is None else min(all_min, pmin)
            all_max = pmax if all_max is None else max(all_max, pmax)

    # plot all x-values
    # full_range = pd.period_range(start=all_min, end=all_max, freq=freq)
    # instead, enforce start date of 1997 (ACM-DL launched October 1997)
    start_date = pd.Period('1997-01', freq=freq)
    # Ensure we don't cut off data if the max date is somehow older than 1997 (unlikely but safe)
    if all_max < start_date:
        print("Warning: Data ends before 1997 start date.")
        return
    # Create range from 1997 to the actual data end
    full_range = pd.period_range(start=start_date, end=all_max, freq=freq)


    # Reindex all publishers to the shared range
    for label in period_counts:
        period_counts[label] = period_counts[label].reindex(full_range, fill_value=0)

    # Calculate global max for BOTH linear and log
    global_ymax = max(s.max() for s in period_counts.values())

    ncols = 5
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8), squeeze=False)

    for idx, (label, counts) in enumerate(period_counts.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        color = PUBLISHER_COLORS[label]

        ax.bar(range(len(full_range)), counts.values, width=0.8, color=color, alpha=0.9)
        ax.set_title(label, fontsize=AXIS_FONTSIZE, fontweight='bold')

        # Target roughly 6-8 labels per plot, regardless of dataset length
        total_bars = len(full_range)
        target_labels = 7
        step = max(1, total_bars // target_labels)
        
        tick_positions = list(range(0, len(full_range), step))
        tick_labels = [full_range[i].start_time.strftime('%Y-%m') for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=TICK_FONTSIZE)

        if log_scale:
            ax.set_yscale('log')
            # Set log limits: bottom at 0.5 or 1 to avoid log(0) issues, top to global max
            ax.set_ylim(0.5, global_ymax * 2) 
        else:
            ax.set_ylim(0, global_ymax * 1.05)

        # Only show y-label on leftmost column
        if col == 0:
            ax.set_ylabel('Retractions', fontsize=AXIS_FONTSIZE)

    # Hide unused subplots (if any)
    for idx in range(len(PUBLISHERS), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    freq_label = 'Month' if freq == 'M' else 'Quarter'
    scale_label = ' (log)' if log_scale else ''
    # fig.suptitle(f'Retractions Per {freq_label}{scale_label}', fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout()
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f'Saved {output_file}')
    # plt.show()
    plt.close()


#########################################
# Figure 1: Monthly, linear
# plot_small_multiples(df, freq='M', log_scale=False,
#                      output_file='figures/entries-per-month.eps')

#########################################
# Figure 2: Quarterly, linear
# plot_small_multiples(df, freq='Q', log_scale=False,
#                      output_file='figures/entries-per-quarter.eps')

#########################################
# Figure 3: Quarterly, linear (non-log duplicate kept for compatibility)
# plot_small_multiples(df, freq='Q', log_scale=False,
#                      output_file='figures/entries-per-quarter-nonlog.eps')

#########################################
# Figure 4: Quarterly, log
plot_small_multiples(df, freq='Q', log_scale=True,
                     output_file='figures/entries-per-quarter-log.eps')

# #########################################
# # Figure 5: Quarterly, log (non-overlap duplicate kept for compatibility)
# plot_small_multiples(df, freq='Q', log_scale=True,
#                      output_file='figures/entries-per-quarter-log-nonoverlap.eps')
