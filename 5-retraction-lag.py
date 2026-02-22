#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Compute retraction lag (time between original publication and retraction)
for each publisher. Generate box plot and violin plot figures, print summary
statistics, and write a results .tex file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from plot_config import PUBLISHER_COLORS

# PREFIX = os.path.basename(__file__).split('-')[0] + '-'

FILTERED_CSV = 'filtered.csv'
BOXPLOT_FIGURE = 'figures/retraction-lag-boxplot.eps'
VIOLIN_FIGURE = 'figures/retraction-lag-violin.eps'
RESULTS_TEX = 'results/retraction-lag.tex'

AXIS_FONTSIZE = 14
TICK_FONTSIZE = 10
LABEL_FONTSIZE = 12

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

# Reverse mapping: full name -> short label
PUBLISHER_LABELS = {}
for k, v_list in PUBLISHERS.items():
    for v in v_list:
        PUBLISHER_LABELS[v] = k


def main():
    # ---- Load and prepare data ------------------------------------------------
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')

    df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], errors='coerce')
    df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], errors='coerce')

    # Keep only rows with both dates present
    df = df.dropna(subset=['OriginalPaperDate', 'RetractionDate'])

    # Compute lag
    df['LagDays'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days
    df['LagYears'] = df['LagDays'] / 365.25

    # Drop negative lags (likely data errors)
    n_negative = (df['LagDays'] < 0).sum()
    if n_negative > 0:
        print(f'Dropping {n_negative} entries with negative retraction lag.')
    df = df[df['LagDays'] >= 0]

    # Map to short publisher labels, keep only known publishers
    df['PublisherLabel'] = df['Publisher'].map(PUBLISHER_LABELS)
    df = df.dropna(subset=['PublisherLabel'])

    # ---- Compute per-publisher statistics -------------------------------------
    stats_rows = []
    for label in PUBLISHERS.keys():
        sub = df[df['PublisherLabel'] == label]['LagDays']
        if len(sub) == 0:
            continue
        q1 = sub.quantile(0.25)
        q3 = sub.quantile(0.75)
        pct_under_1yr = (sub < 365.25).sum() / len(sub) * 100
        stats_rows.append({
            'Publisher': label,
            'N': len(sub),
            'Mean': sub.mean(),
            'Median': sub.median(),
            'Std': sub.std(),
            'Q1': q1,
            'Q3': q3,
            'Min': sub.min(),
            'Max': sub.max(),
            'PctUnder1yr': pct_under_1yr,
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df.sort_values('Median')

    # Print summary
    print('\nRetraction Lag Statistics (days)')
    print('=' * 100)
    print(f"{'Publisher':<20s} {'N':>6s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} "
          f"{'Q1':>8s} {'Q3':>8s} {'Min':>8s} {'Max':>8s} {'<1yr%':>7s}")
    print('-' * 100)
    for _, row in stats_df.iterrows():
        print(f"{row['Publisher']:<20s} {row['N']:>6.0f} {row['Mean']:>8.1f} "
              f"{row['Median']:>8.1f} {row['Std']:>8.1f} {row['Q1']:>8.1f} "
              f"{row['Q3']:>8.1f} {row['Min']:>8.1f} {row['Max']:>8.1f} "
              f"{row['PctUnder1yr']:>6.1f}%")
    print('=' * 100)

    # Sort order for plots (by median lag)
    publisher_order = list(stats_df['Publisher'])

    # ---- Box plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df, x='PublisherLabel', y='LagYears', order=publisher_order,
        hue='PublisherLabel', hue_order=publisher_order,
        palette=PUBLISHER_COLORS, showfliers=False, legend=False, ax=ax,
    )

    # Annotate medians
    for i, pub in enumerate(publisher_order):
        median_val = df[df['PublisherLabel'] == pub]['LagYears'].median()
        ax.annotate(
            f'{median_val:.2f}',
            xy=(i, median_val), xytext=(0, 8),
            textcoords='offset points', ha='center', va='bottom',
            fontsize=TICK_FONTSIZE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8),
        )

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.set_xlabel('Publisher', fontsize=AXIS_FONTSIZE)
    ax.set_ylabel('Retraction Lag (years, log scale)', fontsize=AXIS_FONTSIZE)
    # ax.set_title('Retraction Lag by Publisher', fontsize=AXIS_FONTSIZE + 2)
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE, rotation=20)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(BOXPLOT_FIGURE, format='eps', bbox_inches='tight')
    print(f'\nBox plot saved to {BOXPLOT_FIGURE}')
    plt.close()

    # ---- Violin plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=df, x='PublisherLabel', y='LagYears', order=publisher_order,
        hue='PublisherLabel', hue_order=publisher_order,
        palette=PUBLISHER_COLORS, inner='quartile', cut=0, legend=False, ax=ax,
    )

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.set_xlabel('Publisher', fontsize=AXIS_FONTSIZE)
    ax.set_ylabel('Retraction Lag (years, log scale)', fontsize=AXIS_FONTSIZE)
    # ax.set_title('Retraction Lag Distribution by Publisher', fontsize=AXIS_FONTSIZE + 2)
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE, rotation=20)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(VIOLIN_FIGURE, format='eps', bbox_inches='tight')
    print(f'Violin plot saved to {VIOLIN_FIGURE}')
    plt.close()

    # ---- Generate results .tex file -------------------------------------------
    write_results_tex(stats_df)
    print(f'Results written to {RESULTS_TEX}')


def write_results_tex(stats_df):
    """Write a LaTeX results file with the retraction lag statistics."""

    lines = []
    lines.append('% Retraction Lag Analysis')
    lines.append('% Generated by: 5-retraction-lag.py')
    lines.append('% Figures: figures/retraction-lag-boxplot.eps, figures/retraction-lag-violin.eps')
    lines.append('%')
    lines.append('% This file contains per-publisher retraction lag statistics (in days),')
    lines.append('% computed as RetractionDate - OriginalPaperDate from the Retraction Watch')
    lines.append('% database. Entries with missing dates or negative lag were excluded.')
    lines.append('')

    # Key findings as comments
    lines.append('% --- Key Findings ---')
    for _, row in stats_df.iterrows():
        median_years = row['Median'] / 365.25
        mean_years = row['Mean'] / 365.25
        lines.append(f"% {row['Publisher']}: N={row['N']:.0f}, "
                      f"median={row['Median']:.0f} days ({median_years:.2f} years), "
                      f"mean={row['Mean']:.0f} days ({mean_years:.2f} years), "
                      f"<1 year: {row['PctUnder1yr']:.1f}%")
    lines.append('%')

    # Comparison highlights
    ieee_row = stats_df[stats_df['Publisher'] == 'IEEE']
    acm_row = stats_df[stats_df['Publisher'] == 'ACM']
    if len(ieee_row) > 0 and len(acm_row) > 0:
        ieee_med = ieee_row.iloc[0]['Median']
        acm_med = acm_row.iloc[0]['Median']
        ratio = acm_med / ieee_med if ieee_med > 0 else float('inf')
        lines.append(f'% ACM vs. IEEE: ACM median lag ({acm_med:.0f} days) is '
                      f'{ratio:.1f}x longer than IEEE ({ieee_med:.0f} days).')
    lines.append('')

    # LaTeX table
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Retraction lag statistics by publisher (in days), '
                 'sorted by median lag.}')
    lines.append('  \\label{tab:retraction-lag}')
    lines.append('  \\begin{tabular}{l r r r r r r r r}')
    lines.append('    \\toprule')
    lines.append('    Publisher & N & Mean & Median & Std & Q1 & Q3 & Min & Max \\\\')
    lines.append('    \\midrule')

    for _, row in stats_df.iterrows():
        pub = row['Publisher'].replace('&', '\\&')
        lines.append(f"    {pub} & {row['N']:.0f} & {row['Mean']:.0f} & "
                      f"{row['Median']:.0f} & {row['Std']:.0f} & "
                      f"{row['Q1']:.0f} & {row['Q3']:.0f} & "
                      f"{row['Min']:.0f} & {row['Max']:.0f} \\\\")

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Commented-out figure blocks
    lines.append('% --- Figure: Box plot ---')
    lines.append('% \\begin{figure}[htbp]')
    lines.append('%   \\centering')
    lines.append('%   \\includegraphics[width=\\linewidth]{figures/retraction-lag-boxplot.eps}')
    lines.append('%   \\caption{Retraction lag (in years, log scale) by publisher. '
                 'Boxes show interquartile range; annotated values indicate the median. '
                 'Outliers are suppressed for readability.}')
    lines.append('%   \\label{fig:retraction-lag-boxplot}')
    lines.append('% \\end{figure}')
    lines.append('')
    lines.append('% --- Figure: Violin plot ---')
    lines.append('% \\begin{figure}[htbp]')
    lines.append('%   \\centering')
    lines.append('%   \\includegraphics[width=\\linewidth]{figures/retraction-lag-violin.eps}')
    lines.append('%   \\caption{Violin plot of retraction lag (in years, log scale) by publisher, '
                 'showing the full distribution shape. Inner lines indicate quartiles.}')
    lines.append('%   \\label{fig:retraction-lag-violin}')
    lines.append('% \\end{figure}')
    lines.append('')

    with open(RESULTS_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
