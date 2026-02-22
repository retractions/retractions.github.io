#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Analyse paywall status of retracted publications by publisher.

The Retraction Watch database records whether the retracted paper is still
behind a paywall (Yes / No / Unknown).  This analysis connects to the
paper's "dark archive" argument: if retracted papers remain paywalled,
readers may not even see the retraction notice.

Outputs:
  figures/paywall-status-by-publisher.eps
  results/paywall-status.tex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FILTERED_CSV = 'filtered.csv'
PAYWALL_PUB_FIGURE = 'figures/paywall-status-by-publisher.eps'
RESULTS_TEX = 'results/paywall-status.tex'

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

from plot_config import PAYWALL_COLORS

PUBLISHER_LABELS = {}
for k, v_list in PUBLISHERS.items():
    for v in v_list:
        PUBLISHER_LABELS[v] = k

PAYWALL_ORDER = ['Yes', 'No', 'Unknown']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    df['PublisherLabel'] = df['Publisher'].map(PUBLISHER_LABELS)
    df = df.dropna(subset=['PublisherLabel'])
    total = len(df)

    print(f'Total entries: {total:,}\n')

    # ---- 1. Global paywall status -------------------------------------------
    global_counts = df['Paywalled'].value_counts()
    print('=' * 70)
    print('GLOBAL PAYWALL STATUS')
    print('=' * 70)
    for val in PAYWALL_ORDER:
        n = global_counts.get(val, 0)
        print(f'  {val:<10s}  {n:>6,}  ({n/total*100:.1f}%)')
    print()

    # ---- 2. Per-publisher paywall status ------------------------------------
    print('=' * 70)
    print('PER-PUBLISHER PAYWALL STATUS')
    print('=' * 70)
    print(f'{"Publisher":<22s} {"Total":>7s} {"Yes":>7s} {"Yes%":>7s} '
          f'{"No":>7s} {"No%":>7s} {"Unk":>5s}')
    print('-' * 70)

    pub_rows = []
    for label in PUBLISHERS.keys():
        pub_df = df[df['PublisherLabel'] == label]
        n = len(pub_df)
        counts = pub_df['Paywalled'].value_counts()
        yes = counts.get('Yes', 0)
        no = counts.get('No', 0)
        unk = counts.get('Unknown', 0)
        pub_rows.append({
            'publisher': label,
            'total': n,
            'yes': yes, 'yes_pct': yes / n * 100 if n else 0,
            'no': no, 'no_pct': no / n * 100 if n else 0,
            'unknown': unk, 'unknown_pct': unk / n * 100 if n else 0,
        })
        print(f'{label:<22s} {n:>7,} {yes:>7,} {yes/n*100:>6.1f}% '
              f'{no:>7,} {no/n*100:>6.1f}% {unk:>5,}')
    print()

    pub_df_stats = pd.DataFrame(pub_rows)

    # ---- 3. ACM vs. IEEE deep comparison ------------------------------------
    print('=' * 70)
    print('ACM vs. IEEE PAYWALL STATUS (DETAIL)')
    print('=' * 70)

    for label in ['ACM', 'IEEE']:
        sub = df[df['PublisherLabel'] == label]
        n = len(sub)
        counts = sub['Paywalled'].value_counts()
        print(f'\n{label} (N={n:,}):')
        for val in PAYWALL_ORDER:
            c = counts.get(val, 0)
            print(f'  {val:<10s}  {c:>6,}  ({c/n*100:.1f}%)')

        # Paywall by retraction nature within this publisher
        print(f'  By retraction nature:')
        for nature in sub['RetractionNature'].unique():
            nat_sub = sub[sub['RetractionNature'] == nature]
            nn = len(nat_sub)
            nc = nat_sub['Paywalled'].value_counts()
            yes = nc.get('Yes', 0)
            no = nc.get('No', 0)
            print(f'    {nature:<25s}  N={nn:>5,}  '
                  f'Yes={yes:>4,} ({yes/nn*100:.1f}%)  '
                  f'No={no:>4,} ({no/nn*100:.1f}%)')
    print()

    # ---- 4. Temporal trend: paywall rate over time --------------------------
    df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], errors='coerce')
    df['RetYear'] = df['RetractionDate'].dt.year

    year_paywall = []
    for year in sorted(df['RetYear'].dropna().unique()):
        yr_df = df[df['RetYear'] == year]
        n = len(yr_df)
        if n < 10:
            continue
        yes = (yr_df['Paywalled'] == 'Yes').sum()
        year_paywall.append({
            'year': int(year),
            'total': n,
            'paywalled': yes,
            'pct': yes / n * 100,
        })

    if year_paywall:
        print('PAYWALL RATE OVER TIME (years with >= 10 entries)')
        print('-' * 50)
        for r in year_paywall:
            bar = '#' * int(r['pct'] / 2)
            print(f"  {r['year']}  {r['paywalled']:>5,}/{r['total']:>6,}  "
                  f"({r['pct']:>5.1f}%)  {bar}")
        print()

    year_paywall_df = pd.DataFrame(year_paywall)

    # ---- Figures ------------------------------------------------------------
    _plot_paywall_by_publisher(pub_df_stats)

    # ---- Results .tex -------------------------------------------------------
    write_results_tex(
        total, global_counts, pub_df_stats, year_paywall_df,
    )
    print(f'Results written to {RESULTS_TEX}')


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_paywall_by_publisher(pub_df: pd.DataFrame):
    """Stacked horizontal bar chart: paywall status per publisher."""
    pub_df = pub_df.sort_values('yes_pct', ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(pub_df))
    bar_h = 0.6

    # Stack: Yes, Unknown, No
    ax.barh(y, pub_df['yes_pct'], height=bar_h,
            color=PAYWALL_COLORS['Yes'], label='Paywalled',
            hatch='///', edgecolor='black', linewidth=0.5)
    ax.barh(y, pub_df['unknown_pct'], height=bar_h,
            left=pub_df['yes_pct'],
            color=PAYWALL_COLORS['Unknown'], label='Unknown',
            hatch='xxx', edgecolor='black', linewidth=0.5)
    ax.barh(y, pub_df['no_pct'], height=bar_h,
            left=pub_df['yes_pct'] + pub_df['unknown_pct'],
            color=PAYWALL_COLORS['No'], label='Not paywalled',
            hatch='', edgecolor='black', linewidth=0.5)

    # Annotate Yes% on bars where visible
    for i, row in enumerate(pub_df.itertuples()):
        if row.yes_pct >= 2:
            ax.text(row.yes_pct / 2, i, f'{row.yes_pct:.1f}%',
                    ha='center', va='center', fontsize=TICK_FONTSIZE,
                    color='white', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(pub_df['publisher'], fontsize=TICK_FONTSIZE)
    ax.set_xlabel('Percentage of retraction entries (%)', fontsize=AXIS_FONTSIZE)
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=LABEL_FONTSIZE, loc='lower right')

    plt.tight_layout()
    plt.savefig(PAYWALL_PUB_FIGURE, format='eps', bbox_inches='tight')
    print(f'Publisher paywall figure saved to {PAYWALL_PUB_FIGURE}')
    plt.close()



# ---------------------------------------------------------------------------
# Results .tex
# ---------------------------------------------------------------------------

def write_results_tex(total, global_counts, pub_df, year_df):
    """Write LaTeX results file with paywall status findings."""

    lines = []
    lines.append('% Paywall Status Analysis')
    lines.append('% Generated by: 8-paywall-status.py')
    lines.append('% Figures: figures/paywall-status-by-publisher.eps')
    lines.append('%')
    lines.append('% This file analyses the paywall status of retracted publications.')
    lines.append('% The Retraction Watch database records whether each retracted paper')
    lines.append('% is still behind a paywall (Yes / No / Unknown). This connects to')
    lines.append("% the paper's argument that ACM's \"dark archive\" may hide retracted")
    lines.append('% content from public scrutiny.')
    lines.append('')

    # Key findings as comments
    g_yes = global_counts.get('Yes', 0)
    g_no = global_counts.get('No', 0)
    g_unk = global_counts.get('Unknown', 0)
    lines.append('% --- Key Findings ---')
    lines.append(f'% Total entries: {total:,}')
    lines.append(f'% Paywalled (Yes): {g_yes:,} ({g_yes/total*100:.1f}%)')
    lines.append(f'% Not paywalled (No): {g_no:,} ({g_no/total*100:.1f}%)')
    lines.append(f'% Unknown: {g_unk:,} ({g_unk/total*100:.1f}%)')
    lines.append('%')

    # ACM and IEEE specific
    acm = pub_df[pub_df['publisher'] == 'ACM'].iloc[0]
    ieee = pub_df[pub_df['publisher'] == 'IEEE'].iloc[0]
    lines.append(f"% ACM: {acm['yes']:.0f}/{acm['total']:.0f} paywalled "
                 f"({acm['yes_pct']:.1f}%), "
                 f"{acm['no']:.0f} not paywalled ({acm['no_pct']:.1f}%)")
    lines.append(f"% IEEE: {ieee['yes']:.0f}/{ieee['total']:.0f} paywalled "
                 f"({ieee['yes_pct']:.1f}%), "
                 f"{ieee['no']:.0f} not paywalled ({ieee['no_pct']:.1f}%)")
    lines.append('%')

    # Highest/lowest paywall rates
    sorted_pub = pub_df.sort_values('yes_pct', ascending=False)
    top_pub = sorted_pub.iloc[0]
    bot_pub = sorted_pub.iloc[-1]
    lines.append(f"% Highest paywall rate: {top_pub['publisher']} "
                 f"({top_pub['yes_pct']:.1f}%)")
    lines.append(f"% Lowest paywall rate: {bot_pub['publisher']} "
                 f"({bot_pub['yes_pct']:.1f}%)")
    lines.append('')

    # Table 1: Global paywall status
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Paywall status of retracted publications across '
                 'all publishers.}')
    lines.append('  \\label{tab:paywall-global}')
    lines.append('  \\begin{tabular}{l r r}')
    lines.append('    \\toprule')
    lines.append('    Paywall status & Entries & \\% \\\\')
    lines.append('    \\midrule')
    for val in PAYWALL_ORDER:
        n = global_counts.get(val, 0)
        lines.append(f'    {val} & {n:,} & {n/total*100:.1f} \\\\')
    lines.append('    \\midrule')
    lines.append(f'    Total & {total:,} & 100.0 \\\\')
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Table 2: Per-publisher paywall status
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Paywall status of retracted publications by publisher. '
                 'Columns show the number and percentage of entries that remain '
                 'paywalled, are openly accessible, or have unknown status.}')
    lines.append('  \\label{tab:paywall-by-publisher}')
    lines.append('  \\begin{tabular}{l r r r r r r r}')
    lines.append('    \\toprule')
    lines.append('    Publisher & Total & Yes & Yes (\\%) & No & No (\\%) '
                 '& Unknown & Unk.~(\\%) \\\\')
    lines.append('    \\midrule')
    for _, row in pub_df.sort_values('yes_pct', ascending=False).iterrows():
        pub = row['publisher'].replace('&', '\\&')
        lines.append(
            f"    {pub} & {row['total']:,.0f} "
            f"& {row['yes']:,.0f} & {row['yes_pct']:.1f} "
            f"& {row['no']:,.0f} & {row['no_pct']:.1f} "
            f"& {row['unknown']:,.0f} & {row['unknown_pct']:.1f} \\\\"
        )
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Temporal trend as comments
    if len(year_df) > 0:
        lines.append('% --- Paywall rate over time ---')
        for _, row in year_df.iterrows():
            lines.append(f"% {row['year']:.0f}: "
                         f"{row['paywalled']:.0f}/{row['total']:.0f} "
                         f"({row['pct']:.1f}%)")
        lines.append('')

    # Commented-out figure blocks
    lines.append('% --- Figure: Paywall status by publisher ---')
    lines.append('% \\begin{figure}[htbp]')
    lines.append('%   \\centering')
    lines.append('%   \\includegraphics[width=\\linewidth]'
                 '{figures/paywall-status-by-publisher.eps}')
    lines.append('%   \\caption{Paywall status of retracted publications by '
                 'publisher. Red segments indicate entries that remain behind '
                 'a paywall; green segments indicate openly accessible entries.}')
    lines.append('%   \\label{fig:paywall-by-publisher}')
    lines.append('% \\end{figure}')
    lines.append('')
    with open(RESULTS_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
