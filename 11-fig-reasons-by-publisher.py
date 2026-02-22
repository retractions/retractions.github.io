#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Panel bar chart of consolidated retraction reasons per publisher (2x5 layout),
analogous to the author-countries figure from 2-fig-affiliations.py.

Uses the same consolidation logic as 4-fig-heatmaps-by-publisher.py:
raw Retraction Watch reason tokens are mapped via substitutes.json, then
expanded (semicolon-split).  Meta-reasons (e.g., "Investigation by
Journal/Publisher", "Information") are excluded so the chart focuses on
substantive retraction causes.

Outputs:
  figures/reasons-by-publisher.eps
  results/reasons-by-publisher.tex
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

FILTERED_CSV = 'filtered.csv'
SUBSTITUTES_JSON = 'substitutes.json'
PANEL_FIGURE = 'figures/reasons-by-publisher.eps'
RESULTS_TEX = 'results/reasons-by-publisher.tex'

AXIS_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12

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

PUBLISHER_LABELS = {}
for k, v_list in PUBLISHERS.items():
    for v in v_list:
        PUBLISHER_LABELS[v] = k

# Meta-reasons to exclude (procedural, not substantive causes).
# Consistent with 4-fig-heatmaps-by-publisher.py.
META_REASONS = {
    'Investigation by Journal/Publisher',
    'Information',
    'Date of Article and/or Notice Unknown',
    'Notice',
    'Upgrade/Update of Prior Notice(s)',
    'Notice - Unable to Access via current resources',
}

SHORT_REASONS = {
    'Compromised Peer Review': 'Compr. Peer Rev.',
    'Results and/or Conclusions': 'Results/Concl.',
    'Concerns/Issues about Data': 'Data Concerns',
    'Concerns/Issues about Peer Review': 'Peer Rev. Concerns',
    'Concerns/Issues about Results and/or Conclusions': 'Results Concerns',
    'Duplication of/in Image': 'Image Dupl.',
    'Duplication of/in Article': 'Article Dupl.',
}


def rewrite_category(cat, substitutes):
    """Apply substring-based substitution to consolidate reason tokens."""
    if pd.isna(cat):
        return cat
    for k, v in substitutes.items():
        if k in str(cat):
            cat = cat.replace(k, v)
    return cat


def expand_semicolon_field(series):
    """Split semicolon-delimited Series, strip whitespace, drop blanks."""
    expanded = series.str.split(';', expand=True).stack().reset_index(level=1, drop=True)
    expanded = expanded.str.strip(' ;+')
    return expanded[expanded != '']


def main():
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    df['PublisherLabel'] = df['Publisher'].map(PUBLISHER_LABELS)
    df = df.dropna(subset=['PublisherLabel'])
    total = len(df)

    with open(SUBSTITUTES_JSON, 'r') as f:
        substitutes = json.load(f)

    df['Consolidated_Reason'] = df['Reason'].apply(
        lambda x: rewrite_category(x, substitutes))

    # ---- 1. Global reason distribution (excluding meta-reasons) -------------
    all_reasons = expand_semicolon_field(df['Consolidated_Reason'])
    all_reasons = all_reasons[~all_reasons.isin(META_REASONS)]
    # Deduplicate within each entry (an entry may list the same consolidated
    # reason multiple times after substitution)
    all_reasons = all_reasons.groupby(level=0).apply(
        lambda g: g.drop_duplicates()).droplevel(0)
    global_counts = all_reasons.value_counts()

    # ---- Shared y-axis (same reasons + same order for all subplots) -------------
    TOPK_SHARED = 10
    shared_reasons = list(global_counts.head(TOPK_SHARED).index)   # highest -> lowest globally
    shared_reasons_rev = shared_reasons[::-1]                      # for barh so highest appears at top

    print('GLOBAL CONSOLIDATED RETRACTION REASONS (excl. meta-reasons)')
    print('=' * 70)
    for t, n in global_counts.head(15).items():
        print(f'  {n:>6,}  ({n/total*100:>5.1f}%)  {t}')
    print(f'  ... {len(global_counts)} unique reasons total')
    print()

    # ---- 2. Per-publisher breakdown -----------------------------------------
    pub_reason_data = {}  # label -> Series of counts
    pub_totals = {}

    for label, rw_names in PUBLISHERS.items():
        pub_df = df[df['Publisher'].isin(rw_names)]
        n = len(pub_df)
        pub_totals[label] = n
        reasons = expand_semicolon_field(pub_df['Consolidated_Reason'])
        reasons = reasons[~reasons.isin(META_REASONS)]
        reasons = reasons.groupby(level=0).apply(
            lambda g: g.drop_duplicates()).droplevel(0)
        counts = reasons.value_counts()
        pub_reason_data[label] = counts

        print(f'{label} (N={n:,}):')
        for t, c in counts.head(5).items():
            print(f'  {c:>6,}  ({c/n*100:>5.1f}%)  {t}')
        print()

    # ---- 3. Panel figure (3x3) ----------------------------------------------
    ncols = 5
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8), squeeze=False, sharey=True)

    for idx, (label, rw_names) in enumerate(PUBLISHERS.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        color = PUBLISHER_COLORS[label]

        counts = pub_reason_data[label]
        n_entries = pub_totals[label]

        # Align publisher counts to the shared reason list so every subplot has the same y-axis
        aligned = counts.reindex(shared_reasons).fillna(0).astype(int)
        labels_list = shared_reasons_rev
        values = list(aligned.reindex(shared_reasons_rev).values)
        pct_values = [c / n_entries * 100 if n_entries else 0 for c in values]

        bars = ax.barh(labels_list, pct_values, color=color, alpha=0.9)
        ax.set_title(label, fontsize=AXIS_FONTSIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)

        # Only show y tick labels on the leftmost column
        if col != 0:
            ax.tick_params(axis='y', labelleft=False, left=False)
        else:
            ax.tick_params(axis='y', labelleft=True, left=True)

        # Absolute count labels at end of bars
        for bar, count in zip(bars, values):
            if count <= 0:
                continue
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f'{count:,}',
                va='center', fontsize=TICK_FONTSIZE,
            )

        ax.set_xlim(0, 115)
        formatter = FuncFormatter(lambda y, _: f'{y:.0f}')
        ax.xaxis.set_major_formatter(formatter)

        if col == 0:
            ax.set_ylabel('Retraction Reason', fontsize=AXIS_FONTSIZE)
        if row == nrows - 1:
            ax.set_xlabel('Rel. Frequency (%)', fontsize=AXIS_FONTSIZE)

    # Hide unused subplots
    for idx in range(len(PUBLISHERS), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # fig.suptitle('Consolidated Retraction Reasons per Publisher (Relative)',
    #              fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout()
    plt.savefig(PANEL_FIGURE, format='eps', bbox_inches='tight')
    print(f'Panel figure saved to {PANEL_FIGURE}')
    plt.close()

    # ---- 4. Write results .tex ----------------------------------------------
    write_results_tex(total, global_counts, pub_reason_data, pub_totals)
    print(f'Results written to {RESULTS_TEX}')


def write_results_tex(total, global_counts, pub_reason_data, pub_totals):
    """Write LaTeX results file with retraction reason findings."""

    lines = []
    lines.append('% Consolidated Retraction Reasons by Publisher')
    lines.append('% Generated by: 11-fig-reasons-by-publisher.py')
    lines.append('% Figures: figures/reasons-by-publisher.eps')
    lines.append('%')
    lines.append('% Distribution of consolidated retraction reasons per publisher.')
    lines.append('% Meta-reasons (Investigation by Journal/Publisher, Information,')
    lines.append('% Date Unknown, Notice, Upgrade/Update) are excluded to focus on')
    lines.append('% substantive retraction causes.')
    lines.append('')

    # Key findings as comments
    lines.append('% --- Key Findings ---')
    lines.append(f'% Total entries: {total:,}')
    lines.append(f'% Unique consolidated reasons (excl. meta): {len(global_counts)}')
    lines.append('%')
    lines.append('% Global top 10 reasons:')
    for t, n in global_counts.head(10).items():
        lines.append(f'%   {n:,} ({n/total*100:.1f}%)  {t}')
    lines.append('%')

    # Per-publisher highlights
    for label in ['ACM', 'IEEE']:
        counts = pub_reason_data[label]
        n = pub_totals[label]
        top3 = [(t, c, c/n*100) for t, c in counts.head(3).items()]
        parts = [f'{t} {c:,} ({p:.1f}%)' for t, c, p in top3]
        lines.append(f'% {label} (N={n:,}): ' + '; '.join(parts))
    lines.append('')

    # Table 1: Global reason distribution (top 15)
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append(f'  \\caption{{Top 15 consolidated retraction reasons across all '
                 f'publishers (N={total:,}). Meta-reasons excluded. '
                 f'Percentages are relative to total entries; an entry may '
                 f'list multiple reasons.}}')
    lines.append('  \\label{tab:reasons-global}')
    lines.append('  \\begin{tabular}{l r r}')
    lines.append('    \\toprule')
    lines.append('    Reason & Entries & \\% \\\\')
    lines.append('    \\midrule')
    for t, n in global_counts.head(15).items():
        t_esc = t.replace('&', '\\&')
        lines.append(f'    {t_esc} & {n:,} & {n/total*100:.1f} \\\\')
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Table 2: Per-publisher top 3 reasons
    lines.append('\\begin{table*}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Top three consolidated retraction reasons per publisher '
                 '(percentage of entries within that publisher). '
                 'Meta-reasons excluded. Reason names are abbreviated for space.}')
    lines.append('  \\label{tab:reasons-by-publisher-top3}')
    lines.append('  \\begin{tabular}{l r r l r l r l}')
    lines.append('    \\toprule')
    lines.append('    Publisher & N '
                 '& \\multicolumn{2}{c}{1st reason} '
                 '& \\multicolumn{2}{c}{2nd reason} '
                 '& \\multicolumn{2}{c}{3rd reason} \\\\')
    lines.append('    \\midrule')

    for label in PUBLISHERS.keys():
        counts = pub_reason_data[label]
        n = pub_totals[label]
        top3 = list(counts.head(3).items())
        cells = [label.replace('&', '\\&'), f'{n:,}']
        for t, c in top3:
            pct = c / n * 100
            short = SHORT_REASONS.get(t, t)
            short_esc = short.replace('&', '\\&')
            cells.append(f'{pct:.0f}\\%')
            cells.append(short_esc)
        while len(cells) < 8:
            cells.extend(['', ''])
        lines.append('    ' + ' & '.join(cells) + ' \\\\')

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table*}')
    lines.append('')

    # Commented-out figure block
    lines.append('% --- Figure: Reasons by publisher (panel) ---')
    lines.append('% \\begin{figure*}[htbp]')
    lines.append('%   \\centering')
    lines.append('%   \\includegraphics[width=\\linewidth]'
                 '{figures/reasons-by-publisher.eps}')
    lines.append('%   \\caption{Consolidated retraction reasons per publisher. '
                 'Bars show relative frequency (\\%); absolute counts are '
                 'annotated at bar ends. Meta-reasons (e.g., Investigation by '
                 'Journal/Publisher) are excluded.}')
    lines.append('%   \\label{fig:reasons-by-publisher}')
    lines.append('% \\end{figure*}')
    lines.append('')

    with open(RESULTS_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
