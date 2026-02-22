#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Analyse the distribution of article types across retracted publications
by publisher.

The Retraction Watch ArticleType field is semicolon-delimited (an entry
may have multiple types).  This script expands the field, counts each
token per publisher, and produces a 2x5 panel figure analogous to the
author-countries figure from 2-fig-affiliations.py.

Outputs:
  figures/article-types-by-publisher.eps
  results/article-types.tex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

FILTERED_CSV = 'filtered.csv'
PANEL_FIGURE = 'figures/article-types-by-publisher.eps'
RESULTS_TEX = 'results/article-types.tex'

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

PUBLISHER_LABELS = {}
for k, v_list in PUBLISHERS.items():
    for v in v_list:
        PUBLISHER_LABELS[v] = k

# Article types shown individually; all others grouped as "Other".
TOP_TYPES = [
    'Research Article',
    'Conference Abstract/Paper',
    'Review Article',
    'Clinical Study',
    'Meta-Analysis',
    'Article in Press',
    'Letter',
    'Case Report',
    'Book Chapter/Reference Work',
    'Commentary/Editorial',
]


def expand_article_types(series: pd.Series) -> pd.Series:
    """Split semicolon-delimited ArticleType, strip whitespace, drop blanks."""
    expanded = series.str.split(';', expand=True).stack().reset_index(level=1, drop=True)
    expanded = expanded.str.strip(' ;')
    return expanded[expanded != '']


def main():
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    df['PublisherLabel'] = df['Publisher'].map(PUBLISHER_LABELS)
    df = df.dropna(subset=['PublisherLabel'])
    total = len(df)

    # ---- 1. Global article type distribution --------------------------------
    all_types = expand_article_types(df['ArticleType'])
    global_counts = all_types.value_counts()

    print('GLOBAL ARTICLE TYPE DISTRIBUTION')
    print('=' * 60)
    for t, n in global_counts.items():
        pct = n / total * 100
        print(f'  {n:>6,}  ({pct:>5.1f}%)  {t}')
    print()

    # ---- 2. Per-publisher breakdown -----------------------------------------
    print('PER-PUBLISHER TOP ARTICLE TYPES')
    print('=' * 70)

    pub_type_data = {}  # label -> {type: count}
    pub_totals = {}     # label -> total entries

    for label, rw_names in PUBLISHERS.items():
        pub_df = df[df['Publisher'].isin(rw_names)]
        n = len(pub_df)
        pub_totals[label] = n
        types = expand_article_types(pub_df['ArticleType'])
        counts = types.value_counts()
        pub_type_data[label] = counts

        print(f'\n{label} (N={n:,}):')
        for t, c in counts.head(8).items():
            print(f'  {c:>6,}  ({c/n*100:>5.1f}%)  {t}')

    print()

    # ---- 3. Panel figure (3x3) ----------------------------------------------
    top_set = set(TOP_TYPES)
    ncols = 5
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8), squeeze=False)

    # Build a consistent y-axis order from global counts across all publishers
    global_display = {}
    global_other = 0
    for t, c in global_counts.items():
        if t in top_set:
            global_display[t] = global_display.get(t, 0) + c
        else:
            global_other += c
    if global_other > 0:
        global_display['Other'] = global_other
    # Sorted ascending so highest count is at the top of the horizontal bar chart,
    # but force "Other" to the bottom (first position = bottom of barh)
    sorted_items = sorted(global_display.items(), key=lambda x: x[1])
    consistent_labels = [t for t, _ in sorted_items if t != 'Other']
    if 'Other' in global_display:
        consistent_labels.insert(0, 'Other')

    for idx, (label, rw_names) in enumerate(PUBLISHERS.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        color = PUBLISHER_COLORS[label]

        pub_df = df[df['Publisher'].isin(rw_names)]
        n_entries = len(pub_df)
        types = expand_article_types(pub_df['ArticleType'])
        counts = types.value_counts()

        # Consolidate minor types into "Other"
        display = {}
        other_count = 0
        for t, c in counts.items():
            if t in top_set:
                display[t] = c
            else:
                other_count += c
        if other_count > 0:
            display['Other'] = other_count

        # Use consistent label order; missing types get 0
        values = [display.get(t, 0) for t in consistent_labels]
        pct_values = [c / n_entries * 100 for c in values]

        bars = ax.barh(consistent_labels, pct_values, color=color, alpha=0.9)
        ax.set_title(label, fontsize=AXIS_FONTSIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)

        # Absolute count labels at end of bars (only for non-zero)
        for bar, count in zip(bars, values):
            if count > 0:
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f'{count:,}',
                    va='center', fontsize=TICK_FONTSIZE,
                )

        ax.set_xlim(0, 115)
        formatter = FuncFormatter(lambda y, _: f'{y:.0f}')
        ax.xaxis.set_major_formatter(formatter)

        # Only show y-tick labels on the leftmost column
        if col == 0:
            ax.set_ylabel('Article Type', fontsize=AXIS_FONTSIZE)
        else:
            ax.set_yticklabels([])
        if row == nrows - 1:
            ax.set_xlabel('Rel. Frequency (%)', fontsize=AXIS_FONTSIZE)

    # Hide unused subplots
    for idx in range(len(PUBLISHERS), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # fig.suptitle('Article Type Distribution per Publisher (Relative)',
    #              fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout()
    plt.savefig(PANEL_FIGURE, format='eps', bbox_inches='tight')
    print(f'Panel figure saved to {PANEL_FIGURE}')
    plt.close()

    # ---- 4. Write results .tex ----------------------------------------------
    write_results_tex(total, global_counts, pub_type_data, pub_totals)
    print(f'Results written to {RESULTS_TEX}')


def write_results_tex(total, global_counts, pub_type_data, pub_totals):
    """Write LaTeX results file with article type findings."""

    lines = []
    lines.append('% Article Type Analysis')
    lines.append('% Generated by: 10-article-types.py')
    lines.append('% Figures: figures/article-types-by-publisher.eps')
    lines.append('%')
    lines.append('% Distribution of article types across retracted publications')
    lines.append('% by publisher. The Retraction Watch ArticleType field is')
    lines.append('% semicolon-delimited; entries may carry multiple type labels.')
    lines.append('')

    # Key findings as comments
    lines.append('% --- Key Findings ---')
    lines.append(f'% Total entries: {total:,}')
    for t, n in global_counts.head(10).items():
        lines.append(f'% {t}: {n:,} ({n/total*100:.1f}%)')
    lines.append('%')

    # ACM / IEEE highlights
    for label in ['ACM', 'IEEE']:
        counts = pub_type_data[label]
        n = pub_totals[label]
        top3 = [(t, c, c/n*100) for t, c in counts.head(3).items()]
        parts = [f'{t} {c:,} ({p:.1f}%)' for t, c, p in top3]
        lines.append(f'% {label} (N={n:,}): ' + '; '.join(parts))
    lines.append('')

    # Table 1: Global article type distribution
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append(f'  \\caption{{Distribution of article types in the retraction dataset. '
                 f'An entry may carry multiple type labels; percentages are relative '
                 f'to the total number of entries (N={total:,}).}}')
    lines.append('  \\label{tab:article-type-global}')
    lines.append('  \\begin{tabular}{l r r}')
    lines.append('    \\toprule')
    lines.append('    Article type & Entries & \\% \\\\')
    lines.append('    \\midrule')
    for t, n in global_counts.items():
        t_esc = t.replace('&', '\\&')
        lines.append(f'    {t_esc} & {n:,} & {n/total*100:.1f} \\\\')
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Table 2: Per-publisher top 5 article types
    lines.append('\\begin{table*}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Top five article types per publisher (percentage '
                 'of entries within that publisher). '
                 'Conference Abstract/Paper abbreviated as Conf.\\ Paper.}')
    lines.append('  \\label{tab:article-type-by-publisher}')

    # Build a wide table: Publisher | #1 type (%) | #2 type (%) | ... | #5 type (%)
    lines.append('  \\begin{tabular}{l r l r l r l r l r}')
    lines.append('    \\toprule')
    lines.append('    Publisher '
                 '& \\multicolumn{2}{c}{1st} '
                 '& \\multicolumn{2}{c}{2nd} '
                 '& \\multicolumn{2}{c}{3rd} '
                 '& \\multicolumn{2}{c}{4th} '
                 '& \\multicolumn{2}{c}{5th} \\\\')
    lines.append('    \\midrule')

    short_names = {
        'Conference Abstract/Paper': 'Conf. Paper',
        'Research Article': 'Research Art.',
        'Review Article': 'Review Art.',
        'Clinical Study': 'Clinical St.',
        'Article in Press': 'Art. in Press',
        'Book Chapter/Reference Work': 'Book Ch./Ref.',
        'Commentary/Editorial': 'Commentary',
        'Correction/Erratum/Corrigendum': 'Correction',
        'Supplementary Materials': 'Suppl. Mat.',
        'Technical Report/White Paper': 'Tech. Report',
    }

    for label in PUBLISHERS.keys():
        counts = pub_type_data[label]
        n = pub_totals[label]
        top5 = list(counts.head(5).items())
        cells = [label.replace('&', '\\&')]
        for t, c in top5:
            short = short_names.get(t, t)
            short = short.replace('&', '\\&')
            pct = c / n * 100
            cells.append(f'{pct:.0f}\\%')
            cells.append(short)
        # Pad if fewer than 5
        while len(cells) < 11:
            cells.extend(['', ''])
        lines.append('    ' + ' & '.join(cells) + ' \\\\')

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table*}')
    lines.append('')

    # Commented-out figure block
    lines.append('% --- Figure: Article types by publisher (panel) ---')
    lines.append('% \\begin{figure*}[htbp]')
    lines.append('%   \\centering')
    lines.append('%   \\includegraphics[width=\\linewidth]'
                 '{figures/article-types-by-publisher.eps}')
    lines.append('%   \\caption{Distribution of article types per publisher. '
                 'Bars show relative frequency (\\%); absolute counts are '
                 'annotated at bar ends. Minor types are grouped as Other.}')
    lines.append('%   \\label{fig:article-types-by-publisher}')
    lines.append('% \\end{figure*}')
    lines.append('')

    with open(RESULTS_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
