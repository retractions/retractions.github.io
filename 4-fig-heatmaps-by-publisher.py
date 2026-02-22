#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Generate a panel of heatmaps (one per publisher) showing retraction reasons
by author country. The top N countries across all publishers are shown
individually; remaining countries are aggregated under "Other".

Reasons are consolidated using substitutes.json. The top M reasons across all
publishers are shown; the rest are dropped for readability.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

from plot_config import HEATMAP_CMAP

# PREFIX = os.path.basename(__file__).split('-')[0] + '-' # Example: '20-'

FILTERED_CSV = 'filtered.csv'
SUBSTITUTES_JSON = 'substitutes.json'
OUTPUT_FIGURE = 'figures/heatmap-panel-by-publisher.eps'

TOP_N_COUNTRIES = 8   # number of individual countries to show
TOP_M_REASONS = 10    # number of consolidated reasons to show

AXIS_FONTSIZE = 20
TICK_FONTSIZE = 16
LABEL_FONTSIZE = 18

# Abbreviated country labels for compact display
COUNTRY_ABBREV = {
    'China': 'China',
    'United States': 'US',
    'India': 'India',
    'Iran': 'Iran',
    'United Kingdom': 'UK',
    'Japan': 'Japan',
    'Saudi Arabia': 'Saudi Ar.',
    'Germany': 'Germany',
    'Other': 'Other',
}

# Abbreviated reason labels
REASON_ABBREV = {
    'Misconduct': 'Misconduct',
    'Results and/or Conclusions': 'Results/Concl.',
    'Plagiarism': 'Plagiarism',
    'Compromised Peer Review': 'Compr. Peer Rev.',
    'Third Party': 'Third Party',
    'Concerns/Issues about Data': 'Data Concerns',
    'Rogue Editor': 'Rogue Editor',
    'Removed': 'Removed',
    'Duplication of/in Image': 'Image Dupl.',
    'Author': 'Author Issues',
}

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_substitutes():
    with open(SUBSTITUTES_JSON, 'r') as f:
        return json.load(f)


def rewrite_category(cat, substitutes):
    if pd.isna(cat):
        return cat
    for k, v in substitutes.items():
        if k in str(cat):
            cat = cat.replace(k, v)
    return cat


def expand_semicolon_field(series):
    """Split a semicolon-delimited Series, strip whitespace, drop blanks."""
    expanded = series.str.split(';', expand=True).stack().reset_index(level=1, drop=True)
    expanded = expanded.str.strip(' ;+')
    return expanded[expanded != '']

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    substitutes = load_substitutes()

    # Consolidate reasons
    df['Consolidated_Reason'] = df['Reason'].apply(
        lambda x: rewrite_category(x, substitutes))

    # ---- Determine the global top countries (across all publishers) --------
    all_countries = expand_semicolon_field(df['Country'])
    global_top_countries = list(all_countries.value_counts().head(TOP_N_COUNTRIES).index)
    print(f'Global top {TOP_N_COUNTRIES} countries: {global_top_countries}')

    # ---- Determine the global top consolidated reasons ---------------------
    all_reasons = expand_semicolon_field(df['Consolidated_Reason'])
    # Filter out meta-reasons that are not substantive retraction causes
    meta_reasons = {
        'Investigation by Journal/Publisher', 'Information',
        'Date of Article and/or Notice Unknown', 'Notice',
        'Upgrade/Update of Prior Notice(s)',
        'Notice - Unable to Access via current resources',
    }
    all_reasons = all_reasons[~all_reasons.isin(meta_reasons)]
    global_top_reasons = list(all_reasons.value_counts().head(TOP_M_REASONS).index)
    print(f'Global top {TOP_M_REASONS} reasons: {global_top_reasons}\n')

    # ---- Build per-publisher crosstabs ------------------------------------
    crosstabs = {}
    for label, rw_names in PUBLISHERS.items():
        pub_df = df[df['Publisher'].isin(rw_names)].copy()
        if len(pub_df) == 0:
            continue

        # Expand countries: assign each row's countries to that row
        country_expanded = expand_semicolon_field(pub_df['Country'])
        # Map non-top countries to "Other"
        country_mapped = country_expanded.map(
            lambda c: c if c in global_top_countries else 'Other')

        # Expand reasons per row (need aligned index)
        reason_expanded = expand_semicolon_field(pub_df['Consolidated_Reason'])
        reason_expanded = reason_expanded[~reason_expanded.isin(meta_reasons)]

        # We need a cross of (country, reason) per original row.
        # Build an expanded DataFrame with one row per (record, country) and
        # one row per (record, reason), then join on the original index.
        country_df = country_mapped.to_frame('Country')
        reason_df = reason_expanded.to_frame('Reason')

        # Cross-join within each original row via the shared index
        merged = country_df.join(reason_df, how='inner')

        # Filter to top reasons only
        merged = merged[merged['Reason'].isin(global_top_reasons)]

        if len(merged) == 0:
            continue

        ct = pd.crosstab(merged['Reason'], merged['Country'])

        # Ensure all top countries + Other appear as columns
        for c in global_top_countries + ['Other']:
            if c not in ct.columns:
                ct[c] = 0
        ct = ct[global_top_countries + ['Other']]

        # Ensure all top reasons appear as rows
        for r in global_top_reasons:
            if r not in ct.index:
                ct.loc[r] = 0
        ct = ct.loc[[r for r in global_top_reasons if r in ct.index]]

        crosstabs[label] = ct
        print(f'{label}: {len(pub_df)} entries, crosstab shape {ct.shape}')

    # ---- Plot panel -------------------------------------------------------
    n_pubs = len(crosstabs)
    ncols = 5
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(28, 12), squeeze=False)

    # Compute a shared color scale across all panels
    all_vals = np.concatenate([ct.values.flatten() for ct in crosstabs.values()])
    vmin = 1
    vmax = max(all_vals.max(), 2)
    log_norm = LogNorm(vmin=vmin, vmax=vmax)

    for idx, (label, ct) in enumerate(crosstabs.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # Abbreviated labels
        col_labels = [COUNTRY_ABBREV.get(c, c) for c in ct.columns]
        row_labels = [REASON_ABBREV.get(r, r) for r in ct.index]

        # Plot with NaN for zeros so empty cells stay white
        plot_data = ct.copy()
        plot_data.columns = col_labels
        plot_data.index = row_labels

        sns.heatmap(
            plot_data.replace(0, np.nan),
            ax=ax,
            norm=log_norm,
            annot=ct.values,
            fmt='d',
            cmap=HEATMAP_CMAP,
            cbar=False,
            annot_kws={'size': TICK_FONTSIZE},
            linewidths=0.5,
            linecolor='white',
        )

        ax.set_title(label, fontsize=AXIS_FONTSIZE, fontweight='bold',
                     pad=6)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha='right',
                           fontsize=TICK_FONTSIZE)
        if col == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                               fontsize=TICK_FONTSIZE)
        else:
            ax.set_yticklabels([])

    # Hide unused subplot slots
    for idx in range(n_pubs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.subplots_adjust(hspace=0.35, wspace=0.08, right=0.92)

    # Force each subplot to be square
    for ax_row in axes:
        for ax in ax_row:
            ax.set_aspect('equal')

    # Add a shared colorbar in the remaining right margin
    sm = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP, norm=log_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.935, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Count (log scale)')

    # fig.suptitle('Retraction Reasons by Country per Publisher',
    #              fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_FIGURE, format='eps', bbox_inches='tight')
    print(f'\nFigure saved to {OUTPUT_FIGURE}')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    main()
