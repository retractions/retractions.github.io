#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Normalize retraction counts by total publication volume per publisher.

Queries the Dimensions Analytics API for total publication counts (1997-2026)
for each publisher in the Retraction Watch dataset, then computes retraction
rates (per 10,000 publications) and generates a comparative bar chart.

Note: In Dimensions, "Springer" and "Springer - Nature Publishing Group"
(from Retraction Watch) are both indexed under "Springer Nature". We therefore
merge these two Retraction Watch categories into a single "Springer Nature"
entry for the normalization.

Both data sources are filtered to research articles only:
- Dimensions: type in ["article", "proceeding"] (excludes chapters, monographs, books)
- Retraction Watch: ArticleType must contain at least one research type token
  (Research Article, Conference Abstract/Paper, Review Article, Meta-Analysis,
  Clinical Study, Case Report, Preprint). Entries that are purely non-research
  (e.g., Retraction Notice, Commentary/Editorial, Letter) are excluded.
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()


# PREFIX = os.path.basename(__file__).split('-')[0] + '-' # example: '20-'


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIMENSIONS_API_KEY = os.getenv('DIMENSIONS_API_KEY')
AUTH_URL = 'https://app.dimensions.ai/api/auth.json'
DSL_URL = 'https://app.dimensions.ai/api/dsl/v2'

FILTERED_CSV = 'filtered.csv'
CACHE_FILE = 'publisher_totals.json'
OUTPUT_FIGURE = 'figures/retraction-rates-normalized.eps'
OUTPUT_TEX = 'results/normalized-retraction-counts-per-publisher.tex'

YEAR_MIN = 1997
YEAR_MAX = 2026

# Dimensions publication types that constitute research output.
# Excludes chapters, monographs, books, edited-books, etc.
DIMENSIONS_TYPES = ['article', 'proceeding']

# Retraction Watch ArticleType tokens that indicate a research contribution.
# Entries whose ArticleType contains at least one of these tokens are kept;
# entries that are purely non-research (e.g., "Retraction Notice;",
# "Commentary/Editorial;", "Letter;") are excluded.
RW_RESEARCH_TYPES = {
    'Research Article',
    'Conference Abstract/Paper',
    'Review Article',
    'Meta-Analysis',
    'Clinical Study',
    'Case Report',
    'Preprint',
}

# Each entry: display label -> (Dimensions publisher names to query, RW names to sum)
# Dimensions uses the `in` operator for publisher matching.
PUBLISHERS = {
    'ACM': {
        'dimensions_names': ['Association for Computing Machinery (ACM)'],
        'rw_names': ['Association for Computing Machinery (ACM)'],
    },
    'IEEE': {
        'dimensions_names': ['Institute of Electrical and Electronics Engineers (IEEE)'],
        'rw_names': ['IEEE: Institute of Electrical and Electronics Engineers'],
    },
    'Elsevier': {
        'dimensions_names': ['Elsevier'],
        'rw_names': ['Elsevier', 'Elsevier - Cell Press'],
    },
    'Springer Nature': {
        'dimensions_names': ['Springer Nature'],
        'rw_names': ['Springer', 'Springer - Nature Publishing Group', 'Springer - Biomed Central (BMC)'],
    },
    'Wiley': {
        'dimensions_names': ['Wiley'],
        'rw_names': ['Wiley'],
    },
    'Taylor & Francis': {
        'dimensions_names': ['Taylor & Francis'],
        'rw_names': ['Taylor and Francis', 'Taylor and Francis - Dove Press'],
    },
    'SAGE': {
        'dimensions_names': ['SAGE Publications'],
        'rw_names': ['SAGE Publications'],
    },
    'Hindawi': {
        'dimensions_names': ['Hindawi'],
        'rw_names': ['Hindawi'],
    },
    'IOS Press': {
        'dimensions_names': ['IOS Press'],
        'rw_names': ['IOS Press (bought by Sage November 2023)'],
    },
    'PLoS': {
        'dimensions_names': ['Public Library of Science (PLoS)'],
        'rw_names': ['PLoS'],
    },
}

# Plot styling (consistent with other scripts)
AXIS_FONTSIZE = 14
TICK_FONTSIZE = 10
LABEL_FONTSIZE = 12
from plot_config import ACM_COLOR, OTHER_COLOR

# ---------------------------------------------------------------------------
# Dimensions API helpers
# ---------------------------------------------------------------------------

def get_auth_token(api_key: str) -> str:
    """Authenticate with Dimensions and return a JWT token."""
    resp = requests.post(AUTH_URL, json={'key': api_key})
    resp.raise_for_status()
    return resp.json()['token']


def query_dimensions(token: str, dsl_query: str) -> dict:
    """Execute a DSL query against the Dimensions API."""
    headers = {'Authorization': f'JWT {token}'}
    resp = requests.post(DSL_URL, data=dsl_query, headers=headers)
    if resp.status_code != 200:
        print(f'  API error {resp.status_code}: {resp.text[:500]}')
        resp.raise_for_status()
    return resp.json()


def get_publication_count(token: str, publisher_names: list[str]) -> int:
    """Return the total number of publications for a publisher (1997-2026).

    Uses the DSL ``in`` operator to match one or more publisher name variants.
    """
    names_str = ', '.join([f'"{n}"' for n in publisher_names])
    types_str = ', '.join([f'"{t}"' for t in DIMENSIONS_TYPES])
    query = (
        f'search publications where publisher in [{names_str}] '
        f'and type in [{types_str}] '
        f'and year >= {YEAR_MIN} and year <= {YEAR_MAX} '
        f'return publications[doi] limit 1'
    )
    print(f'  Query: {query}')
    result = query_dimensions(token, query)
    total = result.get('_stats', {}).get('total_count', 0)
    print(f'  -> {total:,} publications')
    return total

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Step 1: Load retraction counts from filtered.csv
    # ------------------------------------------------------------------
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')

    # Compute paywall % per publisher using ALL entries (before filtering).
    all_entry_counts = df['Publisher'].value_counts().to_dict()
    paywall_counts = df[df['Paywalled'] == 'Yes']['Publisher'].value_counts().to_dict()

    # Filter to research articles only: keep rows where ArticleType contains
    # at least one research-type token.
    def is_research(article_type):
        if pd.isna(article_type):
            return False
        tokens = {t.strip() for t in article_type.split(';') if t.strip()}
        return bool(tokens & RW_RESEARCH_TYPES)

    n_before = len(df)
    df = df[df['ArticleType'].apply(is_research)]
    n_after = len(df)
    print(f'Retraction Watch: {n_before} total entries, {n_after} research articles '
          f'({n_before - n_after} non-research entries excluded)\n')

    retraction_counts = df['Publisher'].value_counts().to_dict()

    print('Retraction counts (research articles only):')
    for pub, count in sorted(retraction_counts.items(), key=lambda x: -x[1]):
        print(f'  {count:>6}  {pub}')
    print()

    # ------------------------------------------------------------------
    # Step 2: Query Dimensions for total publication counts (or load cache)
    # ------------------------------------------------------------------
    if os.path.exists(CACHE_FILE):
        print(f'Loading cached publication counts from {CACHE_FILE}')
        with open(CACHE_FILE, 'r') as f:
            pub_totals = json.load(f)
    else:
        print('Authenticating with Dimensions API...')
        token = get_auth_token(DIMENSIONS_API_KEY)
        print('Authenticated successfully.\n')

        pub_totals = {}
        for label, info in PUBLISHERS.items():
            dim_names = info['dimensions_names']
            print(f'Querying: {label}  ->  Dimensions: {dim_names}')
            count = get_publication_count(token, dim_names)
            pub_totals[label] = {
                'dimensions_names': dim_names,
                'total_publications': count,
            }
            time.sleep(2)  # conservative rate limiting
            print()

        with open(CACHE_FILE, 'w') as f:
            json.dump(pub_totals, f, indent=2)
        print(f'Saved publication counts to {CACHE_FILE}\n')

    # ------------------------------------------------------------------
    # Step 3: Compute retraction rates
    # ------------------------------------------------------------------
    rows = []
    for label, info in PUBLISHERS.items():
        retractions = sum(retraction_counts.get(rw, 0) for rw in info['rw_names'])
        all_entries = sum(all_entry_counts.get(rw, 0) for rw in info['rw_names'])
        paywalled = sum(paywall_counts.get(rw, 0) for rw in info['rw_names'])
        total_pubs = pub_totals[label]['total_publications']
        rate = (retractions / total_pubs * 10_000) if total_pubs > 0 else 0
        paywall_pct = (paywalled / all_entries * 100) if all_entries > 0 else 0
        rows.append({
            'label': label,
            'all_entries': all_entries,
            'retractions': retractions,
            'paywalled': paywalled,
            'paywall_pct': paywall_pct,
            'total_publications': total_pubs,
            'rate_per_10k': rate,
        })

    rates_df = pd.DataFrame(rows).sort_values('rate_per_10k', ascending=False)

    print('Normalized retraction rates (per 10,000 publications):')
    print('-' * 100)
    print(f'{"Publisher":<20} {"Entries":>8} {"Retract.":>10} {"Total Pubs":>14} '
          f'{"Rate/10k":>10} {"Paywall%":>10}')
    print('-' * 100)
    for _, row in rates_df.iterrows():
        print(f'{row["label"]:<20} {row["all_entries"]:>8,} {row["retractions"]:>10,} '
              f'{row["total_publications"]:>14,} {row["rate_per_10k"]:>10.2f} '
              f'{row["paywall_pct"]:>9.1f}%')
    print()

    # ------------------------------------------------------------------
    # Step 4: Generate bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [ACM_COLOR if lab == 'ACM' else OTHER_COLOR for lab in rates_df['label']]

    bars = ax.bar(
        rates_df['label'],
        rates_df['rate_per_10k'],
        color=colors,
        edgecolor='white',
        width=0.6,
    )

    # Add value labels on top of bars
    for bar, val in zip(bars, rates_df['rate_per_10k']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f'{val:.2f}',
            ha='center', va='bottom',
            fontsize=TICK_FONTSIZE,
        )

    ax.set_ylabel('Retractions per 10,000 Publications', fontsize=AXIS_FONTSIZE)
    ax.set_xlabel('Publisher', fontsize=AXIS_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.set_title('Retraction Rates Normalized by Publication Volume (1997\u20132026)')

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, format='eps', bbox_inches='tight')
    print(f'Figure saved to {OUTPUT_FIGURE}')
    # plt.show()
    plt.close()

    # ------------------------------------------------------------------
    # Step 5: Write results .tex file
    # ------------------------------------------------------------------
    write_results_tex(rates_df)
    print(f'Results table saved to {OUTPUT_TEX}')


def write_results_tex(rates_df: pd.DataFrame):
    """Write the normalized retraction rates table as a LaTeX file."""
    lines = []
    lines.append('% ----------------------------')
    lines.append('% Normalized retraction rates per publisher (research articles only)')
    lines.append('% Generated from: 3-normalization.py')
    lines.append('% Figure: figures/retraction-rates-normalized.eps')
    lines.append('%')
    lines.append('% Both data sources are filtered to research articles only:')
    lines.append('%   - Dimensions: type in [article, proceeding]')
    lines.append('%   - Retraction Watch: ArticleType contains at least one research-type token')
    lines.append('%     (Research Article, Conference Abstract/Paper, Review Article,')
    lines.append('%      Meta-Analysis, Clinical Study, Case Report, Preprint).')
    lines.append('%     Entries that are purely non-research (e.g., Retraction Notice,')
    lines.append('%     Commentary/Editorial, Letter) are excluded.')
    lines.append('% Sub-brands consolidated: BMC -> Springer Nature, Cell Press -> Elsevier,')
    lines.append('%   Dove Press -> Taylor & Francis.')
    lines.append('% ----------------------------')
    lines.append('')
    lines.append(r'\begin{table}[htb]')
    lines.append(r'\caption{Normalized retraction rates per publisher, filtered to research articles only. Retraction counts are drawn from the Retraction Watch database (1997--2026); total publication counts are drawn from the Dimensions database for the same period. Sub-brands are merged under parent companies. Paywall percentages are computed on all entry types.}')
    lines.append(r'\label{tab:normalized-retraction-rates}')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Publisher & Retractions & Total Publications & Rate per 10k & Paywalled (\%) \\')
    lines.append(r'\midrule')

    for _, row in rates_df.iterrows():
        label = row['label']
        if label == 'Taylor & Francis':
            label = r'Taylor \& Francis'
        retractions = f'{int(row["retractions"]):,}'
        total_pubs = f'{int(row["total_publications"]):,}'
        rate = f'{row["rate_per_10k"]:.2f}'
        paywall = f'{row["paywall_pct"]:.1f}'
        lines.append(f'{label:<20} & {retractions:>6} & {total_pubs:>10} & {rate:>6} & {paywall} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(OUTPUT_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
