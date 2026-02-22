#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Compute China's share of publications per publisher to contextualize
retraction base rates.

Queries the Dimensions Analytics API for publications where
research_org_country_names includes "China" for each of the ten publishers,
then computes the ratio of China's retraction share to China's publication
share. A ratio of 1.0 means proportional representation; >1 means
overrepresented in retractions.

Uses the same authentication flow, publisher mapping, year range, and
publication types as 3-normalization.py.
"""

import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIMENSIONS_API_KEY = os.getenv('DIMENSIONS_API_KEY')
AUTH_URL = 'https://app.dimensions.ai/api/auth.json'
DSL_URL = 'https://app.dimensions.ai/api/dsl/v2'

FILTERED_CSV = 'filtered.csv'
TOTALS_CACHE = 'publisher_totals.json'
CACHE_FILE = 'china_publication_share.json'
OUTPUT_TEX = 'results/china-publication-share.tex'

YEAR_MIN = 1997
YEAR_MAX = 2026

DIMENSIONS_TYPES = ['article', 'proceeding']

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


def get_china_publication_count(token: str, publisher_names: list[str]) -> int:
    """Return the number of China-affiliated publications for a publisher."""
    names_str = ', '.join([f'"{n}"' for n in publisher_names])
    types_str = ', '.join([f'"{t}"' for t in DIMENSIONS_TYPES])
    query = (
        f'search publications where publisher in [{names_str}] '
        f'and research_org_country_names in ["China"] '
        f'and type in [{types_str}] '
        f'and year >= {YEAR_MIN} and year <= {YEAR_MAX} '
        f'return publications[doi] limit 1'
    )
    print(f'  Query: {query}')
    result = query_dimensions(token, query)
    total = result.get('_stats', {}).get('total_count', 0)
    print(f'  -> {total:,} China-affiliated publications')
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Step 1: Load total publication counts
    # ------------------------------------------------------------------
    if not os.path.exists(TOTALS_CACHE):
        print(f'Error: {TOTALS_CACHE} not found. Run 3-normalization.py first.')
        return
    with open(TOTALS_CACHE, 'r') as f:
        pub_totals = json.load(f)

    # ------------------------------------------------------------------
    # Step 2: Load retraction data and compute China retraction counts
    # ------------------------------------------------------------------
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')

    # Expand country field: each entry may list multiple countries separated by ";"
    retraction_counts_by_pub = {}
    china_retraction_counts = {}

    for label, info in PUBLISHERS.items():
        mask = df['Publisher'].isin(info['rw_names'])
        pub_df = df[mask]
        total = len(pub_df)
        china = pub_df['Country'].fillna('').str.contains('China', case=False).sum()
        retraction_counts_by_pub[label] = total
        china_retraction_counts[label] = china

    # Aggregate across all publishers
    total_retractions_all = len(df)
    china_retractions_all = df['Country'].fillna('').str.contains('China', case=False).sum()

    # ------------------------------------------------------------------
    # Step 3: Query Dimensions for China-affiliated publication counts
    # ------------------------------------------------------------------
    if os.path.exists(CACHE_FILE):
        print(f'Loading cached China publication counts from {CACHE_FILE}')
        with open(CACHE_FILE, 'r') as f:
            china_pubs = json.load(f)
    else:
        print('Authenticating with Dimensions API...')
        token = get_auth_token(DIMENSIONS_API_KEY)
        print('Authenticated successfully.\n')

        china_pubs = {}
        for label, info in PUBLISHERS.items():
            dim_names = info['dimensions_names']
            print(f'Querying China publications: {label}  ->  {dim_names}')
            count = get_china_publication_count(token, dim_names)
            china_pubs[label] = {
                'dimensions_names': dim_names,
                'china_publications': count,
            }
            time.sleep(2)
            print()

        # Also query aggregate (all 10 publishers combined)
        all_dim_names = []
        for info in PUBLISHERS.values():
            all_dim_names.extend(info['dimensions_names'])
        print(f'Querying China publications: ALL PUBLISHERS')
        names_str = ', '.join([f'"{n}"' for n in all_dim_names])
        types_str = ', '.join([f'"{t}"' for t in DIMENSIONS_TYPES])
        query = (
            f'search publications where publisher in [{names_str}] '
            f'and research_org_country_names in ["China"] '
            f'and type in [{types_str}] '
            f'and year >= {YEAR_MIN} and year <= {YEAR_MAX} '
            f'return publications[doi] limit 1'
        )
        print(f'  Query: {query}')
        result = query_dimensions(token, query)
        agg_total = result.get('_stats', {}).get('total_count', 0)
        print(f'  -> {agg_total:,} China-affiliated publications (aggregate)')
        china_pubs['_aggregate'] = {
            'china_publications': agg_total,
        }
        print()

        with open(CACHE_FILE, 'w') as f:
            json.dump(china_pubs, f, indent=2)
        print(f'Saved China publication counts to {CACHE_FILE}\n')

    # ------------------------------------------------------------------
    # Step 4: Compute ratios and print summary
    # ------------------------------------------------------------------
    rows = []
    for label in PUBLISHERS:
        total_pubs = pub_totals[label]['total_publications']
        china_pub_count = china_pubs[label]['china_publications']
        china_pub_pct = (china_pub_count / total_pubs * 100) if total_pubs > 0 else 0

        total_ret = retraction_counts_by_pub[label]
        china_ret = china_retraction_counts[label]
        china_ret_pct = (china_ret / total_ret * 100) if total_ret > 0 else 0

        ratio = (china_ret_pct / china_pub_pct) if china_pub_pct > 0 else float('inf')

        rows.append({
            'label': label,
            'total_publications': total_pubs,
            'china_publications': china_pub_count,
            'china_pub_pct': china_pub_pct,
            'total_retractions': total_ret,
            'china_retractions': china_ret,
            'china_ret_pct': china_ret_pct,
            'ratio': ratio,
        })

    # Aggregate row
    total_pubs_all = sum(pub_totals[l]['total_publications'] for l in PUBLISHERS)
    china_pub_all = china_pubs['_aggregate']['china_publications']
    china_pub_pct_all = (china_pub_all / total_pubs_all * 100) if total_pubs_all > 0 else 0
    china_ret_pct_all = (china_retractions_all / total_retractions_all * 100) if total_retractions_all > 0 else 0
    ratio_all = (china_ret_pct_all / china_pub_pct_all) if china_pub_pct_all > 0 else float('inf')

    # Print summary table
    print(f'{"Publisher":<20} {"China Pub %":>12} {"China Ret %":>12} {"Ratio":>8}')
    print('-' * 56)
    for r in sorted(rows, key=lambda x: -x['ratio']):
        print(f'{r["label"]:<20} {r["china_pub_pct"]:>11.1f}% {r["china_ret_pct"]:>11.1f}% {r["ratio"]:>8.2f}')
    print('-' * 56)
    print(f'{"Aggregate":<20} {china_pub_pct_all:>11.1f}% {china_ret_pct_all:>11.1f}% {ratio_all:>8.2f}')
    print()

    print(f'Aggregate: China accounts for {china_pub_pct_all:.1f}% of publications '
          f'and {china_ret_pct_all:.1f}% of retractions across these ten publishers.')
    print(f'Overrepresentation ratio: {ratio_all:.1f}x')

    # ------------------------------------------------------------------
    # Step 5: Generate LaTeX results file
    # ------------------------------------------------------------------
    os.makedirs('results', exist_ok=True)

    lines = []
    lines.append('% China publication share vs. retraction share per publisher')
    lines.append('% Generated by 13-china-publication-share.py')
    lines.append(f'% Data source: Dimensions API ({YEAR_MIN}-{YEAR_MAX}), Retraction Watch (filtered.csv)')
    lines.append('%')
    lines.append('% "China Pub %" = share of publications with at least one China-affiliated')
    lines.append('%   research organization, queried via research_org_country_names in Dimensions.')
    lines.append('% "China Ret %" = share of retraction entries listing China in the Country field.')
    lines.append('% "Ratio" = China Ret % / China Pub %. A value of 1.0 means proportional;')
    lines.append('%   >1 means overrepresented in retractions relative to publication output.')
    lines.append('')
    lines.append('\\begin{table}[!htbp]')
    lines.append('\\caption{China-affiliated publication share vs.\\ retraction share per publisher.')
    lines.append('Publication counts from Dimensions (1997--2026); retraction counts from Retraction Watch.')
    lines.append('The ratio divides the retraction share by the publication share; values above 1.0')
    lines.append('indicate overrepresentation in retractions relative to publication output.}')
    lines.append('\\label{tab:china-base-rate}')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append('\\begin{tabular}{lrrr}')
    lines.append('\\toprule')
    lines.append('Publisher & China Pub \\% & China Ret \\% & Ratio \\\\')
    lines.append('\\midrule')
    for r in sorted(rows, key=lambda x: -x['ratio']):
        lines.append(f'{r["label"]:<20} & {r["china_pub_pct"]:.1f}\\% & {r["china_ret_pct"]:.1f}\\% & {r["ratio"]:.2f} \\\\')
    lines.append('\\midrule')
    lines.append(f'\\textbf{{Aggregate}} & {china_pub_pct_all:.1f}\\% & {china_ret_pct_all:.1f}\\% & {ratio_all:.2f} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    with open(OUTPUT_TEX, 'w') as f:
        f.write('\n'.join(lines))
    print(f'LaTeX table written to {OUTPUT_TEX}')


if __name__ == '__main__':
    main()
