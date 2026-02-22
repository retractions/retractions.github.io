#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Cross-validate Dimensions publication counts against the Crossref REST API.

For each of the 10 publishers in publisher_totals.json, this script queries
the Crossref REST API for total journal-article and proceedings-article counts
(1997-2026) and compares them to the Dimensions totals. It computes per-publisher
percentage deviations, the Pearson correlation, and the mean absolute percentage
deviation (MAPD).

Crossref member IDs were identified via https://api.crossref.org/members?query=<name>.

Known caveats:
- Hindawi (member 98) was acquired by Wiley in January 2023. Post-acquisition
  DOIs may be registered under Wiley's member ID (311), so the Hindawi Crossref
  count will underestimate its Dimensions total.
- IOS Press (member 7437) publishes most content as book-series chapters.
  We include the book-chapter type in addition to journal-article and
  proceedings-article, mirroring the Dimensions approach.
- Taylor & Francis is registered as "Informa UK Limited" (member 301).
"""

import json
import os
import time

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIMENSIONS_CACHE = 'publisher_totals.json'
CROSSREF_CACHE = 'publisher_totals_crossref.json'
OUTPUT_TEX = 'results/validation-dimensions-crossref.tex'

MAILTO = 'joppenla@vt.edu'
CROSSREF_BASE = 'https://api.crossref.org'

YEAR_MIN = 1997
YEAR_MAX = 2026

# Crossref member IDs and document types for each publisher.
# Default types: journal-article, proceedings-article
# IOS Press additionally includes book-chapter (see docstring).
PUBLISHERS = {
    'ACM':              {'member_id': 320,  'types': ['journal-article', 'proceedings-article']},
    'IEEE':             {'member_id': 263,  'types': ['journal-article', 'proceedings-article']},
    'Elsevier':         {'member_id': 78,   'types': ['journal-article', 'proceedings-article']},
    'Springer Nature':  {'member_id': 297,  'types': ['journal-article', 'proceedings-article']},
    'Wiley':            {'member_id': 311,  'types': ['journal-article', 'proceedings-article']},
    'Taylor & Francis': {'member_id': 301,  'types': ['journal-article', 'proceedings-article']},
    'SAGE':             {'member_id': 179,  'types': ['journal-article', 'proceedings-article']},
    'Hindawi':          {'member_id': 98,   'types': ['journal-article', 'proceedings-article']},
    'IOS Press':        {'member_id': 7437, 'types': ['journal-article', 'proceedings-article', 'book-chapter']},
    'PLoS':             {'member_id': 340,  'types': ['journal-article', 'proceedings-article']},
}


# ---------------------------------------------------------------------------
# Crossref API helpers
# ---------------------------------------------------------------------------

def crossref_count(member_id: int, doc_type: str) -> int:
    """Return the total-results count for a member + type filter (rows=0)."""
    url = (
        f'{CROSSREF_BASE}/members/{member_id}/works'
        f'?filter=type:{doc_type},'
        f'from-pub-date:{YEAR_MIN},'
        f'until-pub-date:{YEAR_MAX}'
        f'&rows=0'
        f'&mailto={MAILTO}'
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f'  WARNING: HTTP {resp.status_code} for member {member_id}, type {doc_type}')
        print(f'  {resp.text[:300]}')
        return 0
    data = resp.json()
    return data['message']['total-results']


def get_publisher_total(label: str, info: dict) -> int:
    """Sum counts across all document types for a publisher."""
    total = 0
    for doc_type in info['types']:
        count = crossref_count(info['member_id'], doc_type)
        print(f'  {doc_type}: {count:,}')
        total += count
        time.sleep(1)  # polite rate limiting
    return total


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def generate_latex_table(rows: list, pearson_r: float, mapd: float) -> str:
    """Generate a LaTeX table comparing Dimensions and Crossref counts."""
    lines = []
    lines.append('% Cross-validation of Dimensions publication counts against Crossref REST API')
    lines.append('% Generated from: 3b-validate-normalization.py')
    lines.append('%')
    lines.append('% Crossref queries: /members/{id}/works?filter=type:{type},from-pub-date:1997,until-pub-date:2026&rows=0')
    lines.append('% Document types: journal-article + proceedings-article (+ book-chapter for IOS Press)')
    lines.append('')
    lines.append('\\begin{table}[htb]')
    lines.append('\\caption{Cross-validation of Dimensions publication counts against the Crossref REST API (1997--2026). ')
    lines.append('Both sources are filtered to journal articles and proceedings articles (IOS Press additionally includes book chapters). ')
    lines.append(f'Pearson~$r = {pearson_r:.4f}$; mean absolute percentage deviation = {mapd:.1f}\\%.}}')
    lines.append('\\label{tab:crossref-validation}')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append('\\begin{tabular}{lrrr}')
    lines.append('\\toprule')
    lines.append('Publisher & Dimensions & Crossref & Deviation (\\%) \\\\')
    lines.append('\\midrule')

    for row in rows:
        label = row['label'].replace('&', '\\&')
        dim = f'{row["dimensions"]:,}'
        cr = f'{row["crossref"]:,}'
        dev = f'{row["deviation"]:+.1f}'
        lines.append(f'{label:<20} & {dim:>12} & {cr:>12} & {dev:>6} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load Dimensions counts
    with open(DIMENSIONS_CACHE, 'r') as f:
        dim_data = json.load(f)

    # Query Crossref (or load cache)
    if os.path.exists(CROSSREF_CACHE):
        print(f'Loading cached Crossref counts from {CROSSREF_CACHE}\n')
        with open(CROSSREF_CACHE, 'r') as f:
            cr_data = json.load(f)
    else:
        print('Querying Crossref REST API...\n')
        cr_data = {}
        for label, info in PUBLISHERS.items():
            print(f'{label} (member {info["member_id"]}):')
            total = get_publisher_total(label, info)
            cr_data[label] = {
                'member_id': info['member_id'],
                'types': info['types'],
                'total_publications': total,
            }
            print(f'  TOTAL: {total:,}\n')
            time.sleep(1)

        with open(CROSSREF_CACHE, 'w') as f:
            json.dump(cr_data, f, indent=2)
        print(f'Saved Crossref counts to {CROSSREF_CACHE}\n')

    # Compare
    print(f'{"Publisher":<20} {"Dimensions":>14} {"Crossref":>14} {"Dev (%)":>10}')
    print('-' * 62)

    rows = []
    dim_vals = []
    cr_vals = []

    for label in PUBLISHERS:
        dim_count = dim_data[label]['total_publications']
        cr_count = cr_data[label]['total_publications']
        deviation = (cr_count - dim_count) / dim_count * 100 if dim_count > 0 else 0

        rows.append({
            'label': label,
            'dimensions': dim_count,
            'crossref': cr_count,
            'deviation': deviation,
        })
        dim_vals.append(dim_count)
        cr_vals.append(cr_count)

        print(f'{label:<20} {dim_count:>14,} {cr_count:>14,} {deviation:>+10.1f}')

    # Summary statistics
    pearson_r = float(np.corrcoef(dim_vals, cr_vals)[0, 1])
    abs_devs = [abs(r['deviation']) for r in rows]
    mapd = float(np.mean(abs_devs))
    max_dev_row = max(rows, key=lambda r: abs(r['deviation']))

    print()
    print(f'Pearson r:  {pearson_r:.4f}')
    print(f'MAPD:       {mapd:.1f}%')
    print(f'Max |dev|:  {abs(max_dev_row["deviation"]):.1f}% ({max_dev_row["label"]})')

    # Sort rows by Dimensions count descending for the table
    rows.sort(key=lambda r: r['dimensions'], reverse=True)

    # Generate LaTeX table
    tex = generate_latex_table(rows, pearson_r, mapd)
    os.makedirs(os.path.dirname(OUTPUT_TEX), exist_ok=True)
    with open(OUTPUT_TEX, 'w') as f:
        f.write(tex)
    print(f'\nLaTeX table saved to {OUTPUT_TEX}')


if __name__ == '__main__':
    main()
