#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Analyse repeat authors in retracted publications using Dimensions
researcher_id for author disambiguation.

Requires author_ids_cache.json (produced by 7-fetch-author-ids.py).
Falls back to name+institution matching for authors without a
researcher_id, using Dimensions affiliations when available and the
Retraction Watch Institution field otherwise.

Outputs:
  figures/repeat-author-distribution.eps
  figures/repeat-authors-per-publisher.eps
  results/repeat-authors.tex
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter, defaultdict


FILTERED_CSV = 'filtered.csv'
CACHE_FILE = 'author_ids_cache.json'
SUBSTITUTES_FILE = 'substitutes.json'
DIST_FIGURE = 'figures/repeat-author-distribution.eps'
PUB_FIGURE = 'figures/repeat-authors-per-publisher.eps'
RESULTS_TEX = 'results/repeat-authors.tex'

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

from plot_config import PUBLISHER_COLORS, BIN_COLORS, ACM_COLOR

# Reverse mapping: full publisher name -> short label
PUBLISHER_LABELS = {}
for k, v_list in PUBLISHERS.items():
    for v in v_list:
        PUBLISHER_LABELS[v] = k


# ---------------------------------------------------------------------------
# Author identity resolution
# ---------------------------------------------------------------------------

def _normalize_institution(raw: str) -> str:
    """Normalize an institution string for use as a disambiguation key.

    Lowercases, strips whitespace, sorts semicolon-delimited parts,
    and filters out 'unavailable' or empty tokens.
    """
    if not raw or pd.isna(raw):
        return ''
    parts = [p.strip().lower() for p in str(raw).split(';')]
    parts = [p for p in parts if p and p != 'unavailable']
    return '|'.join(sorted(parts))


def resolve_authors(row, cache: dict) -> list[tuple[str, str]]:
    """Return list of (author_key, display_name) for a retraction entry.

    Uses a three-tier disambiguation strategy:
      1. Dimensions researcher_id (persistent identifier)
      2. Name + institution (Dimensions affiliations preferred,
         Retraction Watch Institution field as fallback)
      3. Name only (when no institution data is available)
    """
    doi = row.get('OriginalPaperDOI')
    rw_institution = _normalize_institution(row.get('Institution', ''))
    authors = []

    if pd.notna(doi) and str(doi).strip().lower() in cache:
        for a in cache[str(doi).strip().lower()]:
            rid = a.get('researcher_id')
            name = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
            if not name or name.lower() == 'unknown':
                continue
            if rid:
                authors.append((f'rid::{rid}', name))
            else:
                # Try Dimensions affiliations first
                dim_affils = a.get('affiliations', [])
                if dim_affils:
                    inst = '|'.join(sorted(
                        n.strip().lower() for n in dim_affils if n.strip()
                    ))
                    authors.append((f'name::{name}||inst::{inst}', name))
                elif rw_institution:
                    authors.append(
                        (f'name::{name}||inst::{rw_institution}', name))
                else:
                    authors.append((f'name::{name}||inst::', name))
    else:
        # Fall back to Retraction Watch Author field
        raw = row.get('Author', '')
        if pd.isna(raw) or not str(raw).strip():
            return []
        for name in str(raw).split(';'):
            name = name.strip()
            if not name or name.lower() == 'unknown':
                continue
            if rw_institution:
                authors.append(
                    (f'name::{name}||inst::{rw_institution}', name))
            else:
                authors.append((f'name::{name}||inst::', name))

    return authors


def load_reason_substitutes() -> dict:
    """Load the reason consolidation mapping from substitutes.json."""
    with open(SUBSTITUTES_FILE, 'r') as f:
        return json.load(f)


def consolidate_reason(token: str, subs: dict) -> str:
    """Map a raw Retraction Watch reason token to its consolidated category.

    Uses case-insensitive substring matching, consistent with
    4-fig-heatmaps-by-publisher.py.
    """
    token_stripped = token.strip().rstrip(';').strip()
    if not token_stripped:
        return ''
    token_lower = token_stripped.lower()
    for pattern, replacement in subs.items():
        if pattern.lower() in token_lower:
            return replacement if replacement else ''
    return token_stripped


def parse_reasons(reason_str, subs: dict) -> list[str]:
    """Parse semicolon-delimited reason string into consolidated tokens."""
    if pd.isna(reason_str) or not str(reason_str).strip():
        return []
    tokens = [t.strip().rstrip(';').strip() for t in str(reason_str).split(';')]
    consolidated = []
    for t in tokens:
        c = consolidate_reason(t, subs)
        if c and c not in consolidated:
            consolidated.append(c)
    return consolidated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Load data --------------------------------------------------------
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    df['PublisherLabel'] = df['Publisher'].map(PUBLISHER_LABELS)
    df = df.dropna(subset=['PublisherLabel'])

    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    print(f'Loaded {len(cache):,} DOIs from author cache')
    print(f'Dataset: {len(df):,} entries\n')

    # ---- Resolve authors per entry ----------------------------------------
    # Build: author_key -> list of (entry_index, publisher, display_name)
    author_entries = defaultdict(list)
    entries_with_authors = 0

    for idx, row in df.iterrows():
        authors = resolve_authors(row, cache)
        if authors:
            entries_with_authors += 1
        pub_label = row['PublisherLabel']
        for key, name in authors:
            author_entries[key].append({
                'idx': idx,
                'publisher': pub_label,
                'name': name,
            })

    n_unique = len(author_entries)
    n_rid = sum(1 for k in author_entries if k.startswith('rid::'))
    n_name_inst = sum(1 for k in author_entries
                      if k.startswith('name::') and '||inst::' in k
                      and not k.endswith('||inst::'))
    n_name_only = n_unique - n_rid - n_name_inst
    print(f'Entries with resolved authors: {entries_with_authors:,}')
    print(f'Unique author identifiers:     {n_unique:,}')
    print(f'  By researcher_id:            {n_rid:,} ({n_rid/n_unique*100:.1f}%)')
    print(f'  By name + institution:       {n_name_inst:,} ({n_name_inst/n_unique*100:.1f}%)')
    print(f'  By name only:                {n_name_only:,} ({n_name_only/n_unique*100:.1f}%)')
    print()

    # ---- 2a. Global repeat-author statistics --------------------------------
    retraction_counts = {k: len(v) for k, v in author_entries.items()}
    count_series = pd.Series(retraction_counts)

    print('=' * 70)
    print('GLOBAL REPEAT-AUTHOR DISTRIBUTION')
    print('=' * 70)
    thresholds = [1, 2, 3, 5, 10, 20, 50]
    dist_rows = []
    for t in thresholds:
        n = (count_series >= t).sum()
        pct = n / n_unique * 100
        label = f'{t}+' if t > 1 else 'exactly 1'
        if t == 1:
            n = (count_series == 1).sum()
            pct = n / n_unique * 100
        print(f'  Authors with {label:>12s} retractions: {n:>7,} ({pct:.2f}%)')
        dist_rows.append({'threshold': label, 'count': n, 'pct': pct})
    print()

    # Top 20 most-retracted authors
    top20 = count_series.nlargest(20)
    print('TOP 20 MOST-RETRACTED AUTHORS')
    print('-' * 70)
    print(f'{"Rank":<6} {"Retractions":>12} {"Name":<35} {"Publishers"}')
    print('-' * 70)
    top20_rows = []
    for rank, (key, n_ret) in enumerate(top20.items(), 1):
        entries = author_entries[key]
        name = entries[0]['name']
        pubs = sorted(set(e['publisher'] for e in entries))
        pubs_str = ', '.join(pubs)
        print(f'{rank:<6} {n_ret:>12,} {name:<35} {pubs_str}')
        top20_rows.append({
            'rank': rank, 'retractions': n_ret,
            'name': name, 'publishers': pubs_str, 'key': key,
        })
    print()

    # ---- 2b. Per-publisher repeat-author breakdown --------------------------
    print('=' * 70)
    print('PER-PUBLISHER REPEAT-AUTHOR BREAKDOWN')
    print('=' * 70)

    pub_stats = []
    for label in PUBLISHERS.keys():
        # Authors appearing in this publisher
        pub_authors = {}
        for key, entries in author_entries.items():
            pub_count = sum(1 for e in entries if e['publisher'] == label)
            if pub_count > 0:
                pub_authors[key] = pub_count

        n_authors = len(pub_authors)
        if n_authors == 0:
            pub_stats.append({
                'publisher': label, 'n_authors': 0,
                'n_2plus': 0, 'pct_2plus': 0,
                'n_5plus': 0, 'pct_5plus': 0,
                'mean': 0, 'median': 0,
            })
            continue

        counts = pd.Series(pub_authors)
        n_2plus = (counts >= 2).sum()
        n_5plus = (counts >= 5).sum()
        pub_stats.append({
            'publisher': label,
            'n_authors': n_authors,
            'n_2plus': n_2plus,
            'pct_2plus': n_2plus / n_authors * 100,
            'n_5plus': n_5plus,
            'pct_5plus': n_5plus / n_authors * 100,
            'mean': counts.mean(),
            'median': counts.median(),
        })
        print(f'{label:<22s}  authors={n_authors:>6,}  '
              f'2+={n_2plus:>5,} ({n_2plus/n_authors*100:>5.1f}%)  '
              f'5+={n_5plus:>5,} ({n_5plus/n_authors*100:>5.1f}%)  '
              f'mean={counts.mean():.2f}  median={counts.median():.1f}')
    print()

    pub_stats_df = pd.DataFrame(pub_stats)

    # ---- 2c. Cross-publisher repeat offenders --------------------------------
    print('=' * 70)
    print('CROSS-PUBLISHER REPEAT OFFENDERS')
    print('=' * 70)

    cross_pub = {}
    for key, entries in author_entries.items():
        pubs = set(e['publisher'] for e in entries)
        if len(pubs) >= 2:
            cross_pub[key] = {
                'n_publishers': len(pubs),
                'publishers': sorted(pubs),
                'total_retractions': len(entries),
                'name': entries[0]['name'],
            }

    n_cross = len(cross_pub)
    n_cross_5plus = sum(1 for v in cross_pub.values() if v['total_retractions'] >= 5)
    print(f'Authors in 2+ publishers:             {n_cross:,}')
    print(f'Of those, with 5+ total retractions:  {n_cross_5plus:,}')
    print()

    # Top cross-publisher offenders
    cross_sorted = sorted(cross_pub.items(),
                          key=lambda x: x[1]['total_retractions'], reverse=True)
    print('Top 10 cross-publisher repeat offenders:')
    for key, info in cross_sorted[:10]:
        print(f"  {info['total_retractions']:>4} retractions  "
              f"{info['name']:<30s}  {', '.join(info['publishers'])}")
    print()

    # ---- 2d. Prolific offenders (>= 50 retractions) with reason profiles ----
    subs = load_reason_substitutes()
    prolific_threshold = 50
    prolific_keys = [k for k, v in retraction_counts.items()
                     if v >= prolific_threshold]
    prolific_keys.sort(key=lambda k: retraction_counts[k], reverse=True)

    print('=' * 70)
    print(f'PROLIFIC OFFENDERS (>= {prolific_threshold} RETRACTIONS) — REASON PROFILES')
    print('=' * 70)
    print(f'Number of authors with >= {prolific_threshold} retractions: {len(prolific_keys)}\n')

    prolific_profiles = []
    for key in prolific_keys:
        entries = author_entries[key]
        n_ret = len(entries)
        name = entries[0]['name']
        pubs = sorted(set(e['publisher'] for e in entries))

        # Extract Dimensions ID if available
        dim_id = key.split('::', 1)[1] if key.startswith('rid::') else None
        id_type = 'researcher_id' if dim_id else 'name-based'

        # Collect consolidated reasons across all retraction entries
        reason_counter = Counter()
        for e in entries:
            reason_str = df.loc[e['idx'], 'Reason']
            reasons = parse_reasons(reason_str, subs)
            for r in reasons:
                reason_counter[r] += 1

        # Sort reasons by frequency
        reason_list = reason_counter.most_common()

        print(f'{name}  ({n_ret} retractions)')
        print(f'  ID: {key}  (type: {id_type})')
        print(f'  Publishers: {", ".join(pubs)}')
        print(f'  Retraction reasons:')
        for reason, count in reason_list:
            pct = count / n_ret * 100
            print(f'    {count:>4} ({pct:>5.1f}%)  {reason}')
        print()

        prolific_profiles.append({
            'key': key,
            'name': name,
            'dim_id': dim_id,
            'id_type': id_type,
            'n_retractions': n_ret,
            'publishers': ', '.join(pubs),
            'reasons': reason_list,
        })

    # ---- Figures ------------------------------------------------------------
    _plot_distribution(count_series)
    _plot_per_publisher(author_entries)

    # ---- Results .tex -------------------------------------------------------
    write_results_tex(
        n_unique, n_rid, n_name_inst, n_name_only, dist_rows, top20_rows,
        pub_stats_df, n_cross, n_cross_5plus, count_series,
        prolific_profiles,
    )

    print(f'Results written to {RESULTS_TEX}')


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_distribution(count_series: pd.Series):
    """Histogram of retractions-per-author (log y-axis)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    max_val = int(count_series.max())
    bins = np.arange(1, min(max_val + 2, 102))  # cap at 100 for readability

    ax.hist(count_series.clip(upper=100), bins=bins, color=ACM_COLOR,
            edgecolor='white', linewidth=0.3)
    ax.set_yscale('log')
    ax.set_xlabel('Number of retractions per author', fontsize=AXIS_FONTSIZE)
    ax.set_ylabel('Number of authors (log scale)', fontsize=AXIS_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.set_xlim(0.5, 100.5)

    plt.tight_layout()
    plt.savefig(DIST_FIGURE, format='eps', bbox_inches='tight')
    print(f'Distribution figure saved to {DIST_FIGURE}')
    plt.close()


def _plot_per_publisher(author_entries: dict):
    """Grouped bar chart: per publisher, % of authors in retraction-count bins."""
    bins_def = [(1, 1, '1'), (2, 4, '2–4'), (5, 9, '5–9'), (10, 9999, '10+')]
    bin_labels = [b[2] for b in bins_def]

    pub_labels = list(PUBLISHERS.keys())
    data = {bl: [] for bl in bin_labels}

    for label in pub_labels:
        # Count retractions per author within this publisher
        pub_counts = Counter()
        for key, entries in author_entries.items():
            c = sum(1 for e in entries if e['publisher'] == label)
            if c > 0:
                pub_counts[key] = c

        total = len(pub_counts)
        for lo, hi, bl in bins_def:
            n = sum(1 for c in pub_counts.values() if lo <= c <= hi)
            pct = n / total * 100 if total > 0 else 0
            data[bl].append(pct)

    x = np.arange(len(pub_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))

    hatches = ['', '///', '\\\\\\', 'xxx']
    for i, bl in enumerate(bin_labels):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[bl], width, label=bl, color=BIN_COLORS[i],
               hatch=hatches[i], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Percentage of authors (%)', fontsize=AXIS_FONTSIZE)
    ax.set_xlabel('Publisher', fontsize=AXIS_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(pub_labels, rotation=30, ha='right', fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.legend(title='Retractions', fontsize=LABEL_FONTSIZE,
              title_fontsize=LABEL_FONTSIZE)

    plt.tight_layout()
    plt.savefig(PUB_FIGURE, format='eps', bbox_inches='tight')
    print(f'Per-publisher figure saved to {PUB_FIGURE}')
    plt.close()


# ---------------------------------------------------------------------------
# Results .tex
# ---------------------------------------------------------------------------

def write_results_tex(
    n_unique, n_rid, n_name_inst, n_name_only, dist_rows, top20_rows,
    pub_stats_df, n_cross, n_cross_5plus, count_series,
    prolific_profiles,
):
    """Write LaTeX results file with repeat-author analysis findings."""

    lines = []
    lines.append('% Repeat-Author Analysis')
    lines.append('% Generated by: 7-repeat-authors.py')
    lines.append('% Figures: figures/repeat-author-distribution.eps, '
                 'figures/repeat-authors-per-publisher.eps')
    lines.append('%')
    lines.append('% This file analyses how many authors have multiple retracted')
    lines.append('% papers and whether repeat offenders are more common in certain')
    lines.append('% publishers. Author disambiguation uses Dimensions researcher_id')
    lines.append('% where available, then name+institution, then name-only fallback.')
    lines.append('')

    # Key findings as comments
    lines.append('% --- Key Findings ---')
    lines.append(f'% Unique author identifiers: {n_unique:,}')
    lines.append(f'% Disambiguated by researcher_id: {n_rid:,} ({n_rid/n_unique*100:.1f}%)')
    lines.append(f'% Disambiguated by name + institution: {n_name_inst:,} ({n_name_inst/n_unique*100:.1f}%)')
    lines.append(f'% Name-only fallback: {n_name_only:,} ({n_name_only/n_unique*100:.1f}%)')
    lines.append(f'%')
    for r in dist_rows:
        lines.append(f"% Authors with {r['threshold']} retractions: "
                     f"{r['count']:,} ({r['pct']:.2f}%)")
    lines.append(f'%')
    lines.append(f'% Cross-publisher authors (2+ publishers): {n_cross:,}')
    lines.append(f'% Cross-publisher with 5+ total retractions: {n_cross_5plus:,}')
    lines.append(f'%')
    lines.append(f'% Mean retractions per author: {count_series.mean():.2f}')
    lines.append(f'% Median retractions per author: {count_series.median():.1f}')
    lines.append(f'% Max retractions by a single author: {count_series.max():,}')
    lines.append('')

    # Table 1: Global distribution
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Distribution of retractions per author across all publishers.}')
    lines.append('  \\label{tab:repeat-author-distribution}')
    lines.append('  \\begin{tabular}{l r r}')
    lines.append('    \\toprule')
    lines.append('    Retractions & Authors & \\% \\\\')
    lines.append('    \\midrule')
    for r in dist_rows:
        lines.append(f"    {r['threshold']} & {r['count']:,} & {r['pct']:.2f} \\\\")
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Table 2: Top 20 most-retracted authors
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Top 20 most-retracted authors by number of retraction entries.}')
    lines.append('  \\label{tab:top-repeat-authors}')
    lines.append('  \\begin{tabular}{r r l l}')
    lines.append('    \\toprule')
    lines.append('    Rank & Retractions & Author & Publishers \\\\')
    lines.append('    \\midrule')
    for r in top20_rows:
        name_escaped = r['name'].replace('&', '\\&')
        pubs_escaped = r['publishers'].replace('&', '\\&')
        lines.append(f"    {r['rank']} & {r['retractions']:,} "
                     f"& {name_escaped} & {pubs_escaped} \\\\")
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Table 3: Per-publisher breakdown
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Per-publisher repeat-author statistics. '
                 'Columns show the number of unique authors, the count and percentage '
                 'with 2+ and 5+ retractions within that publisher, '
                 'and the mean/median retractions per author.}')
    lines.append('  \\label{tab:repeat-authors-per-publisher}')
    lines.append('  \\begin{tabular}{l r r r r r r r}')
    lines.append('    \\toprule')
    lines.append('    Publisher & Authors & 2+ & 2+ (\\%) & 5+ & 5+ (\\%) '
                 '& Mean & Median \\\\')
    lines.append('    \\midrule')
    for _, row in pub_stats_df.sort_values('n_authors', ascending=False).iterrows():
        pub = row['publisher'].replace('&', '\\&')
        lines.append(
            f"    {pub} & {row['n_authors']:,} "
            f"& {row['n_2plus']:,} & {row['pct_2plus']:.1f} "
            f"& {row['n_5plus']:,} & {row['pct_5plus']:.1f} "
            f"& {row['mean']:.2f} & {row['median']:.1f} \\\\"
        )
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Table 4: Prolific offenders (>= 50 retractions) with Dimensions ID
    if prolific_profiles:
        lines.append(f'% --- Prolific Offenders (>= 50 retractions): '
                     f'{len(prolific_profiles)} authors ---')
        lines.append('')
        lines.append('\\begin{table}[htbp]')
        lines.append('  \\centering')
        lines.append('  \\small')
        lines.append('  \\caption{Authors with 50 or more retraction entries. '
                     'Dimensions researcher\\_id is provided where available '
                     'for disambiguation.}')
        lines.append('  \\label{tab:prolific-offenders}')
        lines.append('  \\begin{tabular}{r l r l l}')
        lines.append('    \\toprule')
        lines.append('    \\# & Author & Retractions & Dimensions ID & Publishers \\\\')
        lines.append('    \\midrule')
        for i, p in enumerate(prolific_profiles, 1):
            name_esc = p['name'].replace('&', '\\&')
            pubs_esc = p['publishers'].replace('&', '\\&')
            dim_str = p['dim_id'] if p['dim_id'] else '---'
            lines.append(
                f"    {i} & {name_esc} & {p['n_retractions']:,} "
                f"& \\texttt{{{dim_str}}} & {pubs_esc} \\\\"
            )
        lines.append('    \\bottomrule')
        lines.append('  \\end{tabular}')
        lines.append('\\end{table}')
        lines.append('')

        # Per-author reason profiles as comments
        for p in prolific_profiles:
            lines.append(f"% --- Reason profile: {p['name']} "
                         f"({p['n_retractions']} retractions) ---")
            dim_str = p['dim_id'] if p['dim_id'] else 'name-based'
            lines.append(f"% Dimensions ID: {dim_str}")
            lines.append(f"% Publishers: {p['publishers']}")
            for reason, count in p['reasons']:
                pct = count / p['n_retractions'] * 100
                lines.append(f'%   {count:>4} ({pct:>5.1f}%)  {reason}')
            lines.append('')

        # Table 5: Reason breakdown for prolific offenders
        # Collect all unique reasons across prolific authors
        all_reasons = set()
        for p in prolific_profiles:
            for reason, _ in p['reasons']:
                all_reasons.add(reason)
        all_reasons = sorted(all_reasons)

        lines.append('\\begin{table*}[htbp]')
        lines.append('  \\centering')
        lines.append('  \\small')
        lines.append('  \\caption{Consolidated retraction reasons for authors with '
                     '50+ retraction entries. Each cell shows the number of entries '
                     'listing that reason (an entry may list multiple reasons).}')
        lines.append('  \\label{tab:prolific-offender-reasons}')
        n_cols = len(prolific_profiles)
        col_spec = 'l' + ' r' * n_cols
        lines.append(f'  \\begin{{tabular}}{{{col_spec}}}')
        lines.append('    \\toprule')
        # Header: reason + author names (abbreviated)
        header_parts = ['Reason']
        for p in prolific_profiles:
            # Use last name only for compactness
            parts = p['name'].split()
            short = parts[-1] if parts else p['name']
            header_parts.append(short.replace('&', '\\&'))
        lines.append('    ' + ' & '.join(header_parts) + ' \\\\')
        lines.append('    \\midrule')
        # Build a quick lookup
        for reason in all_reasons:
            row_parts = [reason.replace('&', '\\&')]
            for p in prolific_profiles:
                rdict = dict(p['reasons'])
                val = rdict.get(reason, 0)
                row_parts.append(str(val) if val > 0 else '---')
            lines.append('    ' + ' & '.join(row_parts) + ' \\\\')
        lines.append('    \\bottomrule')
        lines.append('  \\end{tabular}')
        lines.append('\\end{table*}')
        lines.append('')

    # Commented-out figure blocks
    lines.append('% --- Figure: Repeat-author distribution ---')
    lines.append('% \\begin{figure}[htbp]')
    lines.append('%   \\centering')
    lines.append('%   \\includegraphics[width=\\linewidth]'
                 '{figures/repeat-author-distribution.eps}')
    lines.append('%   \\caption{Distribution of retractions per author '
                 '(log-scaled y-axis). The majority of authors have a single '
                 'retraction, but a long tail of repeat offenders is visible.}')
    lines.append('%   \\label{fig:repeat-author-distribution}')
    lines.append('% \\end{figure}')
    lines.append('')
    lines.append('% --- Figure: Per-publisher repeat authors ---')
    lines.append('% \\begin{figure}[htbp]')
    lines.append('%   \\centering')
    lines.append('%   \\includegraphics[width=\\linewidth]'
                 '{figures/repeat-authors-per-publisher.eps}')
    lines.append('%   \\caption{Percentage of authors falling into each '
                 'retraction-count bin (1, 2--4, 5--9, 10+) by publisher.}')
    lines.append('%   \\label{fig:repeat-authors-per-publisher}')
    lines.append('% \\end{figure}')
    lines.append('')

    with open(RESULTS_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
