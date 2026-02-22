#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Sensitivity analysis comparing three author disambiguation strategies.

Strategies:
  S1 (Name-only):        name-based keys only (no institution, no researcher_id)
  S2 (Name+Institution): researcher_id where available, else name+institution
  S3 (Dimensions-only):  only authors with a Dimensions researcher_id

Outputs:
  results/sensitivity-authors.tex — two LaTeX tables (global + per-publisher)
"""

import json
import pandas as pd
from collections import defaultdict

FILTERED_CSV = 'filtered.csv'
CACHE_FILE = 'author_ids_cache.json'
RESULTS_TEX = 'results/sensitivity-authors.tex'

PUBLISHERS = {
    'ACM': ['Association for Computing Machinery (ACM)'],
    'IEEE': ['IEEE: Institute of Electrical and Electronics Engineers'],
    'Elsevier': ['Elsevier', 'Elsevier - Cell Press'],
    'Springer Nature': ['Springer', 'Springer - Nature Publishing Group',
                        'Springer - Biomed Central (BMC)'],
    'Wiley': ['Wiley'],
    'Taylor & Francis': ['Taylor and Francis', 'Taylor and Francis - Dove Press'],
    'SAGE': ['SAGE Publications'],
    'Hindawi': ['Hindawi'],
    'IOS Press': ['IOS Press (bought by Sage November 2023)'],
    'PLoS': ['PLoS'],
}

PUBLISHER_LABELS = {}
for k, v_list in PUBLISHERS.items():
    for v in v_list:
        PUBLISHER_LABELS[v] = k


def _normalize_institution(raw) -> str:
    """Normalize an institution string for disambiguation keys."""
    if not raw or pd.isna(raw):
        return ''
    parts = [p.strip().lower() for p in str(raw).split(';')]
    parts = [p for p in parts if p and p != 'unavailable']
    return '|'.join(sorted(parts))


def resolve_s1(row, cache: dict) -> list[tuple[str, str]]:
    """S1: Name-only (original baseline). No IDs, no institutions."""
    doi = row.get('OriginalPaperDOI')
    authors = []

    if pd.notna(doi) and str(doi).strip().lower() in cache:
        for a in cache[str(doi).strip().lower()]:
            name = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
            if not name or name.lower() == 'unknown':
                continue
            authors.append((f'name::{name}', name))
    else:
        raw = row.get('Author', '')
        if pd.isna(raw) or not str(raw).strip():
            return []
        for name in str(raw).split(';'):
            name = name.strip()
            if not name or name.lower() == 'unknown':
                continue
            authors.append((f'name::{name}', name))
    return authors


def resolve_s2(row, cache: dict) -> list[tuple[str, str]]:
    """S2: Name+Institution (the improved approach from 7-repeat-authors.py)."""
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


def resolve_s3(row, cache: dict) -> list[tuple[str, str]]:
    """S3: Dimensions-only (conservative). Only researcher_id matches."""
    doi = row.get('OriginalPaperDOI')
    authors = []

    if pd.notna(doi) and str(doi).strip().lower() in cache:
        for a in cache[str(doi).strip().lower()]:
            rid = a.get('researcher_id')
            if not rid:
                continue
            name = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
            if not name or name.lower() == 'unknown':
                continue
            authors.append((f'rid::{rid}', name))
    return authors


def compute_stats(df, cache, resolve_fn):
    """Run a disambiguation strategy and compute summary statistics."""
    author_entries = defaultdict(list)

    for _, row in df.iterrows():
        authors = resolve_fn(row, cache)
        pub_label = row['PublisherLabel']
        for key, name in authors:
            author_entries[key].append({
                'publisher': pub_label,
                'name': name,
            })

    n_unique = len(author_entries)
    counts = pd.Series({k: len(v) for k, v in author_entries.items()})

    if n_unique == 0:
        return {
            'n_unique': 0, 'n_2plus': 0, 'n_5plus': 0,
            'n_10plus': 0, 'n_50plus': 0,
            'per_publisher': {},
        }

    global_stats = {
        'n_unique': n_unique,
        'n_2plus': int((counts >= 2).sum()),
        'n_5plus': int((counts >= 5).sum()),
        'n_10plus': int((counts >= 10).sum()),
        'n_50plus': int((counts >= 50).sum()),
    }

    # Per-publisher breakdown
    per_pub = {}
    for label in PUBLISHERS.keys():
        pub_authors = {}
        for key, entries in author_entries.items():
            pub_count = sum(1 for e in entries if e['publisher'] == label)
            if pub_count > 0:
                pub_authors[key] = pub_count

        n = len(pub_authors)
        if n == 0:
            per_pub[label] = {'n_authors': 0, 'n_2plus': 0, 'n_5plus': 0}
            continue

        pub_counts = pd.Series(pub_authors)
        per_pub[label] = {
            'n_authors': n,
            'n_2plus': int((pub_counts >= 2).sum()),
            'n_5plus': int((pub_counts >= 5).sum()),
        }

    global_stats['per_publisher'] = per_pub
    return global_stats


def write_tex(results: dict):
    """Write LaTeX tables comparing the three strategies."""
    lines = []
    lines.append('% Sensitivity Analysis: Author Disambiguation Strategies')
    lines.append('% Generated by: 7-sensitivity-authors.py')
    lines.append('')

    # Table 1: Global comparison
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\caption{Sensitivity analysis of author disambiguation '
                 'strategies. S1 uses author name only, S2 uses Dimensions '
                 '\\texttt{researcher\\_id} with name+institution fallback '
                 '(primary analysis), and S3 uses Dimensions '
                 '\\texttt{researcher\\_id} only.}')
    lines.append('  \\label{tab:sensitivity-authors}')
    lines.append('  \\begin{tabular}{l r r r}')
    lines.append('    \\toprule')
    lines.append('    Metric & S1 (Name) & S2 (Name+Inst.) & S3 (Dim. only) \\\\')
    lines.append('    \\midrule')

    s1, s2, s3 = results['S1'], results['S2'], results['S3']
    rows = [
        ('Unique authors', 'n_unique'),
        ('With 2+ retractions', 'n_2plus'),
        ('With 5+ retractions', 'n_5plus'),
        ('With 10+ retractions', 'n_10plus'),
        ('With 50+ retractions', 'n_50plus'),
    ]
    for label, key in rows:
        lines.append(f'    {label} & {s1[key]:,} & {s2[key]:,} & {s3[key]:,} \\\\')

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Table 2: Per-publisher comparison (unique authors + 2+ rate)
    lines.append('\\begin{table*}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\caption{Per-publisher unique author counts and repeat-author '
                 'rates (2+) under each disambiguation strategy.}')
    lines.append('  \\label{tab:sensitivity-authors-per-publisher}')
    lines.append('  \\begin{tabular}{l r r r r r r}')
    lines.append('    \\toprule')
    lines.append('    & \\multicolumn{2}{c}{S1 (Name)} '
                 '& \\multicolumn{2}{c}{S2 (Name+Inst.)} '
                 '& \\multicolumn{2}{c}{S3 (Dim. only)} \\\\')
    lines.append('    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}')
    lines.append('    Publisher & Authors & 2+ (\\%) '
                 '& Authors & 2+ (\\%) '
                 '& Authors & 2+ (\\%) \\\\')
    lines.append('    \\midrule')

    for label in PUBLISHERS.keys():
        pub_esc = label.replace('&', '\\&')
        parts = [pub_esc]
        for skey in ['S1', 'S2', 'S3']:
            pp = results[skey]['per_publisher'].get(label, {})
            n = pp.get('n_authors', 0)
            n2 = pp.get('n_2plus', 0)
            pct = n2 / n * 100 if n > 0 else 0
            parts.append(f'{n:,}')
            parts.append(f'{pct:.1f}')
        lines.append('    ' + ' & '.join(parts) + ' \\\\')

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table*}')
    lines.append('')

    with open(RESULTS_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Results written to {RESULTS_TEX}')


def main():
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    df['PublisherLabel'] = df['Publisher'].map(PUBLISHER_LABELS)
    df = df.dropna(subset=['PublisherLabel'])

    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    print(f'Loaded {len(cache):,} DOIs from author cache')
    print(f'Dataset: {len(df):,} entries\n')

    strategies = {
        'S1': ('Name-only', resolve_s1),
        'S2': ('Name+Institution', resolve_s2),
        'S3': ('Dimensions-only', resolve_s3),
    }

    results = {}
    for key, (label, fn) in strategies.items():
        print(f'Running {key} ({label})...')
        stats = compute_stats(df, cache, fn)
        results[key] = stats
        print(f'  Unique authors: {stats["n_unique"]:,}')
        print(f'  2+ retractions: {stats["n_2plus"]:,}')
        print(f'  5+ retractions: {stats["n_5plus"]:,}')
        print(f'  50+ retractions: {stats["n_50plus"]:,}')
        print()

    write_tex(results)


if __name__ == '__main__':
    main()
