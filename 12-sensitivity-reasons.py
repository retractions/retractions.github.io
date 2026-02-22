#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Sensitivity analysis for retraction-reason consolidation.

Tests three alternative mappings (S1--S3) against the baseline (S0) to
assess whether reassigning contested reason codes changes the per-publisher
reason distributions.  Contested codes were flagged by a reviewer:

  S1 – Conflict of Interest  -> Misconduct  (was Authorship)
  S2 – Copyright Claims      -> Other       (was Plagiarism)
       Taken from Diss./Thesis-> Other       (was Plagiarism)
  S3 – Stress test combining all S1/S2 overrides plus:
       Criminal Proceedings   -> Third Party (was Misconduct)
       Referencing/Attribution-> Authorship  (was Plagiarism)
       Breach of Policy       -> Other       (was Misconduct)

Metrics per publisher, per scheme vs. baseline:
  - Max |delta|  : largest absolute percentage-point shift for any reason
  - Spearman rho : rank correlation of reason-count vectors
  - Top-3 stable : whether the three most frequent reasons match S0

Outputs:
  results/sensitivity-reasons.tex
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FILTERED_CSV = 'filtered.csv'
SUBSTITUTES_JSON = 'substitutes.json'
RESULTS_TEX = 'results/sensitivity-reasons.tex'

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

META_REASONS = {
    'Investigation by Journal/Publisher',
    'Information',
    'Date of Article and/or Notice Unknown',
    'Notice',
    'Upgrade/Update of Prior Notice(s)',
    'Notice - Unable to Access via current resources',
}

# ---------- Alternative mapping overrides -----------------------------------
# Each dict maps original reason substrings to their alternative target.

S1_OVERRIDES = {
    'Conflict of Interest': 'Misconduct',
}

S2_OVERRIDES = {
    'Copyright Claims': 'Other',
    'Taken from Dissertation/Thesis': 'Other',
}

S3_OVERRIDES = {
    'Conflict of Interest': 'Misconduct',
    'Copyright Claims': 'Other',
    'Taken from Dissertation/Thesis': 'Other',
    'Criminal Proceedings': 'Third Party',
    'Referencing/Attribution': 'Authorship',
    'Breach of Policy by Author': 'Other',
}


# ---------- Helpers (from 11-fig-reasons-by-publisher.py) -------------------

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


def reason_counts_for(df, substitutes):
    """Return per-publisher reason count dicts under a given substitution map."""
    df = df.copy()
    df['_reason'] = df['Reason'].apply(lambda x: rewrite_category(x, substitutes))

    pub_counts = {}
    pub_totals = {}
    for label, rw_names in PUBLISHERS.items():
        pub_df = df[df['Publisher'].isin(rw_names)]
        n = len(pub_df)
        pub_totals[label] = n
        reasons = expand_semicolon_field(pub_df['_reason'])
        reasons = reasons[~reasons.isin(META_REASONS)]
        reasons = reasons.groupby(level=0).apply(
            lambda g: g.drop_duplicates()).droplevel(0)
        pub_counts[label] = reasons.value_counts()

    return pub_counts, pub_totals


def compare_schemes(baseline_counts, alt_counts, pub_totals):
    """Compare alternative vs. baseline reason distributions per publisher.

    Returns a dict keyed by publisher label with:
      max_delta  – largest absolute percentage-point change
      rho        – Spearman rank correlation
      top3_stable – bool, whether top-3 reasons (and order) match
    """
    results = {}
    for label in PUBLISHERS:
        bc = baseline_counts[label]
        ac = alt_counts[label]
        n = pub_totals[label]
        if n == 0:
            results[label] = {'max_delta': 0.0, 'rho': 1.0, 'top3_stable': True}
            continue

        # Align to the union of all reason codes
        all_reasons = sorted(set(bc.index) | set(ac.index))
        bv = np.array([bc.get(r, 0) for r in all_reasons], dtype=float)
        av = np.array([ac.get(r, 0) for r in all_reasons], dtype=float)

        # Percentage-point differences
        bp = bv / n * 100
        ap = av / n * 100
        max_delta = float(np.max(np.abs(bp - ap)))

        # Spearman rho on categories that are non-zero in at least one scheme
        # (avoids inflated tied-zero ranks for sparse publishers like ACM)
        mask = (bv > 0) | (av > 0)
        bv_nz = bv[mask]
        av_nz = av[mask]
        if len(bv_nz) < 2 or np.all(bv_nz == av_nz):
            rho = 1.0
        else:
            rho, _ = spearmanr(bv_nz, av_nz)

        # Top-3 stability
        top3_base = list(bc.head(3).index)
        top3_alt = list(ac.head(3).index)
        top3_stable = top3_base == top3_alt

        results[label] = {
            'max_delta': max_delta,
            'rho': rho,
            'top3_stable': top3_stable,
        }
    return results


def write_tex(all_results, pub_totals):
    """Write LaTeX table and prose summary to results/sensitivity-reasons.tex."""
    lines = []
    lines.append('% Sensitivity Analysis: Reason Consolidation')
    lines.append('% Generated by: 12-sensitivity-reasons.py')
    lines.append('')

    # --- Table ---
    lines.append('\\begin{table*}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\caption{Sensitivity analysis of reason consolidation. '
                 'Three alternative mappings (S1--S3) are compared against '
                 'the baseline (S0). '
                 'S1 moves Conflict of Interest from Authorship to Misconduct. '
                 'S2 moves Copyright Claims and Taken from Dissertation/Thesis '
                 'from Plagiarism to Other. '
                 'S3 combines all contested reassignments '
                 '(S1 + S2 plus Criminal Proceedings to Third Party, '
                 'Referencing/Attribution to Authorship, '
                 'and Breach of Policy by Author to Other). '
                 'Max $|\\Delta|$ is the largest absolute percentage-point '
                 'shift for any reason category. '
                 '$\\rho$ is the Spearman rank correlation between baseline '
                 'and alternative count vectors (computed over non-zero '
                 'categories only). '
                 'Top-3 indicates whether the three most frequent reasons '
                 'match the baseline in identity and rank order '
                 '(\\checkmark = stable, $\\times$ = changed).}')
    lines.append('  \\label{tab:sensitivity-reasons}')
    lines.append('  \\small')
    lines.append('  \\begin{tabular}{l r rrr rrr rrr}')
    lines.append('    \\toprule')
    lines.append('    & '
                 '& \\multicolumn{3}{c}{S1} '
                 '& \\multicolumn{3}{c}{S2} '
                 '& \\multicolumn{3}{c}{S3} \\\\')
    lines.append('    \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}')
    lines.append('    Publisher & N '
                 '& $|\\Delta|$ & $\\rho$ & Top-3 '
                 '& $|\\Delta|$ & $\\rho$ & Top-3 '
                 '& $|\\Delta|$ & $\\rho$ & Top-3 \\\\')
    lines.append('    \\midrule')

    for label in PUBLISHERS:
        n = pub_totals[label]
        pub_esc = label.replace('&', '\\&')
        cells = [pub_esc, f'{n:,}']

        for scheme in ['S1', 'S2', 'S3']:
            r = all_results[scheme][label]
            cells.append(f'{r["max_delta"]:.2f}')
            cells.append(f'{r["rho"]:.3f}')
            cells.append('\\checkmark' if r['top3_stable'] else '$\\times$')

        lines.append('    ' + ' & '.join(cells) + ' \\\\')

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table*}')
    lines.append('')

    with open(RESULTS_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    df['PublisherLabel'] = df['Publisher'].map(PUBLISHER_LABELS)
    df = df.dropna(subset=['PublisherLabel'])
    print(f'Dataset: {len(df):,} entries')

    with open(SUBSTITUTES_JSON, 'r') as f:
        baseline_subs = json.load(f)

    # Baseline
    baseline_counts, pub_totals = reason_counts_for(df, baseline_subs)

    # Alternative schemes
    schemes = {
        'S1': S1_OVERRIDES,
        'S2': S2_OVERRIDES,
        'S3': S3_OVERRIDES,
    }

    all_results = {}
    for scheme_name, overrides in schemes.items():
        alt_subs = dict(baseline_subs)
        alt_subs.update(overrides)
        alt_counts, _ = reason_counts_for(df, alt_subs)
        comp = compare_schemes(baseline_counts, alt_counts, pub_totals)
        all_results[scheme_name] = comp

        print(f'\n{scheme_name}:')
        for label in PUBLISHERS:
            r = comp[label]
            stable = 'yes' if r['top3_stable'] else 'NO'
            print(f'  {label:20s}  max|d|={r["max_delta"]:5.2f} pp  '
                  f'rho={r["rho"]:.4f}  top3={stable}')

    write_tex(all_results, pub_totals)
    print(f'\nResults written to {RESULTS_TEX}')


if __name__ == '__main__':
    main()
