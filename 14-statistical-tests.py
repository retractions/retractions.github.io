#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Statistical tests of retraction-reason concentration across publishers.

Computes per-publisher:
  - Herfindahl-Hirschman Index (HHI) on reason-mention shares
  - Shannon entropy on reason-mention shares
  - Permutation test: observed HHI vs. null of random sampling from pooled
    distribution (B=10,000)
  - Chi-square goodness-of-fit: ACM reason distribution vs. pooled rest
    (with Monte Carlo simulation, B=10,000)

Reuses data loading, reason consolidation, semicolon expansion,
deduplication, and meta-reason exclusion logic from
11-fig-reasons-by-publisher.py.

Outputs:
  results/statistical-tests.tex
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

FILTERED_CSV = 'filtered.csv'
SUBSTITUTES_JSON = 'substitutes.json'
RESULTS_TEX = 'results/statistical-tests.tex'

SEED = 42
B_PERM = 10_000
B_MC = 10_000

PUBLISHERS = {
    'ACM': ['Association for Computing Machinery (ACM)'],
    'IEEE': ['IEEE: Institute of Electrical and Electronics Engineers'],
    'Elsevier': ['Elsevier', 'Elsevier - Cell Press'],
    'Springer Nature': ['Springer', 'Springer - Nature Publishing Group',
                        'Springer - Biomed Central (BMC)'],
    'Wiley': ['Wiley'],
    'Taylor & Francis': ['Taylor and Francis',
                         'Taylor and Francis - Dove Press'],
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
    expanded = (series.str.split(';', expand=True)
                .stack().reset_index(level=1, drop=True))
    expanded = expanded.str.strip(' ;+')
    return expanded[expanded != '']


def compute_hhi(counts):
    """HHI from a Series of counts. Returns float in [1/K, 1]."""
    shares = counts / counts.sum()
    return float((shares ** 2).sum())


def compute_entropy(counts):
    """Shannon entropy (natural log) from a Series of counts."""
    shares = counts / counts.sum()
    shares = shares[shares > 0]
    return float(-(shares * np.log(shares)).sum())


def get_reason_counts_for_entries(df_subset):
    """
    Expand, filter meta-reasons, deduplicate within entries,
    and return a value_counts Series of reason mentions.
    """
    reasons = expand_semicolon_field(df_subset['Consolidated_Reason'])
    reasons = reasons[~reasons.isin(META_REASONS)]
    reasons = (reasons.groupby(level=0)
               .apply(lambda g: g.drop_duplicates())
               .droplevel(0))
    return reasons.value_counts()


def hhi_from_row_sums(row_sums):
    """Compute HHI from a 1-D array of reason counts (row sums of the
    entry-by-reason matrix for a subset of entries)."""
    total = row_sums.sum()
    if total == 0:
        return 0.0
    shares = row_sums / total
    return float((shares ** 2).sum())


def main():
    # ---- Load and prepare data ------------------------------------------------
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    df['PublisherLabel'] = df['Publisher'].map(PUBLISHER_LABELS)
    df = df.dropna(subset=['PublisherLabel'])

    with open(SUBSTITUTES_JSON, 'r') as f:
        substitutes = json.load(f)

    df['Consolidated_Reason'] = df['Reason'].apply(
        lambda x: rewrite_category(x, substitutes))

    # ---- Per-publisher reason counts ------------------------------------------
    pub_counts = {}    # label -> Series of reason counts
    pub_n = {}         # label -> number of entries
    for label, rw_names in PUBLISHERS.items():
        pub_df = df[df['Publisher'].isin(rw_names)]
        pub_n[label] = len(pub_df)
        pub_counts[label] = get_reason_counts_for_entries(pub_df)

    # ---- 1. HHI and Shannon entropy per publisher -----------------------------
    pub_hhi = {}
    pub_entropy = {}
    for label in PUBLISHERS:
        pub_hhi[label] = compute_hhi(pub_counts[label])
        pub_entropy[label] = compute_entropy(pub_counts[label])

    print('Per-publisher HHI and Shannon entropy')
    print('=' * 60)
    for label in PUBLISHERS:
        print(f'  {label:20s}  N={pub_n[label]:>6,}  '
              f'HHI={pub_hhi[label]:.4f}  H={pub_entropy[label]:.4f}')
    print()

    # ---- 2. Permutation test on HHI -------------------------------------------
    # Null: publisher's HHI could arise from drawing N entries from the pooled
    # distribution.  Sampling unit = entry (preserves within-entry correlation).
    #
    # Optimization: pre-compute a binary entry-by-reason matrix once, then
    # permutation just sums sampled rows (fast numpy indexing).
    rng = np.random.default_rng(SEED)

    # Pre-compute per-entry deduplicated reason sets
    all_reasons_expanded = expand_semicolon_field(df['Consolidated_Reason'])
    all_reasons_expanded = all_reasons_expanded[
        ~all_reasons_expanded.isin(META_REASONS)]
    all_reasons_expanded = (all_reasons_expanded.groupby(level=0)
                            .apply(lambda g: g.drop_duplicates())
                            .droplevel(0))

    # Build vocabulary of all unique reasons
    all_unique_reasons = sorted(all_reasons_expanded.unique())
    reason_to_idx = {r: i for i, r in enumerate(all_unique_reasons)}
    n_reasons = len(all_unique_reasons)
    n_entries = len(df)

    # Build binary matrix: entry_matrix[i, j] = 1 if entry i mentions reason j
    # Use positional indexing (df reset to 0..N-1)
    df_reset = df.reset_index(drop=True)
    entry_matrix = np.zeros((n_entries, n_reasons), dtype=np.float32)

    # Map original df index to positional index
    orig_to_pos = {orig: pos for pos, orig in enumerate(df.index)}
    for orig_idx, reason in all_reasons_expanded.items():
        pos = orig_to_pos[orig_idx]
        entry_matrix[pos, reason_to_idx[reason]] = 1.0

    # Map publisher labels to positional entry indices
    pub_pos_indices = {}
    for label, rw_names in PUBLISHERS.items():
        mask = df['Publisher'].isin(rw_names)
        pub_pos_indices[label] = np.where(mask.values)[0]

    all_pos_indices = np.arange(n_entries)

    pub_perm_p = {}
    for label in PUBLISHERS:
        n = pub_n[label]
        observed_hhi = pub_hhi[label]
        null_hhis = np.empty(B_PERM)

        for b in range(B_PERM):
            sampled = rng.choice(all_pos_indices, size=n, replace=False)
            col_sums = entry_matrix[sampled].sum(axis=0)
            null_hhis[b] = hhi_from_row_sums(col_sums)

        p_val = (np.sum(null_hhis >= observed_hhi) + 1) / (B_PERM + 1)
        pub_perm_p[label] = p_val
        print(f'  Permutation done: {label:20s}  p = {p_val:.4f}')

    print()
    print('Permutation test p-values (HHI >= observed)')
    print('=' * 60)
    for label in PUBLISHERS:
        p = pub_perm_p[label]
        print(f'  {label:20s}  p = {p:.4f}')
    print()

    # ---- 3. Chi-square goodness-of-fit: ACM vs. pooled rest -------------------
    # Collapse to top-10 global reasons + Other
    all_counts = get_reason_counts_for_entries(df)
    top10_reasons = list(all_counts.head(10).index)

    acm_counts = pub_counts['ACM']
    rest_counts = pd.Series(dtype=float)
    for label in PUBLISHERS:
        if label == 'ACM':
            continue
        rest_counts = rest_counts.add(pub_counts[label], fill_value=0)

    def collapse_to_top10(counts, top10):
        """Collapse counts into top-10 categories + Other."""
        collapsed = {}
        for reason in top10:
            collapsed[reason] = counts.get(reason, 0)
        other = sum(c for r, c in counts.items() if r not in top10)
        collapsed['Other'] = other
        return pd.Series(collapsed)

    acm_collapsed = collapse_to_top10(acm_counts, top10_reasons)
    rest_collapsed = collapse_to_top10(rest_counts, top10_reasons)

    # Expected: proportional to rest distribution, scaled to ACM total mentions
    acm_total = acm_collapsed.sum()
    rest_total = rest_collapsed.sum()
    expected = (rest_collapsed / rest_total) * acm_total

    # Check expected cell counts
    min_expected = expected.min()
    print(f'Chi-square expected cell counts (min = {min_expected:.2f}):')
    for reason in acm_collapsed.index:
        print(f'  {reason:40s}  obs={acm_collapsed[reason]:>6.0f}  '
              f'exp={expected[reason]:>8.2f}')
    print()

    # Asymptotic chi-square
    chi2_stat, chi2_p = stats.chisquare(acm_collapsed.values,
                                        f_exp=expected.values)
    chi2_dof = len(acm_collapsed) - 1

    # Monte Carlo chi-square (robust alternative)
    rng_mc = np.random.default_rng(SEED)
    rest_probs = (rest_collapsed / rest_total).values
    mc_chi2s = np.empty(B_MC)
    for b in range(B_MC):
        simulated = rng_mc.multinomial(int(acm_total), rest_probs)
        mc_chi2s[b] = np.sum((simulated - expected.values) ** 2
                             / expected.values)
    mc_p = (np.sum(mc_chi2s >= chi2_stat) + 1) / (B_MC + 1)

    print(f'Chi-square test: chi2 = {chi2_stat:.1f}, df = {chi2_dof}, '
          f'p = {chi2_p:.2e} (asymptotic), p = {mc_p:.4f} (Monte Carlo)')
    print()

    # ---- 4. Write results .tex ------------------------------------------------
    write_results_tex(pub_n, pub_hhi, pub_entropy, pub_perm_p,
                      chi2_stat, chi2_dof, chi2_p, mc_p,
                      acm_collapsed, expected)
    print(f'Results written to {RESULTS_TEX}')


def write_results_tex(pub_n, pub_hhi, pub_entropy, pub_perm_p,
                      chi2_stat, chi2_dof, chi2_p, mc_p,
                      acm_collapsed, expected):
    """Write LaTeX results file with concentration metrics and test results."""

    lines = []
    lines.append('% Statistical Tests of Reason Concentration')
    lines.append('% Generated by: 14-statistical-tests.py')
    lines.append('%')
    lines.append('% HHI, Shannon entropy, permutation test (B=10,000),')
    lines.append('% and chi-square goodness-of-fit (ACM vs. pooled rest).')
    lines.append('')

    # ---- LaTeX macros for inline use ------------------------------------------
    lines.append('% --- Inline macros ---')

    acm_hhi = pub_hhi['ACM']
    acm_entropy = pub_entropy['ACM']
    acm_perm_p = pub_perm_p['ACM']

    lines.append(f'\\newcommand{{\\acmHHI}}{{{acm_hhi:.2f}}}')
    lines.append(f'\\newcommand{{\\acmEntropy}}{{{acm_entropy:.2f}}}')
    if acm_perm_p < 0.001:
        lines.append('\\newcommand{\\acmPermP}{$< 0.001$}')
    else:
        lines.append(f'\\newcommand{{\\acmPermP}}{{{acm_perm_p:.3f}}}')

    lines.append(f'\\newcommand{{\\acmChiSq}}{{{chi2_stat:.1f}}}')
    lines.append(f'\\newcommand{{\\acmChiDof}}{{{chi2_dof}}}')
    if mc_p < 0.001:
        lines.append('\\newcommand{\\acmChiP}{$< 0.001$}')
    else:
        lines.append(f'\\newcommand{{\\acmChiP}}{{{mc_p:.3f}}}')

    if chi2_p < 0.001:
        lines.append('\\newcommand{\\acmChiPAsymptotic}{$< 0.001$}')
    else:
        lines.append(
            f'\\newcommand{{\\acmChiPAsymptotic}}{{{chi2_p:.3f}}}')
    lines.append('')

    # ---- Concentration table --------------------------------------------------
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\caption{Reason concentration per publisher. '
                 'HHI (Herfindahl-Hirschman Index) ranges from $1/K$ '
                 '(uniform) to 1.0 (single category). '
                 'Shannon entropy $H$ is higher for more diverse profiles. '
                 'Permutation $p$-values test whether the observed HHI '
                 'could arise from randomly drawing $N$ entries from the '
                 'pooled distribution ($B = 10{,}000$).}')
    lines.append('  \\label{tab:concentration}')
    lines.append('  \\begin{tabular}{l r r r r}')
    lines.append('    \\toprule')
    lines.append('    Publisher & $N$ & HHI & $H$ & Perm.\\ $p$ \\\\')
    lines.append('    \\midrule')

    # Sort by HHI descending for readability
    sorted_labels = sorted(PUBLISHERS.keys(),
                           key=lambda l: pub_hhi[l], reverse=True)
    for label in sorted_labels:
        n = pub_n[label]
        hhi = pub_hhi[label]
        h = pub_entropy[label]
        p = pub_perm_p[label]
        p_str = '< 0.001' if p < 0.001 else f'{p:.3f}'
        label_esc = label.replace('&', '\\&')
        lines.append(f'    {label_esc} & {n:,} & {hhi:.4f} & '
                     f'{h:.2f} & {p_str} \\\\')

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # ---- Chi-square details (as comments) -------------------------------------
    lines.append('% --- Chi-square goodness-of-fit: ACM vs. pooled rest ---')
    lines.append(f'% Categories: top 10 global reasons + Other '
                 f'({len(acm_collapsed)} categories)')
    lines.append(f'% chi2 = {chi2_stat:.1f}, df = {chi2_dof}, '
                 f'p = {chi2_p:.2e} (asymptotic)')
    lines.append(f'% Monte Carlo p = {mc_p:.4f} (B = {B_MC:,})')
    lines.append(f'% Min expected cell count = {expected.min():.2f}')
    lines.append('%')
    lines.append('% Observed vs. expected per category:')
    for reason in acm_collapsed.index:
        obs = acm_collapsed[reason]
        exp = expected[reason]
        lines.append(f'%   {reason:40s}  obs={obs:>6.0f}  exp={exp:>8.2f}')
    lines.append('')

    with open(RESULTS_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
