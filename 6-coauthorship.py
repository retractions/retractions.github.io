#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Analyse country co-authorship patterns in retracted publications.
Tests the hypothesis that US retraction counts are inflated by
co-authorship with Chinese-led research teams.

Outputs:
  figures/country-cooccurrence-heatmap.eps
  figures/coauthorship-us-breakdown.eps
  results/coauthorship-analysis.tex
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

FILTERED_CSV = 'filtered.csv'
HEATMAP_FIGURE = 'figures/country-cooccurrence-heatmap.eps'
US_BREAKDOWN_FIGURE = 'figures/coauthorship-us-breakdown.eps'
RESULTS_TEX = 'results/coauthorship-analysis.tex'

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

from plot_config import HEATMAP_CMAP, US_BREAKDOWN_COLORS

# Reverse mapping: full name -> short label
PUBLISHER_LABELS = {}
for k, v_list in PUBLISHERS.items():
    for v in v_list:
        PUBLISHER_LABELS[v] = k


def parse_countries(country_str):
    """Parse semicolon-delimited country string into an ordered list."""
    if pd.isna(country_str) or country_str.strip() == '':
        return []
    return [c.strip(' ;') for c in country_str.split(';') if c.strip(' ;')]


def main():
    # ---- Load data ------------------------------------------------------------
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    df['PublisherLabel'] = df['Publisher'].map(PUBLISHER_LABELS)
    df = df.dropna(subset=['PublisherLabel'])

    # Parse countries into lists (preserving order)
    df['CountryList'] = df['Country'].apply(parse_countries)
    df['NumCountries'] = df['CountryList'].apply(len)
    df['FirstCountry'] = df['CountryList'].apply(lambda x: x[0] if len(x) > 0 else None)

    # ---- 1. Solo vs. multi-country stats --------------------------------------
    total = len(df)
    has_country = df[df['NumCountries'] > 0]
    solo = df[df['NumCountries'] == 1]
    multi = df[df['NumCountries'] >= 2]

    print('='*80)
    print('SOLO vs. MULTI-COUNTRY ENTRIES')
    print('='*80)
    print(f'Total entries:        {total:,}')
    print(f'With country info:    {len(has_country):,}')
    print(f'Solo country:         {len(solo):,} ({len(solo)/total*100:.1f}%)')
    print(f'Multi-country (2+):   {len(multi):,} ({len(multi)/total*100:.1f}%)')
    print()

    # ---- 2. Top-10 countries (expanded) ---------------------------------------
    all_countries = df['CountryList'].explode().dropna()
    all_countries = all_countries[all_countries != '']
    top10_expanded = all_countries.value_counts().head(10)
    top10_names = list(top10_expanded.index)

    print('Top 10 countries (expanded):')
    for c, n in top10_expanded.items():
        print(f'  {c}: {n:,}')
    print()

    # ---- 3. Per-country solo vs. multi breakdown ------------------------------
    print(f'{"Country":<25s} {"Total":>7s} {"Solo":>7s} {"Multi":>7s} {"Solo%":>7s}')
    print('-'*60)
    country_breakdown = []
    for country in top10_names:
        mask = df['CountryList'].apply(lambda x: country in x)
        c_total = mask.sum()
        c_solo = (mask & (df['NumCountries'] == 1)).sum()
        c_multi = c_total - c_solo
        solo_pct = c_solo / c_total * 100 if c_total > 0 else 0
        country_breakdown.append({
            'Country': country, 'Total': c_total,
            'Solo': c_solo, 'Multi': c_multi, 'SoloPct': solo_pct,
        })
        print(f'{country:<25s} {c_total:>7,} {c_solo:>7,} {c_multi:>7,} {solo_pct:>6.1f}%')
    print()

    # ---- 4. US-specific analysis ----------------------------------------------
    us_mask = df['CountryList'].apply(lambda x: 'United States' in x)
    us_df = df[us_mask]
    us_total = len(us_df)
    us_solo = (us_mask & (df['NumCountries'] == 1)).sum()
    us_china = us_mask & df['CountryList'].apply(lambda x: 'China' in x)
    us_china_count = us_china.sum()
    us_other_collab = us_total - us_solo - us_china_count

    print('US ENTRIES BREAKDOWN')
    print('='*60)
    print(f'Total with US:            {us_total:,}')
    print(f'Solo US:                  {us_solo:,} ({us_solo/us_total*100:.1f}%)')
    print(f'US + China:               {us_china_count:,} ({us_china_count/us_total*100:.1f}%)')
    print(f'US + Other (no China):    {us_other_collab:,} ({us_other_collab/us_total*100:.1f}%)')
    print()

    # China listed first when co-occurring with US?
    china_us_entries = df[us_china]
    china_first = china_us_entries['FirstCountry'].eq('China').sum()
    print(f'China+US entries where China is listed first: '
          f'{china_first}/{len(china_us_entries)} '
          f'({china_first/len(china_us_entries)*100:.1f}%)')
    print()

    # ---- 5. Co-occurrence matrix (top 10 countries + Other) --------------------
    top10_set = set(top10_names)
    labels = top10_names + ['Other']
    cooc = pd.DataFrame(0, index=labels, columns=labels, dtype=int)

    for countries in df['CountryList']:
        if len(countries) < 2:
            continue
        # Map each country to its label (keep top 10, rest -> Other)
        mapped = []
        for c in countries:
            mapped.append(c if c in top10_set else 'Other')
        # Deduplicate while preserving order (multiple "Other" countries
        # in one entry should count as a single Other involvement)
        seen = set()
        unique_mapped = []
        for m in mapped:
            if m not in seen:
                seen.add(m)
                unique_mapped.append(m)
        for a, b in itertools.combinations(unique_mapped, 2):
            cooc.loc[a, b] += 1
            cooc.loc[b, a] += 1

    print('Co-occurrence matrix (top 10 + Other):')
    print(cooc)
    print()

    # Heatmap — lower triangle only
    n = len(labels)
    mask_upper = np.triu(np.ones((n, n), dtype=bool))  # mask diagonal + upper
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cooc, annot=True, fmt='d', cmap=HEATMAP_CMAP,
        mask=mask_upper, linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Co-occurrence count'},
    )
    # ax.set_title('Country Co-occurrence in Retracted Publications', fontsize=AXIS_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(HEATMAP_FIGURE, format='eps', bbox_inches='tight')
    print(f'Heatmap saved to {HEATMAP_FIGURE}')
    plt.close()

    # ---- 6. First-listed-only ranking vs. expanded ranking --------------------
    first_counts = df['FirstCountry'].value_counts().head(10)
    top10_first = list(first_counts.index)

    print('\nEXPANDED vs. FIRST-LISTED COUNTRY RANKING')
    print('='*80)
    print(f'{"Rank":<6s} {"Expanded":<25s} {"N(exp)":>8s}   '
          f'{"First-listed":<25s} {"N(1st)":>8s}')
    print('-'*80)
    for i in range(10):
        exp_c = top10_expanded.index[i]
        exp_n = top10_expanded.iloc[i]
        first_c = first_counts.index[i] if i < len(first_counts) else ''
        first_n = first_counts.iloc[i] if i < len(first_counts) else 0
        print(f'{i+1:<6d} {exp_c:<25s} {exp_n:>8,}   '
              f'{first_c:<25s} {first_n:>8,}')
    print()

    # ---- 7. Per-publisher US breakdown ----------------------------------------
    pub_us_stats = []
    print('PER-PUBLISHER US BREAKDOWN')
    print('='*80)
    print(f'{"Publisher":<22s} {"US(any)":>8s} {"US solo":>8s} '
          f'{"US+CN":>8s} {"US+Other":>8s}')
    print('-'*80)

    for label in PUBLISHERS.keys():
        pub_df = df[df['PublisherLabel'] == label]
        p_us = pub_df['CountryList'].apply(lambda x: 'United States' in x)
        p_us_total = p_us.sum()
        p_us_solo = (p_us & (pub_df['NumCountries'] == 1)).sum()
        p_us_china = (p_us & pub_df['CountryList'].apply(
            lambda x: 'China' in x)).sum()
        p_us_other = p_us_total - p_us_solo - p_us_china
        pub_us_stats.append({
            'Publisher': label, 'US_any': p_us_total,
            'US_solo': p_us_solo, 'US_China': p_us_china,
            'US_Other': p_us_other,
        })
        print(f'{label:<22s} {p_us_total:>8,} {p_us_solo:>8,} '
              f'{p_us_china:>8,} {p_us_other:>8,}')
    print()

    pub_us_df = pd.DataFrame(pub_us_stats)

    # ---- 8. US breakdown bar chart per publisher ------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter to publishers with at least 1 US entry
    plot_df = pub_us_df[pub_us_df['US_any'] > 0].copy()
    plot_df = plot_df.sort_values('US_any', ascending=True)

    y_pos = range(len(plot_df))
    bar_height = 0.6

    ax.barh(y_pos, plot_df['US_solo'], height=bar_height,
            label='US only', color=US_BREAKDOWN_COLORS[0],
            hatch='', edgecolor='black', linewidth=0.5)
    ax.barh(y_pos, plot_df['US_China'], height=bar_height,
            left=plot_df['US_solo'],
            label='US + China', color=US_BREAKDOWN_COLORS[1],
            hatch='///', edgecolor='black', linewidth=0.5)
    ax.barh(y_pos, plot_df['US_Other'], height=bar_height,
            left=plot_df['US_solo'] + plot_df['US_China'],
            label='US + Other', color=US_BREAKDOWN_COLORS[2],
            hatch='xxx', edgecolor='black', linewidth=0.5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df['Publisher'])
    ax.set_xlabel('Number of retraction entries involving US', fontsize=AXIS_FONTSIZE)
    ax.legend(fontsize=LABEL_FONTSIZE, loc='lower right')
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(US_BREAKDOWN_FIGURE, format='eps', bbox_inches='tight')
    print(f'US breakdown chart saved to {US_BREAKDOWN_FIGURE}')
    plt.close()

    # ---- 9. Write results .tex file -------------------------------------------
    write_results_tex(
        total, solo, multi, us_total, us_solo, us_china_count,
        us_other_collab, china_first, china_us_entries,
        top10_expanded, first_counts, country_breakdown,
        pub_us_df, cooc,
    )
    print(f'Results written to {RESULTS_TEX}')


def write_results_tex(
    total, solo, multi, us_total, us_solo, us_china_count,
    us_other_collab, china_first, china_us_entries,
    top10_expanded, first_counts, country_breakdown,
    pub_us_df, cooc,
):
    """Write LaTeX results file with co-authorship analysis findings."""

    lines = []
    lines.append('% Co-authorship Analysis')
    lines.append('% Generated by: 6-coauthorship.py')
    lines.append('% Figures: figures/country-cooccurrence-heatmap.eps, '
                 'figures/coauthorship-us-breakdown.eps')
    lines.append('%')
    lines.append('% This file analyses country co-authorship patterns in retracted')
    lines.append('% publications, testing the hypothesis that US retraction counts')
    lines.append('% are inflated by co-authorship with Chinese-led research teams.')
    lines.append('')

    # Key findings as comments
    lines.append('% --- Key Findings ---')
    lines.append(f'% Total entries: {total:,}')
    lines.append(f'% Solo-country entries: {len(solo):,} ({len(solo)/total*100:.1f}%)')
    lines.append(f'% Multi-country entries: {len(multi):,} ({len(multi)/total*100:.1f}%)')
    lines.append(f'%')
    lines.append(f'% US entries total: {us_total:,}')
    lines.append(f'% US solo: {us_solo:,} ({us_solo/us_total*100:.1f}%)')
    lines.append(f'% US + China: {us_china_count:,} ({us_china_count/us_total*100:.1f}%)')
    lines.append(f'% US + Other (no China): {us_other_collab:,} '
                 f'({us_other_collab/us_total*100:.1f}%)')
    cf_pct = china_first / len(china_us_entries) * 100
    lines.append(f'%')
    lines.append(f'% China+US entries where China is listed first: '
                 f'{china_first}/{len(china_us_entries)} ({cf_pct:.1f}%)')
    lines.append('')

    # Table 1: Expanded vs. first-listed country ranking
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Top 10 country rankings: expanded (all countries per entry) '
                 'vs.\\ first-listed country only.}')
    lines.append('  \\label{tab:country-ranking-comparison}')
    lines.append('  \\begin{tabular}{r l r l r}')
    lines.append('    \\toprule')
    lines.append('    Rank & Country (expanded) & N & Country (first-listed) & N \\\\')
    lines.append('    \\midrule')

    for i in range(10):
        exp_c = top10_expanded.index[i]
        exp_n = top10_expanded.iloc[i]
        first_c = first_counts.index[i] if i < len(first_counts) else ''
        first_n = first_counts.iloc[i] if i < len(first_counts) else 0
        lines.append(f'    {i+1} & {exp_c} & {exp_n:,} & {first_c} & {first_n:,} \\\\')

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Table 2: Per-country solo vs. multi-country breakdown
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{Solo vs.\\ multi-country authorship breakdown '
                 'for the top 10 retraction countries.}')
    lines.append('  \\label{tab:solo-multi-country}')
    lines.append('  \\begin{tabular}{l r r r r}')
    lines.append('    \\toprule')
    lines.append('    Country & Total & Solo & Multi & Solo (\\%) \\\\')
    lines.append('    \\midrule')

    for row in country_breakdown:
        lines.append(f"    {row['Country']} & {row['Total']:,} & {row['Solo']:,} "
                     f"& {row['Multi']:,} & {row['SoloPct']:.1f} \\\\")

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Table 3: Per-publisher US breakdown
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\caption{US involvement in retraction entries by publisher.}')
    lines.append('  \\label{tab:us-publisher-breakdown}')
    lines.append('  \\begin{tabular}{l r r r r}')
    lines.append('    \\toprule')
    lines.append('    Publisher & US (any) & US solo & US + China & US + Other \\\\')
    lines.append('    \\midrule')

    for _, row in pub_us_df.sort_values('US_any', ascending=False).iterrows():
        pub = row['Publisher'].replace('&', '\\&')
        lines.append(f"    {pub} & {row['US_any']:,} & {row['US_solo']:,} "
                     f"& {row['US_China']:,} & {row['US_Other']:,} \\\\")

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')
    lines.append('')

    # Commented-out figure blocks
    lines.append('% --- Figure: Co-occurrence heatmap ---')
    lines.append('% \\begin{figure}[htbp]')
    lines.append('%   \\centering')
    lines.append('%   \\includegraphics[width=\\linewidth]'
                 '{figures/country-cooccurrence-heatmap.eps}')
    lines.append('%   \\caption{Co-occurrence of the top 10 retraction countries. '
                 'Each cell shows the number of retraction entries listing both '
                 'countries. Diagonal is masked.}')
    lines.append('%   \\label{fig:country-cooccurrence-heatmap}')
    lines.append('% \\end{figure}')
    lines.append('')
    lines.append('% --- Figure: US breakdown ---')
    lines.append('% \\begin{figure}[htbp]')
    lines.append('%   \\centering')
    lines.append('%   \\includegraphics[width=\\linewidth]'
                 '{figures/coauthorship-us-breakdown.eps}')
    lines.append('%   \\caption{Breakdown of US involvement in retraction entries '
                 'by publisher: solo US authorship, US--China co-authorship, '
                 'and US with other countries (excluding China).}')
    lines.append('%   \\label{fig:coauthorship-us-breakdown}')
    lines.append('% \\end{figure}')
    lines.append('')

    with open(RESULTS_TEX, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
