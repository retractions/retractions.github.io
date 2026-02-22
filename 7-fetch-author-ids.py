#!/usr/bin/python3
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Fetch author data (researcher_id, ORCID, names) from the Dimensions API
for all DOIs in the filtered Retraction Watch dataset.

Results are cached to author_ids_cache.json with resume support:
if the cache file exists, already-fetched DOIs are skipped.

Outputs:
  author_ids_cache.json — {doi: [{first_name, last_name, researcher_id, orcid, affiliations}, ...], ...}
"""

import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIMENSIONS_API_KEY = os.getenv('DIMENSIONS_API_KEY')
AUTH_URL = 'https://app.dimensions.ai/api/auth.json'
DSL_URL = 'https://app.dimensions.ai/api/dsl/v2'

FILTERED_CSV = 'filtered.csv'
CACHE_FILE = 'author_ids_cache.json'

BATCH_SIZE = 400
SLEEP_BETWEEN_BATCHES = 1  # seconds


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
    if resp.status_code == 403:
        # Token may have expired — raise so caller can re-auth
        raise PermissionError(f'API returned 403: {resp.text[:300]}')
    if resp.status_code != 200:
        print(f'  API error {resp.status_code}: {resp.text[:500]}')
        resp.raise_for_status()
    return resp.json()


def fetch_authors_for_dois(token: str, dois: list[str]) -> dict:
    """Query Dimensions for author data for a batch of DOIs.

    Returns {doi: [{first_name, last_name, researcher_id, orcid}, ...], ...}
    """
    escaped = ', '.join([f'"{d}"' for d in dois])
    query = (
        f'search publications where doi in [{escaped}] '
        f'return publications[doi+authors] limit {len(dois)}'
    )
    result = query_dimensions(token, query)

    author_map = {}
    for pub in result.get('publications', []):
        doi = pub.get('doi', '').lower()
        if not doi:
            continue
        authors = []
        for a in pub.get('authors', []):
            # Extract affiliation names from the affiliations array
            affiliations = [
                aff.get('name', '')
                for aff in a.get('affiliations', [])
                if aff.get('name')
            ]
            authors.append({
                'first_name': a.get('first_name', ''),
                'last_name': a.get('last_name', ''),
                'researcher_id': a.get('researcher_id'),
                'orcid': a.get('orcid', [None])[0] if a.get('orcid') else None,
                'affiliations': affiliations,
            })
        author_map[doi] = authors
    return author_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DIMENSIONS_API_KEY:
        print('ERROR: DIMENSIONS_API_KEY not set in environment / .env')
        return

    # Load data
    df = pd.read_csv(FILTERED_CSV, encoding='iso-8859-1')
    all_dois = df['OriginalPaperDOI'].dropna().unique()
    all_dois = [d.strip() for d in all_dois if str(d).strip()]
    print(f'Total entries: {len(df):,}')
    print(f'Entries with DOI: {len(all_dois):,}')

    # Load existing cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        print(f'Loaded cache with {len(cache):,} DOIs')

    # Migration: if any cached entry lacks 'affiliations', clear and re-fetch
    needs_migration = False
    for doi, authors in cache.items():
        for a in authors:
            if 'affiliations' not in a:
                needs_migration = True
                break
        if needs_migration:
            break
    if needs_migration:
        print('Cache entries lack affiliations field. Clearing cache for re-fetch.')
        cache = {}

    # Determine which DOIs still need fetching
    cached_dois = set(cache.keys())
    remaining = [d for d in all_dois if d.lower() not in cached_dois]
    print(f'DOIs remaining to fetch: {len(remaining):,}')

    if not remaining:
        print('All DOIs already cached. Nothing to do.')
        _print_summary(cache)
        return

    # Authenticate
    print('Authenticating with Dimensions API...')
    token = get_auth_token(DIMENSIONS_API_KEY)
    print('Authenticated successfully.\n')

    # Batch fetch
    n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(remaining), BATCH_SIZE), total=n_batches):
        batch = remaining[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f'Batch {batch_num}/{n_batches} ({len(batch)} DOIs)...', end=' ')

        try:
            result = fetch_authors_for_dois(token, batch)
        except PermissionError:
            print('token expired, re-authenticating...')
            token = get_auth_token(DIMENSIONS_API_KEY)
            result = fetch_authors_for_dois(token, batch)

        # Merge into cache (use lowercase DOIs as keys)
        for doi, authors in result.items():
            cache[doi.lower()] = authors

        # Mark DOIs not found in Dimensions as empty lists
        for doi in batch:
            if doi.lower() not in cache:
                cache[doi.lower()] = []

        print(f'found {len(result)}/{len(batch)}')

        # Save after each batch (crash-safe)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)

        if batch_num < n_batches:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f'\nDone. Cache saved to {CACHE_FILE}')
    _print_summary(cache)


def _print_summary(cache: dict):
    """Print summary statistics about the cached author data."""
    total_dois = len(cache)
    dois_with_authors = sum(1 for v in cache.values() if v)
    all_authors = [a for authors in cache.values() for a in authors]
    with_rid = sum(1 for a in all_authors if a.get('researcher_id'))
    with_orcid = sum(1 for a in all_authors if a.get('orcid'))
    with_affil = sum(1 for a in all_authors if a.get('affiliations'))

    print(f'\n--- Summary ---')
    print(f'Total DOIs in cache:          {total_dois:,}')
    print(f'DOIs with author data:        {dois_with_authors:,} '
          f'({dois_with_authors/total_dois*100:.1f}%)')
    print(f'Total author records:         {len(all_authors):,}')
    print(f'Authors with researcher_id:   {with_rid:,} '
          f'({with_rid/len(all_authors)*100:.1f}%)' if all_authors else '')
    print(f'Authors with ORCID:           {with_orcid:,} '
          f'({with_orcid/len(all_authors)*100:.1f}%)' if all_authors else '')
    print(f'Authors with affiliations:    {with_affil:,} '
          f'({with_affil/len(all_authors)*100:.1f}%)' if all_authors else '')


if __name__ == '__main__':
    main()
