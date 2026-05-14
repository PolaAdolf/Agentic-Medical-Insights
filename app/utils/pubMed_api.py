"""
pubmed_api.py
=============
PubMed API Wrapper & Utilities

Uses Biopython's Entrez module to interact with NCBI PubMed.

Public API
----------
- search_pubmed(query, max_results) → list[str]  (PMIDs)
- fetch_abstracts(pmids)            → list[dict]
"""

from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from Bio import Entrez

load_dotenv()

# NCBI requires an email address for all Entrez requests
Entrez.email   = os.getenv("PUBMED_EMAIL")
Entrez.api_key = os.getenv("PUBMED_API_KEY")   

_RATE_LIMIT_DELAY = 0.34  



# 1.  Search PubMed → PMIDs

def search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """
    Search PubMed and return a list of PMIDs.

    Parameters
    ----------
    query       : str   PubMed search query string.
    max_results : int   Maximum number of PMIDs to return.

    Returns
    -------
    list[str]  List of PMID strings.
    """
    max_results = min(max_results, int(os.getenv("MAX_PUBMED_RESULTS", 10)))
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            usehistory="y",
        )
        record = Entrez.read(handle)
        handle.close()
        time.sleep(_RATE_LIMIT_DELAY)
        return record.get("IdList", [])
    except Exception as exc:
        print(f"[PubMed] esearch error for '{query}': {exc}")
        return []


# 2.  Fetch Abstracts by PMID 

def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """
    Fetch PubMed records for a list of PMIDs.

    Parameters
    ----------
    pmids : list[str]   List of PubMed IDs.

    Returns
    -------
    list[dict]  Each dict: {pmid, title, abstract, authors, year, journal}
    """
    if not pmids:
        return []

    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="xml",
            retmode="xml",
        )
        records = Entrez.read(handle)
        handle.close()
        time.sleep(_RATE_LIMIT_DELAY)
    except Exception as exc:
        print(f"[PubMed] efetch error: {exc}")
        return []

    results: list[dict] = []

    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            art     = medline["Article"]

            pmid    = str(medline["PMID"])
            title   = str(art.get("ArticleTitle", ""))
            journal = str(art.get("Journal", {}).get("Title", ""))

            # Abstract
            abstract_obj = art.get("Abstract", {})
            abstract_texts = abstract_obj.get("AbstractText", [])
            if isinstance(abstract_texts, list):
                abstract = " ".join(str(t) for t in abstract_texts)
            else:
                abstract = str(abstract_texts)

            # Authors
            author_list = art.get("AuthorList", [])
            authors = []
            for a in author_list[:5]:  # cap at 5 authors
                last  = a.get("LastName", "")
                fore  = a.get("ForeName", "")
                if last:
                    authors.append(f"{last} {fore}".strip())

            # Year
            pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))

            results.append(
                {
                    "pmid":     pmid,
                    "title":    title,
                    "abstract": abstract,
                    "authors":  authors,
                    "year":     year,
                    "journal":  journal,
                }
            )
        except Exception as parse_exc:
            print(f"[PubMed] parse error: {parse_exc}")
            continue

    return results
