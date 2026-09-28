"""OpenAlex literature search library.

OpenAlex is a free, fully open bibliographic database (~250 M works).
No API key is required; provide an email for the polite pool (higher rate limits).

Adapted from zeropaper/templates/utils/openalex/openalex.py (MIT-style, free to use).

Public API:
    search(query, *, venue, from_year, to_year, work_type, limit, email)
    get_cites(work_id, limit, email)    — forward citations
    get_refs(work_id, limit, email)     — backward references
    get_work(work_id_or_doi, email)     — single work metadata

Email for polite pool: set env var OPENALEX_EMAIL or pass email= directly.
Docs: https://docs.openalex.org/
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.openalex.org"
TIMEOUT = 12
RETRIES = 2
BACKOFF = 1.5

WORK_FIELDS = (
    "id,doi,title,display_name,publication_year,publication_date,"
    "primary_location,authorships,cited_by_count,referenced_works,"
    "open_access,language,type"
)

VENUE_ALIASES: dict[str, str] = {
    "jf":     "S5353659",
    "jfe":    "S149240962",
    "rfs":    "S170137484",
    "jfqa":   "S193228710",
    "ms":     "S33323087",
    "aer":    "S23254222",
    "qje":    "S203860005",
    "jpe":    "S95323914",
    "ecma":   "S95464858",
    "restud": "S88935262",
    "jme":    "S6711363",
}


def _email() -> str:
    val = os.environ.get("OPENALEX_EMAIL", "").strip().strip('"').strip("'")
    if val:
        return val
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        env = parent / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("OPENALEX_EMAIL="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    return ""


def _get(path: str, params: dict, mailto: str = "") -> dict:
    if mailto:
        params = {**params, "mailto": mailto}
    url = f"{API}{path}?{urllib.parse.urlencode(params, safe=',:|')}"
    last_err: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = exc
            if attempt < RETRIES:
                time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f"OpenAlex GET {url} failed: {last_err}")


def _resolve_venue(token: str, mailto: str) -> str | None:
    low = token.lower()
    if low in VENUE_ALIASES:
        return VENUE_ALIASES[low]
    if token.startswith("S") and token[1:].isdigit():
        return token
    payload = _get("/sources", {"search": token, "per-page": "1"}, mailto)
    results = payload.get("results") or []
    if not results:
        return None
    return results[0]["id"].rsplit("/", 1)[-1]


def _build_filter(
    venues: list[str],
    from_year: int | None,
    to_year: int | None,
    work_type: str | None,
) -> str:
    parts: list[str] = []
    if venues:
        parts.append("primary_location.source.id:" + "|".join(venues))
    if from_year and to_year:
        parts.append(f"publication_year:{from_year}-{to_year}")
    elif from_year:
        parts.append(f"publication_year:{from_year}-3000")
    elif to_year:
        parts.append(f"publication_year:1000-{to_year}")
    if work_type:
        parts.append(f"type:{work_type}")
    return ",".join(parts)


def _project(work: dict) -> dict:
    primary = work.get("primary_location") or {}
    src = primary.get("source") or {}
    authors = [
        (au.get("author") or {}).get("display_name")
        for au in (work.get("authorships") or [])[:8]
        if (au.get("author") or {}).get("display_name")
    ]
    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title") or work.get("display_name"),
        "year": work.get("publication_year"),
        "authors": authors,
        "venue": src.get("display_name"),
        "cited_by_count": work.get("cited_by_count"),
        "type": work.get("type"),
        "open_access_pdf": (work.get("open_access") or {}).get("oa_url"),
        "n_references": len(work.get("referenced_works") or []),
    }


def _normalize_id(s: str) -> str:
    s = s.strip()
    if s.startswith("https://openalex.org/"):
        return s.rsplit("/", 1)[-1]
    if s.startswith("doi:") or s.startswith("10."):
        return f"doi:{s}" if not s.startswith("doi:") else s
    if s.startswith("https://doi.org/"):
        return f"doi:{s.split('https://doi.org/', 1)[1]}"
    return s


def search(
    query: str,
    *,
    venue: str | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
    work_type: str = "article",
    sort: str = "relevance",
    limit: int = 10,
    email: str | None = None,
) -> list[dict]:
    """Keyword search on OpenAlex.

    Args:
        query:     Free-text search string.
        venue:     Venue alias (e.g. 'jf', 'jfe') or OpenAlex source ID.
        from_year: Earliest publication year (inclusive).
        to_year:   Latest publication year (inclusive).
        work_type: Filter by type, e.g. 'article', 'preprint'. '' to skip.
        sort:      'relevance' (default) or 'cited' (by citation count desc).
        limit:     Max results to return.
        email:     Polite pool email; falls back to OPENALEX_EMAIL env var.

    Returns:
        List of work dicts with keys: id, doi, title, year, authors,
        venue, cited_by_count, type, open_access_pdf, n_references.
    """
    mailto = email or _email()
    venues: list[str] = []
    if venue:
        vid = _resolve_venue(venue, mailto)
        if vid:
            venues.append(vid)

    flt = _build_filter(venues, from_year, to_year, work_type or None)
    sort_param = "cited_by_count:desc" if sort == "cited" else "relevance_score:desc"
    params: dict = {
        "search": query,
        "per-page": str(limit),
        "sort": sort_param,
        "select": WORK_FIELDS,
    }
    if flt:
        params["filter"] = flt

    payload = _get("/works", params, mailto)
    return [_project(w) for w in (payload.get("results") or [])]


def get_cites(
    work_id: str,
    limit: int = 20,
    email: str | None = None,
) -> list[dict]:
    """Forward citations — works that cite `work_id`.

    Args:
        work_id: OpenAlex ID (W1234567), DOI, or https://doi.org/... URL.
        limit:   Max results.
        email:   Polite pool email.

    Returns:
        List of work dicts sorted by citation count (desc).
    """
    mailto = email or _email()
    wid = _normalize_id(work_id)
    target = _get(f"/works/{wid}", {"select": "id,title"}, mailto)
    oid = target["id"].rsplit("/", 1)[-1]
    params = {
        "filter": f"cites:{oid}",
        "per-page": str(limit),
        "sort": "cited_by_count:desc",
        "select": WORK_FIELDS,
    }
    payload = _get("/works", params, mailto)
    return [_project(w) for w in (payload.get("results") or [])]


def get_refs(
    work_id: str,
    limit: int = 20,
    email: str | None = None,
) -> list[dict]:
    """Backward references — works cited by `work_id`.

    Args:
        work_id: OpenAlex ID, DOI, or https://doi.org/... URL.
        limit:   Max results.
        email:   Polite pool email.

    Returns:
        List of work dicts.
    """
    mailto = email or _email()
    wid = _normalize_id(work_id)
    target = _get(f"/works/{wid}", {"select": "id,title,referenced_works"}, mailto)
    refs = target.get("referenced_works") or []
    if not refs:
        return []
    refs = refs[:limit]
    ids = "|".join(r.rsplit("/", 1)[-1] for r in refs)
    params = {
        "filter": f"openalex_id:{ids}",
        "per-page": str(len(refs)),
        "select": WORK_FIELDS,
    }
    payload = _get("/works", params, mailto)
    return [_project(w) for w in (payload.get("results") or [])]


def get_work(work_id_or_doi: str, email: str | None = None) -> dict:
    """Fetch full metadata for a single work.

    Args:
        work_id_or_doi: OpenAlex ID, DOI string, or https://doi.org/... URL.
        email:          Polite pool email.

    Returns:
        Work dict with all projected fields.
    """
    mailto = email or _email()
    wid = _normalize_id(work_id_or_doi)
    params = {"select": WORK_FIELDS}
    work = _get(f"/works/{wid}", params, mailto)
    return _project(work)
