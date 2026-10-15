"""Turns free-text job locations into US / GB / CA (the three markets Jobzhub serves)."""

from __future__ import annotations

import re

from classifier.models import Location

US_STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland", "MA": "massachusetts",
    "MI": "michigan", "MN": "minnesota", "MS": "mississippi", "MO": "missouri", "MT": "montana",
    "NE": "nebraska", "NV": "nevada", "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico",
    "NY": "new york", "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
    "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina",
    "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont",
    "VA": "virginia", "WA": "washington", "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming",
    "DC": "district of columbia", "PR": "puerto rico",
}
CA_PROVINCES = {
    "AB": "alberta", "BC": "british columbia", "MB": "manitoba", "NB": "new brunswick",
    "NL": "newfoundland and labrador", "NS": "nova scotia", "ON": "ontario", "PE": "prince edward island",
    "QC": "quebec", "SK": "saskatchewan", "NT": "northwest territories", "NU": "nunavut", "YT": "yukon",
}
UK_NATIONS = {"england": "ENG", "scotland": "SCT", "wales": "WLS", "northern ireland": "NIR"}

COUNTRY_WORDS = {
    "US": ["united states", "united states of america", "usa", "u.s.", "u.s.a.", "us", "america"],
    "GB": ["united kingdom", "uk", "u.k.", "great britain", "britain", "gb", "england", "scotland", "wales", "northern ireland"],
    "CA": ["canada", "can"],
}
# Big cities that are often written without a state or country.
CITY_COUNTRY = {
    "new york city": ("US", "NY"), "nyc": ("US", "NY"), "new york": ("US", "NY"), "brooklyn": ("US", "NY"),
    "san francisco": ("US", "CA"), "sf": ("US", "CA"), "bay area": ("US", "CA"), "san jose": ("US", "CA"),
    "palo alto": ("US", "CA"), "mountain view": ("US", "CA"), "menlo park": ("US", "CA"), "sunnyvale": ("US", "CA"),
    "oakland": ("US", "CA"), "los angeles": ("US", "CA"), "san diego": ("US", "CA"), "irvine": ("US", "CA"),
    "santa monica": ("US", "CA"), "seattle": ("US", "WA"), "bellevue": ("US", "WA"), "redmond": ("US", "WA"),
    "austin": ("US", "TX"), "dallas": ("US", "TX"), "houston": ("US", "TX"), "boston": ("US", "MA"),
    "cambridge, ma": ("US", "MA"), "chicago": ("US", "IL"), "denver": ("US", "CO"), "boulder": ("US", "CO"),
    "atlanta": ("US", "GA"), "miami": ("US", "FL"), "washington dc": ("US", "DC"), "washington, d.c.": ("US", "DC"),
    "philadelphia": ("US", "PA"), "pittsburgh": ("US", "PA"), "phoenix": ("US", "AZ"), "salt lake city": ("US", "UT"),
    "portland": ("US", "OR"), "minneapolis": ("US", "MN"), "detroit": ("US", "MI"), "raleigh": ("US", "NC"),
    "nashville": ("US", "TN"), "columbus": ("US", "OH"), "toronto": ("CA", "ON"), "vancouver": ("CA", "BC"),
    "montreal": ("CA", "QC"), "montréal": ("CA", "QC"), "ottawa": ("CA", "ON"), "calgary": ("CA", "AB"),
    "waterloo": ("CA", "ON"), "edmonton": ("CA", "AB"), "london": ("GB", "ENG"), "manchester": ("GB", "ENG"),
    "edinburgh": ("GB", "SCT"), "glasgow": ("GB", "SCT"), "bristol": ("GB", "ENG"), "cambridge, uk": ("GB", "ENG"),
    "oxford": ("GB", "ENG"), "leeds": ("GB", "ENG"), "birmingham, uk": ("GB", "ENG"), "belfast": ("GB", "NIR"),
    "cardiff": ("GB", "WLS"),
}
# Places that would otherwise be misread (London, Ontario is in Canada; Cambridge could be MA or UK).
_REGION_ABBR = {**{k: ("US", k) for k in US_STATES}, **{k: ("CA", k) for k in CA_PROVINCES}}
_REGION_NAME = {**{v: ("US", k) for k, v in US_STATES.items()}, **{v: ("CA", k) for k, v in CA_PROVINCES.items()}}
_FOREIGN = re.compile(
    r"\b(germany|berlin|munich|france|paris|spain|madrid|barcelona|netherlands|amsterdam|ireland|dublin|india|"
    r"bangalore|bengaluru|hyderabad|pune|mumbai|delhi|singapore|australia|sydney|melbourne|japan|tokyo|"
    r"brazil|são paulo|sao paulo|mexico|poland|warsaw|portugal|lisbon|israel|tel aviv|switzerland|zurich|"
    r"sweden|stockholm|italy|milan|china|shanghai|beijing|hong kong|korea|seoul|argentina|colombia|bogota|"
    r"philippines|manila|pakistan|uae|dubai|south africa|nigeria|kenya|egypt|turkey|romania|ukraine|"
    r"czech|prague|austria|vienna|belgium|brussels|denmark|copenhagen|norway|oslo|finland|helsinki|"
    r"emea|apac|latam|europe|asia)\b"
)


def parse_location(raw: str) -> Location:
    text = (raw or "").strip()
    low = text.lower()
    loc = Location(raw=text)
    if not low:
        return loc

    parts = [p.strip() for p in re.split(r"[,/|;()\-–]+", text) if p.strip()]
    lowparts = [p.lower().strip(". ") for p in parts]

    # Explicit country words win ("Remote - US", "Toronto, Canada").
    for code, words in COUNTRY_WORDS.items():
        for w in words:
            if w in lowparts or re.search(rf"(^|[\s,(]){re.escape(w)}($|[\s,)])", low):
                if w in ("us", "can") and not (w in lowparts or re.search(rf"\b{w}\b", text.lower())):
                    continue
                loc.country = code
                break
        if loc.country:
            break

    # "City, ST" / "City, Province" (checked after country words but can refine region).
    for p, lp in zip(parts, lowparts):
        if p.isupper() and p in _REGION_ABBR:
            country, region = _REGION_ABBR[p]
            if loc.country in (None, country):
                loc.country, loc.region = country, region
        elif lp in _REGION_NAME:
            country, region = _REGION_NAME[lp]
            if loc.country in (None, country):
                loc.country, loc.region = country, region
        elif lp in UK_NATIONS:
            loc.country, loc.region = "GB", UK_NATIONS[lp]

    if loc.country is None or loc.city is None:
        for name, (country, region) in CITY_COUNTRY.items():
            if re.search(rf"(^|[\s,]){re.escape(name)}($|[\s,])", low):
                # "London, ON" is Canadian; trust an explicit region over the city table.
                if loc.country and loc.country != country:
                    continue
                loc.country = loc.country or country
                loc.region = loc.region or region
                loc.city = name.split(",")[0].title()
                break

    if loc.country is None and _FOREIGN.search(low):
        loc.country = None
    if loc.city is None and parts and lowparts[0] not in ("remote", "hybrid", "anywhere") and loc.country:
        first = parts[0]
        if first.lower() not in sum(COUNTRY_WORDS.values(), []) and not (first.isupper() and len(first) <= 3):
            loc.city = first
    return loc


_REMOTE = re.compile(r"\bremote\b|\banywhere\b|work from home|\bwfh\b|distributed|telecommut", re.I)
_HYBRID = re.compile(r"\bhybrid\b", re.I)


def remote_mode(*texts: str | None) -> str:
    joined = " ".join(t for t in texts if t)
    if _HYBRID.search(joined):
        return "hybrid"
    if _REMOTE.search(joined):
        return "remote"
    return "onsite"


TARGET_COUNTRIES = frozenset({"US", "GB", "CA"})
