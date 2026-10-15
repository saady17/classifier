from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Location:
    raw: str
    city: str | None = None
    region: str | None = None  # US state / Canadian province code, or UK nation
    country: str | None = None  # "US", "GB" or "CA"; None when unknown or elsewhere


@dataclass
class Job:
    source: str  # greenhouse, lever, ashby, workable, smartrecruiters, remotive, usajobs, adzuna
    external_id: str
    company: str
    title: str
    url: str
    description: str  # plain text
    locations: list[Location] = field(default_factory=list)
    remote: str = "onsite"  # onsite | hybrid | remote
    posted_at: str | None = None  # ISO 8601
    employment_type: str | None = None
    department: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None  # year | month | hour

    @property
    def id(self) -> str:
        return f"{self.source}:{self.external_id}"

    @property
    def countries(self) -> set[str]:
        return {loc.country for loc in self.locations if loc.country}
