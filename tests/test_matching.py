"""Regression tests for the cases found while building the ranker on real job posts."""

import pytest

from classifier.jobparse import parse_job
from classifier.locations import parse_location
from classifier.resume import parse_resume
from classifier.skills import matcher
from classifier.titles import index, level


@pytest.mark.parametrize("raw,country,region", [
    ("San Francisco, CA", "US", "CA"), ("Remote - US", "US", None), ("London, UK", "GB", "ENG"),
    ("London, ON", "CA", "ON"), ("Toronto, Ontario, Canada", "CA", "ON"), ("Berlin, Germany", None, None),
    ("Remote, EMEA", None, None), ("Cambridge, MA", "US", "MA"), ("Remote", None, None),
])
def test_locations(raw, country, region):
    loc = parse_location(raw)
    assert (loc.country, loc.region) == (country, region)


@pytest.mark.parametrize("text,expected,absent", [
    ("Python, Go and PostgreSQL on k8s with Terraform", {"Python", "Go", "PostgreSQL", "Kubernetes", "Terraform"}, set()),
    ("Go above and beyond for customers", set(), {"Go"}),
    ("- Go/Python experience", {"Go", "Python"}, set()),
    ("You will react quickly; rest assured", set(), {"React", "REST APIs", "REST Assured"}),
    ("We use REST Assured for API tests", {"REST Assured"}, set()),
    ("Administer 10 ml doses and assist with ADL", set(), {"Machine learning"}),
    ("Ensure compliance with policy; networking events", set(), {"Compliance", "Networking"}),
    ("Class C license, vitamin c, a lean team", set(), {"C", "Lean"}),
    ("RN with BLS, ACLS, Epic EMR, med-surg and telemetry", {"Registered Nurse", "BLS", "ACLS", "Epic", "Medical-surgical", "Telemetry"}, set()),
    ("CI/CD, A/B testing, FP&A, Excel pivot tables", {"CI/CD", "A/B testing", "Financial analysis", "Excel"}, set()),
])
def test_skills(text, expected, absent):
    found = set(matcher().find(text))
    assert expected <= found, found
    assert not (absent & found), found


@pytest.mark.parametrize("title,soc,lvl", [
    ("Sr. Software Engineer II (Backend)", "15-1252.00", 3),
    ("Staff Accountant", "13-2011.00", 2),
    ("Staff Machine Learning Engineer", "15-2051.00", 4),
    ("Account Executive, Mid-Market", "41-3091.00", 2),
    ("Head of Marketing", "11-2021.00", 6),
    ("VP, Finance", "11-3031.00", 6),
    ("Registered Nurse - Med/Surg, Nights", "29-1141.00", 2),
    ("High School Math Teacher", "25-2031.00", 2),
    ("Warehouse Associate", "53-7065.00", 1),
])
def test_titles(title, soc, lvl):
    assert index().occupation(title)[0] == soc
    assert level(title) == lvl


def test_job_requirements():
    job = parse_job(
        "About us\nWe build things.\nRequirements:\n- 5+ years of Python experience. Kubernetes is a plus.\n"
        "- Bachelor's degree required; Master's preferred\nNice to have:\n- Terraform\n"
        "Benefits\n- Medical, dental and vision insurance\nWe are unable to sponsor visas for this role.",
        company="Acme",
    )
    assert job.min_years == 5
    assert "Python" in job.required and "Kubernetes" in job.preferred and "Terraform" in job.preferred
    assert job.degree == 3 and job.degree_required
    assert "Dental" not in job.required and "Insurance" not in job.required
    assert job.no_sponsorship and not job.sponsors


def test_employer_name_is_not_a_skill():
    assert "Epic" not in parse_job("Requirements:\n- Experience with Epic and Unreal", company="Epic Games").required


@pytest.mark.parametrize("name,title,years,degree,country", [
    ("swe", "Senior Software Engineer", 8.3, 3, "US"),
    ("nurse", "ICU Registered Nurse", 7.1, 3, "US"),
    ("accountant", "Senior Accountant", 6.8, 3, "CA"),
    ("warehouse", "Warehouse Associate", 7.8, 1, "US"),
])
def test_resumes(name, title, years, degree, country):
    r = parse_resume(open(f"tests/fixtures/resume_{name}.txt").read())
    assert r.current_title == title
    assert abs(r.years - years) < 0.6  # "Present" moves with today's date
    assert r.degree == degree and r.location_country == country


def test_rank_only_scores_the_given_jobs():
    import os

    if not os.path.exists("data/jobs.db"):
        pytest.skip("no job store")
    from classifier import store
    from classifier.ranker import Preferences, rank

    db = store.connect()
    ids = [r[0] for r in db.execute("SELECT id FROM jobs WHERE duplicate_of IS NULL AND countries LIKE '%,US,%' LIMIT 15")]
    resume = parse_resume(open("tests/fixtures/resume_nurse.txt").read())
    got = [m.job_id for m in rank(resume, Preferences(), limit=50, db=db, only=ids, group=False)]
    assert set(got) <= set(ids)


@pytest.mark.parametrize("line,title", [
    ("Data Analyst Feb 2026 - Present", "Data Analyst"),
    ("Financial Analyst Sept 2022 – Jul 2024", "Financial Analyst"),
    ("Data Scientist / ML Research Scientist (Graduate RA)Jan 2022 – Present", None),  # any title found is fine
    ("Present", ""),
    ("cleaned and normalized datasets", ""),
    ("PROJECT 4: Data Analytics & Forecast Modeling", ""),
])
def test_title_lines(line, title):
    from classifier.resume import _best_title

    found = _best_title([line])[0]
    if title is None:
        assert found
    elif title == "":
        assert found is None
    else:
        assert found == title


def test_side_job_does_not_decide_the_field():
    text = (
        "EXPERIENCE\nStudent Server Sept 2025 – Present\nServed guests\n"
        "Data Analyst Aug 2022 – July 2025\nSQL, Python, Tableau dashboards\n"
    )
    r = parse_resume(text)
    assert r.current_title == "Data Analyst"
