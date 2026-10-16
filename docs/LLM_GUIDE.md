# Using LLMs to build classifiers (recommendations)

Lessons from building this ranker: where LLMs helped, where they must stay out, and the traps we hit.
Useful for a person directing an LLM coding agent, and for the agent itself.

## 1. Keep the LLM out of the product, put it in the exam room

- The shipped ranker is deterministic: rules, dictionaries, BM25/TF-IDF and gradient-boosted trees.
  It is cheap, fast on a small server, explainable ("missing Terraform, asks for 5+ years") and auditable.
- LLMs are the **exam markers**: they grade the ranker's output so changes can be measured. Without a
  grader, "it looks better" is a guess; with one, every change is a number on a held-out set.
- LLMs as **builders**: writing fetchers, parsers, regexes, tests and docs is where a coding agent
  saves the most time. Review every rule it writes against real data (see section 4).

## 2. How to use LLMs as judges

1. **Same rubric as humans**, with explicit anchors (3 great / 2 good, one gap / 1 stretch / 0 not a
   fit, including license and level rules). Ask for a short reason with evidence; reasons are the best
   source of new features.
2. **Several independent models, blind.** One judge has blind spots; three different vendors rated
   independently and combined by **median/majority** cut noise sharply. Send three-way splits to a
   referee that reads all reasonings.
3. **Batch** 5 jobs per request with the resume once; temperature 0; JSON output; validate that every
   job got a rating in range and retry otherwise.
4. **Redact** names, emails, phones, links and street addresses before sending; prefer providers that
   do not store or train on prompts (OpenRouter `provider.data_collection = "deny"`).
5. **Check judges against people** on a small set (here 88% within one point), and show raters the
   models' ratings only **after** they rate, so the check is not biased.
6. **Measure judge noise** (how often two judges split good vs not good). It caps any metric built on
   their labels; set targets with that ceiling in mind.
7. Cost here: ~30k judgments across three models for roughly US$10.

## 3. Evaluation discipline (the part agents get wrong)

- **Fixed splits** (train / locked test / fresh batch). Tune and pick fixes on train only; look at test
  rarely; treat a batch you inspected for failures as no longer pristine.
- **Pool before rating**: mix the system's picks, a baseline's picks and random jobs, shuffled, so raters
  cannot tell who chose what.
- **Measure the product, not just the proxy.** Pools say whether ordering improves; the real top 10
  from the full catalogue says what users see. Both matter.
- **Compute NDCG against the jobs actually ranked.** We once let a resume's other rated jobs raise the
  ideal and every system, even random order, "dropped" 20%. When random order changes, the metric broke,
  not the model.
- **State precision targets per label** ("Good match is right 90% of the time") and calibrate
  thresholds to them; refuse to issue a label that cannot meet its target.
- Never claim an improvement inside the noise; re-run baselines on the same data in the same run.

## 4. Building the deterministic parts with an agent

- **Start from public taxonomies** (O*NET occupations, titles, tools) and add a hand-curated core
  (~350 skills with aliases). The curated core matters more than the long tail.
- **Every alias is a false-positive risk.** Real cases we hit: "ml" in "10 ml", "react quickly",
  "rest assured" (REST Assured), "Go above and beyond", "C" in "Class C license", "Present" read as
  President, "Indigenous persons" matching "US persons", benefits text ("dental") read as skills, the
  employer's own product as a requirement, a French post in Montreal. Make the agent run each new rule
  on a few thousand real documents and print the most frequent hits before accepting it.
- **Parse real resumes early.** Synthetic samples hid that most resumes put title and dates on one line;
  119 of 275 real resumes had no readable work history until that was fixed.
- **Prefer hard filters for dealbreakers** (country, sponsorship, clearance, ITAR, language, internship
  eligibility) and learned scoring for everything else.
- **Do not infer protected or sensitive traits** (citizenship, nationality) from names or schools; ask
  the person instead.
- **Version derived data** (`READER_VERSION`) so logic changes re-read stored jobs.
- **Learned models after features stall.** A weighted sum plateaued; gradient-boosted trees on ~47
  numbers lifted held-out NDCG 0.70 → 0.75 and top-5 good fits 47% → 62%. Train on the system's own top
  mistakes (rate its real top 10, retrain) — each round helped.

## 5. Ethics and data rules for agents

- Official feeds and APIs only; no scraping of sites whose terms forbid it.
- Personal data (resumes, ratings) stays on the server, git-ignored; never paste it into prompts beyond
  redacted evaluation text, never commit it.
- Secrets live in `.env` (chmod 600). If a key is pasted into a chat, rotate it.

## 6. Prompt templates

Judge (system prompt) — see `RUBRIC` and `CROSSCHECK` in `classifier/judge.py`.

Task prompt for a coding agent extending a rule:

> Add detection for X in `jobparse.py`. Before finishing: run it over 3,000 random jobs from
> `data/jobs.db`, print the 20 most common matched snippets and any false positives you find, add
> regression tests for both true and false cases, bump `READER_VERSION`, recompute features, and report
> `quickeval train` before and after. Do not look at the test split.
