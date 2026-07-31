# Data Card

## Source

This project uses the Yelp Open Dataset locally. The repository does not redistribute the raw JSON files.

Expected local files:

```text
data/raw/yelp/yelp_academic_dataset_business.json
data/raw/yelp/yelp_academic_dataset_user.json
data/raw/yelp/yelp_academic_dataset_review.json
data/raw/yelp/yelp_academic_dataset_checkin.json
data/raw/yelp/yelp_academic_dataset_tip.json
```

## Scope

The tracked analysis focuses on New Orleans, Louisiana, and evaluates pre-COVID target months only:

| Split | Target months |
| --- | --- |
| Train | 2015-02 to 2017-12 |
| Validation | 2018-01 to 2018-12 |
| Test | 2019-01 to 2019-12 |

## What Is Committed

- Summary metrics and interpretation CSVs in `outputs/*.csv`.
- Report figures in `outputs/figures/`.
- `.gitkeep` placeholders under `data/`.
- Synthetic test fixtures created at test runtime.

## What Is Not Committed

- Raw Yelp JSON files.
- City-level interim extracts.
- Processed modeling tables.
- Full prediction CSVs.
- Local academic PDF exports.

These files can be large, can contain Yelp record identifiers or text, and may be restricted by Yelp dataset terms.

## Publication Guardrails

- Keep raw/interim/processed data ignored by Git.
- Publish aggregate metrics, figures, and a small number of named business case-study rows
  when they are necessary to make the model behavior inspectable.
- Do not commit full review text, user-level tables, or unrestricted prediction dumps.
- Run `cf-yelp validate-outputs` before sharing the repo.
