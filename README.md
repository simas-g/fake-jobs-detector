# Fake Jobs Detector

Information Diamond project 10. A data-driven app for a job board's trust and safety team:

1. **Backlog insights**: SQL queries on the reviewed moderation backlog that show which listing traits go with confirmed scams.
2. **Check a posting**: a scam or legitimate verdict for a new posting from a 3-nearest-neighbour model, explained by the 3 most similar reviewed postings and their outcomes.
3. **Model performance**: how the model was chosen and how it scores.

## Run it

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# PostgreSQL connection (defaults shown); point these at bronto or any other server
export PGHOST=localhost PGPORT=5433 PGDATABASE=fakejobs PGUSER=$USER   # PGPASSWORD=... if needed

.venv/bin/python load_data.py      # create database + schema, import data/fake_jobs.csv
.venv/bin/python evaluate.py       # model selection + test-set results -> results/metrics.json
.venv/bin/python -m streamlit run app.py     # http://localhost:8501
```

If PostgreSQL can't be reached, the app builds the same schema in an in-memory SQLite database from the CSV and shows a warning, so the demo still works.

## Files

| File | What it does |
|---|---|
| `sql/schema.sql` | Normalised schema: `category` (8 rows) and `posting` (400 rows), with CHECK constraints for flags, hours and the salary pair |
| `sql/queries.sql` | Every backlog query the app shows, as named blocks |
| `load_data.py` | Creates the database and imports the CSV (empty salaries become `NULL`) |
| `db.py` | Connection (PostgreSQL, SQLite fallback), query loader |
| `features.py` | `prepare()`: the cleaning and feature engineering, used for both the backlog and new postings |
| `knn.py` | KNN written from scratch with numpy: standardization, Euclidean distance, majority vote |
| `evaluate.py` | Hold-out test set, 5-fold CV for scaling / features / k, check against scikit-learn |
| `app.py` | Streamlit app |

## Data cleaning and features

- 400 postings, 109 confirmed scams (27%; the brief says about 32%), no duplicates.
- 124 postings have no pay range. Instead of dropping them (a third of the data, with a higher scam rate) we add a `no_salary` flag.
- **Salary width** = `salary_max / salary_min` (0 when there is no range). In this backlog every legitimate range is at most 1.40× and every scam range is above 1.40×.
- **Posting hour** is encoded on a circle (sin, cos) so 23:00 and 00:00 are neighbours. Every posting that went live between 00:00 and 05:59 is a scam.
- Every feature is **standardized**, with the mean and standard deviation of the training data.
- Final 8 features: salary width, no salary, required experience, skills listed, capitals ratio, hour (sin, cos), vague words. The weak flags (`is_remote`, `free_mail_contact`), `category` and raw salaries made the distances noisier and lowered recall in cross-validation.

## Results (`python evaluate.py`)

80 held-out postings (20%, stratified), k = 3:

| | Predicted legitimate | Predicted scam |
|---|---|---|
| **Actually legitimate** | 58 | 0 |
| **Actually scam** | 0 | 22 |

Accuracy, precision and recall are all 100%. Always answering "legitimate" scores 72.5%. In 5-fold CV on the training set the final setup scores 99.7% accuracy, 100% precision and 98.9% recall; all raw columns with standardization score 92.8% / 95.7% / 77.0%.

Caveats: the test set is small, the backlog looks very cleanly separable (likely synthetic), and the features were partly chosen after exploring the whole file. Expect lower scores on real postings, especially once scammers learn which traits are checked.
