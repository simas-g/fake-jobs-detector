# Fake Jobs Detector

Information Diamond project, dataset 10 (Fake Jobs), Group 43.

A Streamlit app for a job board's trust and safety team:

- **Check a posting**: enter a new posting and get a verdict (scam or legitimate) from a
  3-nearest-neighbour model, with the 3 most similar reviewed postings next to it.
- **Backlog insights**: SQL queries (run with pandasql) on the 400 reviewed postings,
  plus filters to explore them.
- **Model performance**: test results and how the model was chosen.

## Files

| File | What it does |
|---|---|
| `sql/schema.sql` | The two tables: `category` and `posting` |
| `load_data.py` | Puts `data/fake_jobs.csv` into PostgreSQL |
| `db.py` | Loads the tables (PostgreSQL, or the CSV if there is no database) and runs SQL on them with pandasql |
| `features.py` | `prepare()`: turns a posting into the 8 numbers the model uses |
| `knn.py` | Our own KNN: scaling, Euclidean distance, majority vote |
| `evaluate.py` | Cross-validation, test set, comparison with scikit-learn; saves `model/knn.pkl` |
| `app.py` | The Streamlit app |

## Run it

```bash
pip install -r requirements.txt
createdb -p 5433 fakejobs      # once
python load_data.py            # CSV -> PostgreSQL
python evaluate.py             # evaluates and saves model/knn.pkl
streamlit run app.py
```

Without PostgreSQL the app reads the CSV directly, so it also runs on Streamlit Cloud.
