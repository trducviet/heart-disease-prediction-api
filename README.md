# Heart Disease Prediction API

A Flask REST API for binary heart-disease risk prediction using a reproducible preprocessing pipeline and tuned Logistic Regression model.

> Educational portfolio project — not a clinical decision-support tool and not intended for diagnosis or treatment.

## Highlights

- Encodes five categorical variables with persisted `LabelEncoder` instances.
- Normalizes features with `MinMaxScaler`.
- Selects the five most informative features using **SelectKBest** with the chi-squared statistic.
- Trains **Logistic Regression** (`C=10`, L2 regularization, `lbfgs`).
- Returns class probabilities and validates API inputs before prediction.

## Recorded evaluation

The saved notebook experiment used an 80/20 train-test split with `random_state=42`.

| Metric | Result |
| --- | ---: |
| Test samples | 184 |
| Accuracy | **85.87%** |
| ROC-AUC | **0.904** |
| F1 — disease class | 0.87 |

Selected features: `Sex`, `ChestPainType`, `FastingBS`, `ExerciseAngina`, and `ST_Slope`.

## Run the API

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

The service starts at `http://127.0.0.1:8080`.

```bash
curl -X POST http://127.0.0.1:8080/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"input\": {\"Age\": 40, \"Sex\": \"M\", \"ChestPainType\": \"ATA\", \"RestingBP\": 140, \"Cholesterol\": 289, \"FastingBS\": 0, \"RestingECG\": \"Normal\", \"MaxHR\": 172, \"ExerciseAngina\": \"N\", \"Oldpeak\": 0, \"ST_Slope\": \"Up\"}}"
```

## Retrain

Add a CSV with the required schema as `data/heart.csv`, then run:

```bash
python train.py
```

This regenerates the model and preprocessing artifacts under `models/`.

## Privacy and data handling

The dataset is intentionally not published. See [`data/README.md`](data/README.md) for the expected schema and add only data you are authorized to use.
