# Career Level Classification with Machine Learning

🚀 This project demonstrates a complete end-to-end pipeline for predicting **career level** based on job-related data using a machine learning approach. It features data preprocessing, oversampling, feature extraction, model training, and evaluation — all wrapped in a clean and efficient pipeline.

## 📂 Dataset
The data is loaded from an `.ods` file and includes fields such as:

- `title`
- `location`
- `description`
- `function`
- `industry`
- `career_level` (Target)

## ⚙️ Preprocessing Steps

- **Missing values** are removed.
- **Location values** are simplified using regex to extract region/state information.
- **Imbalanced classes** are handled with `RandomOverSampler` for specific minority classes.
- **Textual data** (`title`, `description`) is transformed using `TfidfVectorizer`.
- **Categorical features** (`location`, `function`, `industry`) are encoded using `OneHotEncoder`.

## 🧠 Model & Training

- The model used is a `RandomForestClassifier`, a powerful and flexible ensemble learning method.
- Features are further selected using `SelectPercentile` based on the chi-squared (`chi2`) statistic.
- The training pipeline is wrapped using `Pipeline` and hyperparameter tuning is done using `GridSearchCV`.

## 🧪 Evaluation

After training, the model is evaluated on a separate test set using:
- **Classification Report** with precision, recall, and F1-score.

## 🔍 Highlighted Techniques

- `ColumnTransformer` for elegant column-wise preprocessing
- `Pipeline` for maintaining modularity and reproducibility
- `RandomOverSampler` to combat class imbalance
- `GridSearchCV` for hyperparameter optimization

## 📁 Requirements

- `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `openpyxl` or `odfpy` for reading `.ods` files

Install them with:

```bash
pip install pandas scikit-learn imbalanced-learn openpyxl odfpy
```

## 💡 Pro Tip

When working with job data, especially textual fields like descriptions and titles, TF-IDF vectorization can drastically improve the signal quality. Combine that with robust models and oversampling for minority classes, and you're on track for a solid predictive engine.

---

Made with ❤️ for machine learning enthusiasts who love clean pipelines and structured workflows.