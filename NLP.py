import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.metrics import classification_report

data = pd.read_excel("final_project.ods", dtype=str)

data.dropna(axis="index", inplace=True)

def filter_location(location):
    res = re.findall(r"\,\s[A-Z]{2,}", location)
    if res:
        return res[0][2:]
    else:
        return location

data["location"] = data["location"].apply(filter_location)


target = "career_level"
x=data.drop(target, axis="columns")
y=data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

over_sampler = RandomOverSampler(random_state=42, sampling_strategy={
    "director_business_unit_leader": 400,
    "specialist": 400,
    "managing_director_small_medium_company": 400
})

x_train, y_train = over_sampler.fit_resample(x_train, y_train)

#Preprocessing

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1,1)), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.7, min_df=0.01), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english", ngram_range=(1,1)), "industry")
])

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("features_selector", SelectPercentile(chi2, percentile=10)),
    ("model", RandomForestClassifier(random_state=42, n_jobs=4)),
])

# clf.fit(x_train, y_train)
#
# y_pred = clf.predict(x_test)
# print(classification_report(y_test, y_pred))

params = {
    "model__n_estimators": [100, 200, 300],
    "model__criterion": ["gini", "log_loss", "entropy"],
    "features_selector__percentile": [5, 10, 15, 20]
}

grid_search = GridSearchCV(param_grid=params, cv=4, n_jobs=4, estimator=clf, verbose=2, scoring="recall_weighted")

grid_search.fit(x_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)