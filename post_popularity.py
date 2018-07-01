"""Post popularity prediction"""
from jsonschema import validate
import logging
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from sklearn.base import TransformerMixin
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LassoCV

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PopularityPredictor:
    """Predicts popularity of article posted to social media"""
    def __init__(self):
        pass

    def validate_input(self, json_input):
        """Ensures JSON is valid and in expected format to be used by machine learning model"""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": [
                "timestamp",
                "description"
            ],
            "properties": {
                "timestamp": {
                    "type": "string",
                    "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$"
                },
                "description": {
                    "type": "string",
                    "minLength": 1
                }
            }
        }

        try:
            validate(json_input, schema)

        except ValueError as e:
            logger.error({
                'error': 'Invalid JSON',
                'message': e
            })

        return json_input

    def load_model(self, model_path, model_version='0.0.0'):
        """Makes specified version of machine learning model available to use"""

        return joblib.load(model_path)

    def make_prediction(self, validated_json):
        """Makes prediction by running appropriately formatted JSON through given machine learning model"""
        model = self.load_model('data/pipe.pkl')
        validated_arr = [[pd.Timestamp(validated_json['timestamp']), validated_json['description']]]
        article_info = np.array(validated_arr)

        return model.predict(article_info)


if __name__ == '__main__':
    def select_time_column(X):
        return X[:, 0]


    def select_text_column(X):
        return X[:, 1]


    class DayOfWeekTransformer(TransformerMixin):
        def __init__(self):
            self.one_hot = OneHotEncoder()

        def fit(self, X, y=None):
            df = pd.DataFrame(X, columns=['ct'])
            df_dow = df['ct'].apply(lambda x: x.dayofweek)
            self.one_hot.fit(df_dow.values.reshape(-1, 1))
            return self

        def transform(self, X, **transform_params):
            df = pd.DataFrame(X, columns=['ct'])
            df_dow = df['ct'].apply(lambda x: x.dayofweek)
            return self.one_hot.transform(df_dow.values.reshape(-1, 1))

    class MonthTransformer(TransformerMixin):
        def __init__(self):
            self.one_hot = OneHotEncoder()

        def fit(self, X, y=None):
            df = pd.DataFrame(X, columns=['ct'])
            df_dow = df['ct'].apply(lambda x: x.month)
            self.one_hot.fit(df_dow.values.reshape(-1, 1))
            return self

        def transform(self, X, **transform_params):
            df = pd.DataFrame(X, columns=['ct'])
            df_dow = df['ct'].apply(lambda x: x.month)
            return self.one_hot.transform(df_dow.values.reshape(-1, 1))

    dow_trans = DayOfWeekTransformer()
    month_trans = MonthTransformer()
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)

    prediction = PopularityPredictor()
    valid_json = prediction.validate_input(
        {"timestamp": "2015-09-17 20:50:00",
         "description": "The Seattle Seahawks are a football team owned by Paul Allen."}
    )
    print(prediction.make_prediction(valid_json))
