from flask import Flask, request, Response
import json
from jsonschema import validate
import logging
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)


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

            return json_input

        except ValueError as e:
            logger.error({
                'error': 'Invalid JSON',
                'message': e
            })

    def load_model(self, model_path, model_version='0.0.0'):
        """Makes specified version of machine learning model available to use"""
        try:
            with open(model_path, 'rb') as fo:
                return joblib.load(fo)

        except IOError as e:
            logger.error({
                'error': 'Unable to unpickle model',
                'message': e
            })

    def make_prediction(self, validated_json, model):
        """Makes prediction by running appropriately formatted JSON through given machine learning model"""
        validated_arr = [[pd.Timestamp(validated_json['timestamp']), validated_json['description']]]
        article_info = np.array(validated_arr)
        prediction = model.predict(article_info)
        
        data = {
            'prediction': prediction[0]
        }
        js = json.dumps(data)

        response = Response(js,
                            status=200,
                            mimetype='application/json')

        return response


@app.route('/popularity-predictor', methods=['POST'])
def popularity_predictor():
    predictor = PopularityPredictor()

    # validate input JSON representing news article whose popularity is being predicted
    raw_input = request.get_json()
    valid_json = predictor.validate_input(raw_input)

    # unpickle trained ML model
    ml_model = (predictor.load_model('/Users/katiesimmons/Projects/post-popularity/popularity_predictor/data/pipe.pkl'))

    predicted_score = predictor.make_prediction(valid_json, ml_model)

    return predicted_score, 200


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    app.run(host='0.0.0.0', port=8002)
