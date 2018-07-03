from flask import Flask, request, Response
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import logging
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
        self.model = self.load_model('./data/pipe.pkl')

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

            return True, ''

        except ValidationError as e:
            logger.error({
                'error': 'Invalid JSON',
                'message': e.message
            })
            return False, e.message

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

    def make_prediction(self, validated_json):
        """Makes prediction by running appropriately formatted JSON through given machine learning model"""
        validated_arr = [[pd.Timestamp(validated_json['timestamp']), validated_json['description']]]
        article_info = np.array(validated_arr)
        prediction = self.model.predict(article_info)

        return prediction[0]


predictor = PopularityPredictor()


@app.route('/popularity-predictor', methods=['POST'])
def popularity_predictor():
    # validate input JSON representing news article whose popularity is being predicted
    raw_input = request.get_json()
    is_valid_json, error_msg = predictor.validate_input(raw_input)
    #
    return_status = 200

    data = {}

    if is_valid_json:
        predicted_score = predictor.make_prediction(raw_input)
        data['prediction'] = predicted_score
    else:
        data['error'] = error_msg
        return_status = 400

    js = json.dumps(data)

    response = Response(js,
                        status=return_status,
                        mimetype='application/json')

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
