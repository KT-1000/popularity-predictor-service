import requests


def test_valid_input():
    """Tests valid posted JSON """
    json_input = {
        'timestamp': '2015-09-17 20:50:00',
        'description': ''
    }
    r = requests.post('http://localhost:5000/popularity-predictor', json=json_input)
    try:
        assert r.status_code == 200
        assert r.json() == {'prediction': 0.3627112183586569}

        return "Test valid input PASS"

    except AssertionError as e:
        return "Test valid input FAIL {}".format(e)


def test_invalid_input():
    """Tests invalid posted JSON """
    json_input = {}
    r = requests.post('http://localhost:5000/popularity-predictor', json=json_input)
    try:
        assert r.status_code == 400
        assert r.json() == {'error': "'timestamp' is a required property"}

        return "Test invalid input PASS"

    except AssertionError as e:
        return "Test invalid input FAIL {}".format(e)


def test_invalid_timestamp():
    """Tests posted JSON for proper timestamp field"""
    json_input = {
        'timestamp': 'Sept 17 2015 20:50:00',
        'description': 'The Seattle Seahawks are a football team owned by Paul Allen.'
    }
    r = requests.post('http://localhost:5000/popularity-predictor', json=json_input)
    try:
        assert r.status_code == 400
        assert r.json() == {'error': "'Sept 17 2015 20:50:00' does not match '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$'"}

        return "Test invalid timestamp PASS"

    except AssertionError as e:
        return "Test invalid timestamp FAIL {}".format(e)


def test_invalid_description():
    """Tests posted JSON for proper description field"""
    json_input = {
        'timestamp': '2015-09-17 20:50:00',
        'description': ''}
    r = requests.post('http://localhost:5000/popularity-predictor', json=json_input)
    try:
        assert r.status_code == 400
        assert r.json() == {'error': "'' is too short"}

        return "Test invalid description PASS"

    except AssertionError as e:
        return "Test invalid description FAIL {}".format(e)


if __name__ == "__main__":
    print(test_valid_input())
    print(test_invalid_input())
    print(test_invalid_timestamp())
    print(test_invalid_description())
