import requests


def test_valid_input():
    """Tests valid posted JSON """
    json_input = {
        'timestamp': '2015-09-17 20:50:00',
        'description': 'The Seattle Seahawks are a football team owned by Paul Allen.'
    }
    r = requests.post('http://localhost:5000/popularity-predictor', json=json_input)
    try:
        assert r.status_code == 200
        assert r.json() == {'prediction': 0.3627112183586569}

        return "Test valid input PASS"

    except AssertionError as e:
        return "Test valid input FAIL {}".format(e)


def test_invalid_input():
    """Tests ivalid posted JSON """
    json_input = {}
    r = requests.post('http://localhost:5000/popularity-predictor', json=json_input)
    try:
        assert r.status_code == 400
        assert r.json() == {'error': "'timestamp' is a required property"}

        return "Test invalid input PASS"

    except AssertionError as e:
        return "Test invalid input FAIL {}".format(e)


if __name__ == "__main__":
    print(test_valid_input())
    print(test_invalid_input())
