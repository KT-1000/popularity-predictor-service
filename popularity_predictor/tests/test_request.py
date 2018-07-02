import requests

# simple test using example json and requests library
json = {
    'timestamp': '2015-09-17 20:50:00',
    'description': 'The Seattle Seahawks are a football team owned by Paul Allen.'
}
r = requests.post('http://localhost:5001/popularity-predictor', json=json)
print(r.status_code)
print(r.json())
