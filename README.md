# about
Post-popularity is a lightweight web service to predict the popularity of an article posted on social media using 
a regression model pipeline as a pickle file (not included in this repo) created and provided by a generous team of 
data scientists. Given valid and appropriately formatted JSON representing an article, post-popularity will return 
its score. An article at mean popularity have a score of 0.0; less popular than average would be negative 
(ex. -0.12568999); more popular would be positive (ex. 0.36271122).

The goal of this project is to expand my data engineering knowledge by exploring model-as-a-service. 

##  <a name="technologies"></a>key technologies
- [Python 3](https://www.python.org/downloads/)
- [Flask](http://flask.pocoo.org/)
- [Docker](https://www.docker.com/what-docker)

##  <a name="install-configure"></a>installation and configuration
To install post-popularity, first fork and clone this repo. Then, pop open a terminal and cd to your local 
`post-popularity` repo. Note that this service is expecting a `data` directory containing `.pkl` file(s) - 
the scikit_learn pipeline - which is not included here due to space and privacy concerns. Please ensure you've
added `./data/pipe.pkl` to your local `post-popularity` repo before starting the flask app.

#### Docker - recommended!
Ensure that you have Docker [installed](https://docs.docker.com/get-started/part2/)
and are familiar with [using Docker.](https://docs.docker.com/get-started/)

1. Create the docker image: `docker build -t post-popularity .`
2. Run the docker docker image containing the web app: `docker run -p 5000:5000 post-popularity`

This will start a containerized flask server that you can reach at `http://localhost:5000`.

#### Anaconda
Alternatively - if you have Anaconda 3.6 installed -  create a conda environment, activate it, and then run 
the following to install required dependencies:

`while read requirement; do conda install --yes $requirement; done < requirements.txt`

To run the flask server at `http://localhost:5000`:

`python app.py`

##  <a name="how-to"></a>usage
Posting to `http://localhost:5000/popularity-predictor` endpoint with valid and correctly formatted JSON representing 
a news article will return JSON containing a single prediction field with the predicted popularity of the article on
social media. Currently both timestamp and description fields are required to make a prediction.

Endpoint `/popularity-predictor`

**Input**: POST body containing JSON with valid timestamp and description fields

`{
	"timestamp": "2015-09-17 20:50:00", 
	"description": "About 20 western Virginia high school students were suspended Thursday after holding a rally to protest a new policy banning vehicles with Confederate flag symbols from the school parking lot and refusing to take off clothing displaying the symbol."
}`

**Output**: Returns status and JSON response

Posting valid and correct JSON will result in a `200 Success` response and the prediction JSON

`{
    "prediction": -0.09606195560398303
}`

Posting invalid or incorrect JSON will return a `400 Bad Request` response and optionally a JSON-formatted error

Example 1:

*Input*:

`{
    "timestamp": "2015-09-17 20:20:01", "description": ""
}`

*Output*:

`{
    "error": "'' is too short"
}`

Example 2:

*Input*:

`{
	“timestamp”: “22:07:02 2014-09-17”,
	“description”: “Police are investigating the death of a man whose body was found inside a home in Sammamish Wednesday morning as a homicide.”
}`

*Output*:

`{
    "error": "'Sept 17, 2014 - 1pm' does not match '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$'"
}`


#### tests
With the flask server running at `http://localhost:5000` and from the `post-popularity` dir, 
running `python tests/test_app.py` will output the result of some custom test functions (issues importing the app 
combined with time constraints prevented use of more traditional modules like pytest and unittest). Currently, 
these test for validity and correct formatting of input and for expected status codes. 

Output for tests passing:

`Test valid input PASS`

`Test invalid input PASS`

`Test invalid timestamp PASS`

`Test invalid description PASS`

Output for tests failing:

`Test valid input FAIL`

`Test invalid input FAIL`

`Test invalid timestamp FAIL`

`Test invalid description FAIL`


##  <a name="contact"></a>questions?
Please email Katie Simmons at [katie@katie.codes](mailto:katie@katie.codes)!
