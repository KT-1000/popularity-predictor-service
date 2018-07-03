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
added `/data/pipe.pkl` to your local `post-popularity` repo before starting the flask app.

#### Anaconda
If you have Anaconda 3.6 installed, simply create a conda environment, activate it, and then run the following to install 
required dependencies:

`while read requirement; do conda install --yes $requirement; done < requirements.txt`

To run the flask app's server:

`python app.py`

#### Docker
Alternatively, ensure that you have Docker [installed](https://docs.docker.com/get-started/part2/)
and are familiar with [using Docker.](https://docs.docker.com/get-started/)

1. Create the docker image: `docker build -t post-popularity .`
2. Run the docker docker image containing the web app: `docker run -p 5000:80 post-popularity`

This will start a web server at `http://localhost:5000`.

##  <a name="how-to"></a>usage
Posting to `http://localhost:5000/popularity-predictor` endpoint with valid and correctly formatted JSON representing 
a news article will return JSON containing a single prediction field with the predicted popularity of the article on
social media.

##  <a name="contact"></a>questions?
Please email Katie Simmons at [katie@katie.codes](mailto:katie@katie.codes)!
