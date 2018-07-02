# about
Post-popularity is a lightweight web service to predict the popularity of an article posted on social media using 
a regression model pipeline as a pickle file (not included in this repo) created and provided by a generous team of 
data scientists. Given valid and appropriately formatted JSON representing an article, post-popularity will return 
its score. An article at mean popularity have a score of 0.0; less popular than average would be negative 
(ex. -0.12568999); more popular would be positive (ex. 0.36271122).

The goal of this project is to expand my data engineering knowledge by exploring model-as-a-service. 

##  <a name="technologies"></a>key technologies
- [Python 3](https://www.python.org/downloads/)
- [Falcon](https://falconframework.org/#sectionAbout)
- [Zappa?](https://www.zappa.io//):
    - [AWS lambda](https://aws.amazon.com/lambda/)
    - [AWS API Gateway](https://aws.amazon.com/api-gateway/)
- [Docker?](https://www.docker.com/what-docker)

##  <a name="install-configure"></a>installation and configuration
To install post-popularity, first fork and clone this repo. Then, pop open a terminal and cd to your local 
`post-popularity` repo.
 
#### Option 1: With Docker - recommended
Ensure that you have Docker [installed](https://docs.docker.com/get-started/part2/)
and are familiar with [using Docker.](https://docs.docker.com/get-started/)

1. Create the docker image: `docker build -t post-popularity .`
2. Run the docker docker image containing the web app: `docker run -p 5000:80 post-popularity`

This will start a web server at `http://localhost:5000`.

#### Option 2: Using virtualenv
1. Create a virtualenv, activate it and install dependencies:

`virtualenv venv`

`source venv bin activate`

`pip install -r requirements.txt`

Note that your commands may be different depending on your OS and global config.

2. Run the app:

##  <a name="how-to"></a>usage
`/model` endpoint
