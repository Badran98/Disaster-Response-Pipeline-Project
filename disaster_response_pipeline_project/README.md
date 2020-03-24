# Disaster Response Pipeline Project

- This project is main mission is  to classify disaster response messages through machine learning. 

## Project Motivation

In this project, I used data engineering pipelines , natural language processing, and machine learning skills to analyze message data that people sent during disasters to build a model to classifies these disaster messages. that would help in automatize informinging the nearest rescue station 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
