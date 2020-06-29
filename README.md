# Amazon product recommender

## Introduction

An effective recommender system is a critical part of successful e-commerce. Amazon.com attributes 35% of its revenue to its complex recommendation system. [source: https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers] Modern data science techniques enable recommendations from data including but not limited to shopping history, ratings (explicit and implicit), and website activity (types of web pages visited and duration on those web pages, exit rates, etc.) using techniques such as Singular Value Decomposition (SVD) on user-rating data and cosine similarity on vectorized item descriptions. 

The goal of this project is to use publicly available data comprised of product reviews and meta data from Amazon collected in 2018 to create a recommender system. This project uses the '5-core' review datasets (where every reviewer has at least 5 reviews) and meta data for all products in all departments. Given a user input of product ratings, the goal is to provide recommendations for each department a user rates products in.

The basic process goes as:
1. Take in user input of ratings 
2. Detect which departments the products belong to
3. Fit a separate model with all users and rating data for each department, which include the user's
4. Return recommendations for each department 

## Usage

The recommender is given in src/recommender.py

usage: python recommender.py user_input.csv

"user_input.csv" is a csv file (without headers) with product asins on the left column and ratings in the right column (a example is provided in this directory as sample_submissions.csv)

Data source: https://nijianmo.github.io/amazon/index.html

A script for helping to collect the data and recreate the mongo database used in the project is provided in src/data_collection.py

Samples of ratings and meta data for Software are provided in the data directory. 

Additional necessary files (asins.csv) provided here: https://drive.google.com/file/d/1KoXFwZ84JS13kRM1MaQIYG3oNo2eqjVp/view?usp=sharing

## Model evaluation 

A script for model evaluation is given in src/modelling.py

Models (all at default settings):
SVD and NMF (from the Surprise scikit) with user ratings;
XGBoost with user ratings and TFIDF-transformed 'bags of words' with description, features, brand, and category text;
Cosine similarity on the TFIDF transformations

Metrics:
Overall RMSD and RMSD for products predicted to be in top 5% of ratings per user (SVD, NMF, XGBoost);
Average actual rating for products predicted to be in the top 5% of ratings or similarity per user (all models)

The Video Games, Musical Instruments, Software, Arts Crafts and Sewing, Industrial and Scientific, and Grocery and Gourmet Food departments were used in model evaluation. Data was split in half by time per department and models trained and tested on the earlier and later halves, respectively. The scores for the test portions for each model and metric is shown in the following figures (note that RMSD could not be tested for cosine similarity because that calculates a value between -1 and 1 and not an explicit rating):

![](/img/overall_rmsd.svg)
![](/img/top5_rmsd.svg)
![](/img/top5_actual.svg)

SVD has the lowest RMSD results of any method, while for the actual ratings of top 5% predicted products, the XGBoost method performs best except in Musical Instruments and Software (tied in Grocery and Gourmet Food). XGBoost, despite its higher RMSD, may be predicting more products that the user ended up rated highly, except in the above mentioned departments. The better or worse ability of XGBoost to predicted enjoyable products may depend on the usefulness of the text data utilized, which SVD does not rely upon.

The current version of the recommender uses the SVD model, partly because the wall time for execution is less than the XGBoost method with only a small compromise in scores of the top 5% of predicted products.

## Future directions

More departments could be used in model evaluation (preferably, all of them), and the tendancy for XGBoost to predict more products that the user would enjoy despite higher overall RMSD should be investigated.

