# Amazon product recommender

Introducion

An effective recommender system is a critical part of successful e-commerce. Amazon.com attributes 35% of its revenue to its complex recommendation system. [source: https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers] Modern data science techniques enable recommendations from data including but not limited to shopping history, ratings (explicit and implicit), and website activity (types of web pages visited and duration on those web pages, exit rates, etc.) using techniques such as Singular Value Decomposition (SVD) on user-rating data and cosine similarity on vectorized item descriptions. 

The goal of this project is to use publicly available data comprised of product reviews and meta data from Amazon collected in 2018 to create a recommender system. This project uses the '5-core' review datasets (where every reviewer has at least 5 reviews) and meta data for all products in all departments. Given a user input of product ratings, the goal is to provide recommendations for each department a user rates products in.

The basic process goes as:
1. Take in user input of ratings 
2. Detect which departments the products belong to
3. Fit a separate model with all users and rating data for each department, which include the user's
4. Return recommendations for each department 

Data source: https://nijianmo.github.io/amazon/index.html

A script for helping to collect the data and recreate the mongo database used in the project is provided in src/data_collection.py

A sample of the data is provided: the 5-core reviews and product meta data for the Software department in data/ (Software_5.json.gz and meta_Software.json.gz, respectively). 

Additional necessary files (asins.csv) are provided in this directory.

Model evaluation

Models: 
SVD and NMF (from the Surprise scikit) with user ratings
XGBoost with user ratings and TFIDF-transformed 'bags of words' with description, features, brand, and category text
Cosine similarity on the TFIDF transformations 
(each model is at default settings)

Metrics:
Overall RMSD and RMSD for products predicted to be in top 5% of ratings per user (SVD, NMF, XGBoost)
Average actual rating for products predicted to be in the top 5% of ratings or similarity per user (all models)

The Video Games, Musical Instruments, Software, Arts Crafts and Sewing, Industrial and Scientific, and Grocery and Gourmet Food departments were used in model evaluation. Data was split in half by time per department and models trained and tested on the earlier and later halves, respectively. The scores for the test portions for each model and metric is shown in the following figures (note that RMSD could not be tested for cosine similarity because that calculates a value between -1 and 1 and not an explicit rating):

img
img
img

SVD has the lowest RMSD results of any method, while for the actual ratings of top 5% predicted products, the XGBoost method performs best except in Musical Instruments and Software (tied in Grocery and Gourmet Food). XGBoost, despite its lower RMSD, may be predicting more products that the user ended up rated highly, except in the above mentioned departments. The better or worse ability of XGBoost to predicted enjoyable products may depend on the usefulness of the text data utilized, which SVD does not rely upon.

The current version of the recommender uses the SVD model, partly because the wall time for execution is less than the XGBoost method with only a small compromise in scores of the top 5% of predicted products.



