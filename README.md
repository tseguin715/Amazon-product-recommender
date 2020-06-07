# Amazon product recommender

Introducion

An effective recommender system is a critical part of successful e-commerce. Amazon.com attributes 35% of its revenue to its recommendation system. [source: https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers] Modern data science techniques enable recommendations from data including but not limited to shopping history, ratings (explicit and implicit), and website activity (types of web pages visited and duration on those web pages, exit rates, etc.) using techniques such as Singular Value Decomposition (SVD) on user-rating data and cosine similarity on vectorized item descriptions. 

The goal of this project is to use publicly available data comprised of product reviews and meta data from Amazon collected in 2018 to create a recommender system. This project uses the '5-core' review datasets (where every reviewer has at least 5 reviews) and meta data for all products in all departments. Given a user input of product ratings, the goal is to provide recommendations for each department a user rates products in.

The basic process goes as:
1. Take in user input of ratings 
2. Detect which departments the products belong to
3. Fit a separate model with all users and rating data for each department, which include the user's
4. Return recommendations for each department ('You may also like in these departments...')
5. Find other products based on cosine similarity to the highest-rated items
6. Return the additional recommendations ('Additional recommendations include...')

Data source: https://nijianmo.github.io/amazon/index.html

A script for helping to collect the data and recreate the mongo database used in the project is provided in /src (data_collection.py)

A sample of the data is provided: the 5-core reviews and product meta data for the Software department in /data (Software_5.json.gz and meta_Software.json.gz, respectively). 

Additional necessary files (asins.csv) are provided in this directory.

Model evaluation


