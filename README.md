# Amazon product recommender

Introducion

An effective recommender system is a critical part of successful e-commerce. Amazon.com attributes 35% of its revenue to its recommendation system. [source: https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers] Modern data science techniques enable recommendations from data including but not limited to shopping history, ratings (explicit and implicit), and website activity (types of web pages visited and duration on those web pages, exit rates, etc.) using techniques such as Singular Value Decomposition (SVD) on user-rating data and cosine similarity on vectorized item descriptions. 

The goal of this project is to use publicly available data comprised of product reviews and meta data from Amazon collected in 2018 to create a recommender system. This project uses the '5-core' reviews and meta data for all products in all departments. Given a user input of product ratings, the goal is to provide recommendations for each department a user rates products in.

The basic process goes as:
1. Take in user input of ratings 
2. Detect which departments the products belong to
3. Fit a separate model with all users and rating data for each department, which include the user's
4. Return recommendations for each department ('You may also like in these departments...')
5. Find other products based on cosine similarity to the highest-rated items
6. Return the additional recommendations ('Additional recommendations include...')


