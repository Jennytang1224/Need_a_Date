# online_dating_recommender
Do you need a date? ... or better yet! Would you like a successful date?

This project has partnered with an online dating website that mainly serves Chinese community in US and Canada.
The goal of this project is to use data mining techniques and machine learning algorithms to build a recommender system that maximize the successful matches.

Highlights:
- Hybrid Recommendation that use user profiles as content based filtering and user actions as collaborative filtering to capture latent features and preferences of users
- Use reverse match to solve cold start problem for new users

1.  Raw Data
- 1 million user profiles from past 10 years
- 27 million actions in 2017 (consisted of four datasets: visitation, bookmark, favoritism, message)
- Set threshold for 200,000 active users in 2017 with their 10 million actions
- Due to the landscape of the data, I decided to focus on heterosexual relationships (future work will look into minority groups)
- all data are in csv files

2. Setup
I splitted users into two groups by threshold:
- Existing users: total number of actions >= 10
- New users/ Inactive users: total number of actions < 10
In both user groups, I separated genders - female and male

Because dating is a user to user process, each user will get recommendations and also potentially be recommended to other users, let's call
- the user will receive recommendations: User
- the candidate will be recommended to other user: Host

3. Models

- Hybrid Recommender for Existing Users
There are two main pipes : user actions and user profile.

  - Four user actions datasets: visitation, bookmark, favoritism and message
    -  I merged four type of actions based on user_id and host_id,
      if a user viewed a host, and later the user messaged the host, I chose message as the final action to represent the level of like that the user has towards the host. Therefore, each user will only have one action towards host in my case.
    - I converted user actions to ratings based on scale 1-4:
      Visitation ->  1
    	Bookmark   ->  2
    	Favorite   ->  3
    	Message    ->  4
      and now I have a user to host rating table
    - For each host, I calculated his/her average rating received from other people represents the host's popularity
    - Use host profile features and average ratings as target to perform Gradient Boosting Regression and find feature importance by genders. Here's the result:
      ![Alt text](img/feature_importance.png)

  - User profile is consisted of numerical variables (age, latitude, photo_counts...) and categorical variables (income_range, occupation, body_type...)
    - I filled the missing values and dummfied the categorical variables to make profile consistent for all users
    - I also used a function to compare profile similarities between users by different technique of measurement for numerical and categorical variables. Inside the function, I multiply the score by the weight of each feature generated by GBR.
    - After I have two user similarity matrices for both female and male, I performed Non-Negative Matrix Factorization to find matrix W, which represents latent features between users.

  - Last step is to put both ratings table and latent feature matrix W into Matrix Factorization and output the recommendations!
    ![Alt text](img/model_existing_user.png)

- Reverse Match Recommender for New Users
Due to lack of action data, it's hard to predict new users' preferences. My solution is: instead of predicting whom this new user may like, I use reverse match method to search for other users who may be interested in this new user

  - Example: let's say there's a new male user comes in. Based on his location, I find closest male users in the his area, and compare their profile similarity to filter the users with most similar profile as him.

  - Then I look for females users have previously shown interest in these male users in rating table.

  - Now I can recommend these female users to this new male user, and it's very likely these female users will be interested in this new male user who are similar to the males they were interested in before.

  - After the new user's actions reaches the threshold, he will be categorized as existing user and receive recommendations from recommender system for existing user
  ![Alt text](img/model_new_user.png)

5. Model Comparison
I tried to simulate the current system as baseline model, which is filtering the closest hosts to the user and among them, recommend the most popular hosts to the user. Here's the comparison:
  ![Alt text](img/model_comparison.png)

6. Evaluation
- Offline evaluation
  Dating recommendation focuses on the quality instead of quantity, therefore, the matrix I used to measure my model performance is to see how good are my recommendations.

  - Below formula calculated the average ratings of hosts in recommendation whom have been previously rated by the user, the higher the average rating is, the better quality the recommendation is.
  ![Alt text](img/quality_formula.png)

  - Comparing my model with baseline model on quality of recommendations
  With the increase on numbers of recommendations, my model has an increase in candidates' quality while baseline model's quality is going down. For example, in the plot below, when numbers of recommendations = 300, my model increases the quality by approximately 31%
  ![Alt text](img/eval_plot.png)

- Online evaluation
  The data I am using to train my models on, was generated by the baseline model. Therefore, the best way to test the performace of the mode is to go through A/B test.
