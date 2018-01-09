# online_dating_recommender
This project is partnering with a dating website that mainly services Chinese community in US and Canada.
The goal of this project is to use data mining techniques and machine learning algorithms to build up a recommender system that recommends the matches to the users and maximizes the likelihood the user will be interested in the system recommended potential match.

1. Data:
There are 7 datasets:
- users
- profile
- track_charge_visitation
- view record
- bookmark record
- favorite record
- message record
all data are in csv files and filtered to 1 year

2. Model:
The two main models used in this project are:
- Content based filtering - analyzing the similarities between host profiles
- Collaborative filtering - comparing the user interest similarities
e user will be recommended the hosts that other similar users had been previously shown interest to.

3. Further object:
Furthermore, the users will make purchases on services ie: VIP, in order to message the matches recommended by the system if they are interested in. The more good matches the system recommend to the users, the more profit the company will potentially make.
