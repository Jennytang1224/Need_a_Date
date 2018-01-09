# online_dating_recommender

The goal of this project is to use data mining techniques and machine learning algorithms to build up a recommender system that recommends the matches to the users and maximizes the likelihood the user will be interested in the system recommended potential match.

For convenience, let's call
- the user we recommend matches to, 'user'
- the user we potentially recommend to another user, 'host'
Note: because we recommend users to users, every user is also another user's potential match; every host is also a user.

The two main models used in this project are content based filtering and collaborative filtering.

In the content based filtering model, by analyzing the similarities between host profiles, the user will be recommended those hosts who are sharing the similar profiles ('age','education','location'...) with the hosts, who had previously shown interest by this user.  

In the collaborative filtering model, by comparing the user interest similarities through historical user behaviors, the user will be recommended the hosts that other similar users had been previously shown interest to.

Furthermore, the users will make purchases on services ie: VIP, in order to message the matches recommended by the system if they are interested in. The more good matches the system recommend to the users, the more profit the company will potentially make.
