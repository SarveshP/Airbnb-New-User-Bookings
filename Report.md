
# Airbnb New User Bookings

Sarvesh Prattipati

## I. Definition
***

### Project Overview
I stumbled upon this, Airbnb data, as I was looking to work on real world datasets. Airbnb, is an online marketplace where people list, discover, and book accomodations around the world. It has collected various datapoints about users. These data points are from exisiting users and new users, who haven't booked any accomodations yet. These data points help in prediction about its future users to provide them with customized suggestions in the aim to serve Airbnb customers better. This data is posted in Kaggle by Airbnb. Using user data effectively can help organizations increase metrics such as sales, user experience, customer retention and customer satisfaction. 


### Problem Statement
By accurately predicting where a new user will book their first travel experience, Airbnb can share more personalized content with their community, decrease the average time to first booking, and better forecast demand.

This data is sourced from Kaggle and includes demographics of users and their session data. The Strategy would be to utilize the data, perform data cleaning, data exploration using visualizations, and testing various algorithms for classification of the result.

The result will show the top 5 countries in which a new user can make his first booking through Airbnb.

id | country
--- | ---
5uwns89zht | NDF
5uwns89zht | US
5uwns89zht | other
5uwns89zht | FR
5uwns89zht | IT


### Datasets and Inputs
The dataset is composed of 5 CSV files. It has been obtained from a Kaggle Competition provided by Airbnb.

The most important file is the `train_users` file which has 16 columns containing user id, dates of account creation, first booking date, gender, age, signup method, signup app, destination etc along with the target variable `country_destination` and has 213451 rows. The `test_users` is similar to the previous file discussed but does not have our target variable and we have to use these to predict the destination and has 62096 rows. We have a good amount of data to work with to produce meaningful models.

The other three files contain web session logs (`sessions.csv`) for the users, summary statistics of destination countries (`countries`) and summary statistics of about the users age group, gender, etc. (`age_gender_bkts.csv`)

**File descriptions**
- **train_users.csv** - the training set of users
- **test_users.csv** - the test set of users
    - id: user id
    - date_account_created: the date of account creation
    - timestamp_first_active: timestamp of the first activity, note that it can be earlier than date_account_created or date_first_booking because a user can search before signing up
    - date_first_booking: date of first booking
    - gender
    - age
    - signup_method
    - signup_flow: the page a user came to signup up from
    - language: international language preference
    - affiliate_channel: what kind of paid marketing
    - affiliate_provider: where the marketing is e.g. google, craigslist, other
    - first_affiliate_tracked: whats the first marketing the user interacted with before the signing up
    - signup_app
    - first_device_type
    - first_browser
    - country_destination: this is the target variable you are to predict
-**sessions.csv** - web sessions log for users
    - user_id: to be joined with the column 'id' in users table
    - action
    - action_type
    - action_detail
    - device_type
    - secs_elapsed
- **countries.csv** - summary statistics of destination countries in this dataset and their locations
- **age_gender_bkts.csv** - summary statistics of users' age group, gender, country of destination

### Metrics

This is a multi-class classification problem. Since ranking of countries in the top 5 predicted matters, the evaluation metric NDCG (Normalized Discounted Cumulative Gain) is utilized for this project.

It gives us the measure of ranking quality based on the order of the most relavent document in the result set(ranked). 

Two assumptions are made in using DCG and its related measures.

- Highly relevant documents are more useful when appearing earlier in a search engine result list (have higher ranks)
- Highly relevant documents are more useful than marginally relevant documents, which are in turn more useful than non-relevant documents.

For example, if the most relavent document is in the bottom of the result set(ranked), then DCG decreases. Likewise, DCG increases if it's in the higher rank in the result set. 

$DCG_k=\sum_{i=1}^k\frac{2^{rel_i}-1}{\log_2{\left(i+1\right)}}$

$nDCG_k=\frac{DCG_k}{IDCG_k}$

${IDCG_k}$ is 1 in most scenarios.

where $rel_i$ is the relevance of the result at position $i$ and $k = 5$. 

**Example 1:**<br>
Let us assume for the user "Roneo" the ground truth/booking made is **NDF** (No Destination Found). For each new user, we are to make a maximum of 5 predictions from existing 12 countries for the first country of booking.

`ground_truth = [0 0 0 0 0 0 0 1 0 0 0 0] = NDF`

`predictions =
[ 0.0017368   0.00538507  0.07079092  0.00823662  0.03216999  0.00871099
  0.01175533  0.61346039  0.00279277  0.0009836   0.20804779  0.03592973]`
  
The indices of the above sorted predictions is as below:

`predictions_indices_order = [ 7 10  2 11  4  6  5  3  1  8  0  9]`

Based on above order_of_predictions, selecting top 5 relavent countries and marking the most relevant country as 1, while the rest have relevance = 0.

`
relavence_scores = [ 1       0       0       0        0]
Discounts        = [ 1.00000 1.58496 2.00000 2.32192  2.5849625 ]`

From the above relavence scores it is clear that the highest rank country is the ground truth.

`final_DCG_Score = 1.0`

Calculating for the above example

$rel_1 = 2^{1} - 1 = 1$<br> 
$rel_2 = 2^{0} - 1 = 0$<br>
$rel_3 = 2^{0} - 1 = 0$<br>
$rel_4 = 2^{0} - 1 = 0$<br>
$rel_5 = 2^{0} - 1 = 0$<br>

$DCG=\frac{2^{1}-1}{log_{2}(1+1)}+\frac{2^{0}-1}{log_{2}(2+1)}+\frac{2^{0}-1}{log_{2}(3+1)}+\frac{2^{0}-1}{log_{2}(4+1)}+\frac{2^{0}-1}{log_{2}(5+1)}=\frac{1}{1}+\frac{0}{1.58496}+\frac{0}{2}+\frac{0}{2.32192}+\frac{0}{2.58496}=1$

The DCG score of 1 shows that the classifier has given highest probability to the ground truth.

**Example 2:**<br>
Let us assume for the user "Hary" the ground truth/booking made is **US** country. For each new user, we are to make a maximum of 5 predictions from existing 12 countries for the first country of booking.

`ground_truth = [0 0 0 0 0 0 0 0 0 0 1 0] = US`

`predictions =
[ 0.00265752  0.0074776   0.03245427  0.01112108  0.03771622  0.01119133
  0.0147063   0.49428478  0.00356924  0.00104084  0.33389187  0.04988896]`
  
The indices of the above sorted predictions is as below:

`predictions_indices_order = [ 7 10 11  4  2  6  5  3  1  8  0  9]`

Based on above order_of_predictions, selecting top 5 relavent countries and marking the most relevant country as 1, while the rest have relevance = 0.

`
relavence_scores = [ 0       1       0       0        0]
Discounts        = [ 1.00000 1.58496 2.00000 2.32192  2.5849625 ]`

From the above relavence scores it is clear that the highest rank country is not the ground truth. Instead the classifier has given 2nd rank to the ground rank. This results in DCG score decrease.

`final_DCG_Score = 0.63092`

Calculating for the above example

$rel_1 = 2^{0} - 1 = 0$<br> 
$rel_2 = 2^{1} - 1 = 1$<br>
$rel_3 = 2^{0} - 1 = 0$<br>
$rel_4 = 2^{0} - 1 = 0$<br>
$rel_5 = 2^{0} - 1 = 0$<br>

$DCG=\frac{2^{0}-1}{log_{2}(1+1)}+\frac{2^{1}-1}{log_{2}(2+1)}+\frac{2^{0}-1}{log_{2}(3+1)}+\frac{2^{0}-1}{log_{2}(4+1)}+\frac{2^{0}-1}{log_{2}(5+1)}=\frac{0}{1}+\frac{1}{1.58496}+\frac{0}{2}+\frac{0}{2.32192}+\frac{0}{2.58496}=0.63092$

The decrease in DCG score to 0.63092 from 1 is because, the classifier has given 2nd rank to the ground rank. This means the classifier has not classfied rightly.

## II. Analysis
_____

### Data Exploration

The train users data file has 213451 rows, where each rows describes 15 features about the user. The target variable is `country_destination`. 


```python
Image(url= "img/Gender_Count.PNG", width=800, height=800)
```




<img src="img/Gender_Count.PNG" width="800" height="800"/>



The above visualization shows us that many of the customers haven't provided their gender. Although we can see interesting counts for male and female, comparitively females have upper hand.


```python
Image(url= "img/Countries_Count.PNG", width=800, height=800)
```




<img src="img/Countries_Count.PNG" width="800" height="800"/>



The above visualization shows us that most of the bookings fall under NDF (No destination found) and US countries. 

Most of the Exploratory Data Analysis is performed in the file `Stats&EDA.ipynb`. Check out this for related statistics and more visualizations. 

**Final observations** from the statistics and exploratory data analysis are:

- There is a relationship between gender and the first country booked.
- There is no relationship between the signup device and signup method.
- People don't prefer to travel too long.
- Majority of the users prefer countries with different languages, excluding United states. From the 3rd and 4th points it means more likely Americans prefer travelling to European countries, than to english speaking countries like Canada and Australia
- The size of the country does not influence the destination country.
- Median age of travellers is high in Great Britain, and more younsters travel in spain.
- Hungarian and Indonesian's made almost no bookings.
- Finnish users made the most bookings.
- Mac desktop users have made most of the bookings
- Most of the bookings have been made through Airbnb Webapp
- Who haven't disclosed their age are less likely to make a booking.

### Algorithms and Techniques

Given that this problem was a multi-class supervised classification problem, Decision Trees are good enough for this project as they can handle numerical and categorical data, missing data, along with multiple target classes. 

Although decision trees are good, I decided to use ensemble methods like random forest classifier and XGBoost to improve predictive power. 

**Random Forest** Classifiers fits number of decision trees on subsamples of a dataset and averages the results. 

**XGBoost** is an optimized gradient boosting library designed to be highly efficient, flexible and portable. It produces an ensemble of weak decision tree learners via additive training (boosting). 

## III. Methodology

### Data Preprocessing

- Replacing missing data with -1
- Dropping the "date_first_booking" column
- Removing erroneous ages
- Feature Engineering

#### date_first_booking

It was found that `date_first_booking` feature is not available for the testing dataset. This says that date_first_booking is only available for users who successfully booked a destination and it is implicit that `date_first_booking` would be missing. So, I decided to drop this feature.

#### Age

The below graph shows that age is less than 15 and greater than 100, which is surprising. So marking as NaN if age<15 and age>100



```python
Image(url= "img/Errounous_Age.PNG", width=800, height=800)
```




<img src="img/Errounous_Age.PNG" width="800" height="800"/>




```python
Image(url= "img/Age_correction.PNG", width=800, height=800)
```




<img src="img/Age_correction.PNG" width="800" height="800"/>



The erronous age is now corrected after data cleaning

#### Feature Engineering

**Date Account Created, Timestamp First Active**

The Date Account Created column is split into `dac_y`, `dac_m`, `dac_d`, `dac_wn` (week number), `dac_w_{}` (weekday)

Similar treatment was given to the Timestamp First Active with new columns added as `tfa_y`, `tfa_m`, `tfa_d`, `tfa_h` (hour), `tfa_wn` (week number), `tfa_w_{}` (weekday, it was further split into each day).

**Season (Engineered feature)**

Using the domain knowledge, season of booking can affect the destination choices. For example, people tend to visit cold places or beaches in summer, while the opposite is true in winter.

Engineered two new features `season_dac` and `season_tfa`.

**One-Hot Encoded features**

Other categorical features had to be further one hot encoded. `['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']` were encoded.

Feature Engineering ended up with `198` columns after data preprocessing.

### Implementation

#### Random Forest

I used sklearns RandomForestClassifier along with Grid Search for cross validation. The parameters used for GridSearch were, `min_samples_split` over `2,20` and `max_depth` over `6,8`. The best estimator is:

`RandomForestClassifier(
bootstrap=True, class_weight=None, criterion='gini',
max_depth=6, max_features='auto', max_leaf_nodes=None,
min_impurity_split=1e-07, min_samples_leaf=1,
min_samples_split=20, min_weight_fraction_leaf=0.0,
n_estimators=25, n_jobs=1, oob_score=False, random_state=101,
verbose=0, warm_start=False)`

This model was used to predict the test dataset. It got the following scores:
- Validation Score: 0.81756

#### XGBoost

Training the XGBoost over the entire feature set of 198 columns requires computational capability. Due to this limitation feature selection is the best option. Borrowing the best estimator parameters from the above Random Forest like max_depth, the initial XGBoost model is constructed with the below parameters:

The data was then trained over `[10,20,30,40]` top features and the maximum validation score was achieved for top 30 features. They were:


```python
Image(url= "img/top_30_features.PNG", width=800, height=800)
```




<img src="img/top_30_features.PNG" width="800" height="800"/>



Building the model with only these top 30 features and with the below parameters:

```
params = {'eta': 0.2,
          'max_depth': 6,
          'subsample': 0.5,
          'colsample_bytree': 0.5,
          'objective': 'multi:softprob',
          'num_class': 12}
          
```

This XGBoost model got the following scores:

- Validation Score of 0.81367


#### Hyperparameter Tuning

Finally, after playing with hyperparameter tuning, I tried the following parameters:

```
        {
        max_depth=7,
        learning_rate=0.18,
        n_estimators=80,
        objective="rank:pairwise",
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        missing=None,
        silent=True,
        nthread=-1,
        seed=111
        }
```

This model achieved:
- Validation Score of 0.82384

This was more than our previous tests and highest infact, `rank:pairwaise` was concluded to be a good objective for XGBoost.

**XGBoost Objective functions**
- `objective` was set as `multi:softprob` / `multi:softmax` for first instance and then for hyperparameter tuning the objective is set to `rank:pairwise`. 

## IV. Results

Given the most relavent country as 1 and rest of the countries as 0 for a user prediction.

The XGBoost validation score of 0.82384, which is close to relavance value 1, is a good NDCG score. It means that the model is able to classify the truth values with the highest ranks 82.3% times.


```python
Image(url= "img/results.PNG", width=800, height=800)
```




<img src="img/results.PNG" width="800" height="800"/>



The visualization tells us that most of the bookings will be made from the US, other, France, and Italy. While, NDF means No Destination Found, Airbnb says that if it is NDF, there wasn't any booking. This is for the same reason the `date_first_booking` values are missing if the destination country is NDF. 

**This tells us that 5/6th of the customers who are registered into the website have actually made bookings. Only to this 1/6 th of the customers, Airbnb has to focus on sending customized suggestions based on the predictions.**

### Business Recommendations

So the above top 5 recommendations can be directed to appropiate medium to attrack the new user on their first time booking.

1. People use smaller devices to surf the web and desktops to make the booking.  
2. Investing more resources into iOS apps can increase the experience to the users and also increase number of bookings.
3. Users who have specified their gender and age are more likely to make bookings. The unspecified gender people might just be surfing the Airbnb's.  
4. As Hungarian and Indonesian's made almost no bookings, these recommendations can be targeted towards them. So that these users will want to book the Airbnb's.

### Concluding Remarks

##### Learnings

Mainly, I learnt when and how to utilize the NDCG (Normalized Discounted Cumulative Gain) evaluation metric. I have also learned to perform feature engineering on real datasets, pickle the processed data, hyper parameter tuning on XGBoost, and finally extract information from the visualizations. 

#### Limitations
Mainly my computational capability. The sessions.csv is feature engineered but cannot be merged on with the initial 198 feature to form a complete feature set for this project. 
