# Project3 - Reddit Classifier

### Content
- [Background](#Background)
- [Problem Statement](#Problem-Statement)
- [Data Dictionary](#Data-Dictionary)
- [EDA](#EDA)
- [1st Model - Parametric Model](#1st-Model-\--Parametric-Model)
- [2nd Model - Random Forest](#2nd-Model-\--Random-Forest)
- [Conclusion](#Conclusion)
- [Potential Improvements](#Potential-Improvements)


### Background
Due to multiple cyber attacks recently, many reddit posts which were 3 months and older were taken offline as their data were held hostage and copies were deleted. As the management team had decided to not give in to the demands of the hackers, the original data were not recovered. Luckily, the IT team has managed to recover the data partially with informations such as the title, selftext and other informations. Unfortunately, those data recovered did not have the subreddit name, url and links that would provide identifications to those posts.


### Problem Statement
Being a data scientist in Reddit, you made a suggestion to your manager that perhaps the subreddit name could be inferred/predicted from the remaining information recovered through modeling. As a proof of concept, you have been tasked by your manager to:
- Use the latest 3 months of data, complete with subreddit name, from 2 random subreddits 
- Determine how accurate is the suggested approach in identifying the reddit posts
    - Target goal is to achieve at least 90% accuracy


### Data Dictionary
Below are features used for the modeling

|Predictors|Data Type|Description|
|:---:|:---:|:---:|
|title|String|Title of the reddit post|
|selftext|String|Text in the reddit post|
|author_fullname|String|Author's unique ID|
|whitelist_status|String|Kind of ads allowed to be shown on the reddit post|
|link_flair_text|String|Tag that user can customized|

### EDA
#### What are the top words used in title for each subreddit?
For LinusTechTips subreddit, words that are useful in identifying it are words at the top of the figure below. Examples are linus, pc, help, ltt, gpu and tech. Whereas for TrashTaste, those words that are commonly used in the subreddit are the words at the bottom of the figure. Examples are connor, garnt, chris, joey, trash and taste.

For TrashTaste, those words that regularly appear are names or nicknames of the hosts or guests while for LinusTechTips, those words are more related to technology, pc and games related.
<img src="imgs\Counts and TFIDF of Title.PNG">


#### What are the top words used in selftext for each subreddit?
In general selftext is more commonly used in LinusTechTips and the words used greatly help to differentiate LinusTechTips from TrashTaste as seen in the chart below where the value is significantly higher for LinusTechTips as comapared to TrashTaste. Some examples of terms that are more common and unique for LinusTechTips are 'pc', 'just', like', 'new', 'cpu' and 'gpu'.
<img src="imgs\Counts and TFIDF of Selftext.PNG">


#### Any overlap in authors between the 2 subreddits?
There are no overlap between the 2 subreddits in terms of the authors when comparing the top 50 authors.
<img src="imgs\Author_Fullname.PNG">


#### How does the numbers/types of ads differ between the 2 subreddits?
Usually the content of the posts would affect the kind of ads shown on a website as no advertiser would like to associate themselves with a post that might be inappropriate. From the analysis, majority of posts from LinusTechTips are eligible for any ads as they are more family-friendly while TrashTaste, on the other hand, are only eligible for some ads. Both subreddits have some posts that are allowed to have NSFW (Not Suitable for Work) content.
<img src="imgs\Ads Eligible Posts.PNG">


#### How does the link_flair_text differs between the 2 subreddits?
Most posts for LinusTechTips has tags such as image, discussion, tech, post, video, question and wan. Whereas for TrashTaste, the common tags are meme, screeshot, clip, tweet and art.
<img src="imgs\Counts and TFIDF of link_flair_text.PNG">


### 1st Model - Parametric Model
For the parameteric model, 4 different models were ran where 2 vectorizers and 2 estimators were paired with each other to determine which is the best combination. After using the title of reddit posts as the data, it is concluded that TF-IDF vectorizer with logistic regression has the highest accuracy score among the 4 models with balanced sensitivity and specificity score. Hence, this combination is selected to be used for modeling when more features are added.

|Estimator|Vectorizers|Train Accuracy|TrainCV Accuracy|Test Accuracy|Sensitivity|Specificity|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Logisitic|Tfidf Vectorizer|0.908566|0.850501|0.847931|0.895551|0.800766|
|Naive Bayes|Tfidf Vectorizer|0.873596|0.827397|0.824832|0.767892|0.881226|
|Logisitic|Count Vectorizer|0.914341|0.845047|0.841193|0.907157|0.775862|
|Naive Bayes|Count Vectorizer|0.865897|0.831888|0.833494|0.779497|0.886973|

To improve the accraucy of the model, 4 additional predictors were added:-
- selftext
- link_flair_text
- author_fullname
- whitelist_status

By adding the 4 additional predictors, the accuracy for both trainCV and test greatly improved by another ~13% while maintaining the balance of the sensitivity and specificity score between model 1 and 6.

|Model No|Estimator|Vectorizers|Predictors|Train Accuracy|TrainCV Accuracy|Test Accuracy|Sensitivity|Specificity|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|Logisitic|Tfidf Vectorizer|title|0.908566|0.850501|0.847931|0.895551|0.800766|
|2|Logisitic|Tfidf Vectorizer|title<br>selftext|0.917548|0.868151|0.866217|0.903288|0.829501|
|3|Logisitic|Tfidf Vectorizer|title<br>selftext<br>link_flair_text|0.992941|0.971124|0.977863|0.978723|0.977011|
|4|Logisitic|Tfidf Vectorizer|title<br>selftext<br>link_flair_text<br>author_fullname|1.0|0.958292|0.963426|0.974854|0.952107|
|5|Logisitic|Tfidf Vectorizer|title<br>selftext<br>link_flair_text<br>whitelist_status|0.999679|0.998717|1.0|1.0|1.0|
|6|Logisitic|Tfidf Vectorizer|title<br>selftext<br>link_flair_text<br>author_fullname<br>whitelist_status|1.0|0.995508|0.998075|1.0|0.996168|

Looking at the confusion matrix for the test data, we can see that there are only 2 misclassifications whereby 2 posts from LinusTechTips got misclassified as TrashTaste when using model 6.
<img src="imgs\Tfidf_logisitic_confusion_matrix.PNG">


#### Production Model Insights on the Predictors
From the top 25 predictors for the model, it is observed that when the post's whitelist status is 'some_ads', the odds of the model classifying the post as 'TrashTaste' is ~10 times more likely while the having the word 'meme' in the selftext, title or link_flair_text would it ~2 times more likely for the post to be classified as 'TrashTaste' for each unit increase of those predictors assuming other predictors remain constant.


<img src="imgs\Top 25 Predictors.PNG">

On the other hand, words such as 'images', 'post', 'linus' and 'tech' tends to decrease the odds of classifying the post as 'TrashTaste'. For example, the word 'image' or 'post' would make ~0.4 and ~0.5 times likely for the post to be classified as 'TrashTaste'
<img src="imgs\Bottom 25 Predictors.PNG">


### 2nd Model - Random Forest
For the random forest, 2 models were compared with each other, one with countvectorizer and another with TFIDF vectorizer. In general, the train CV and test score of both models are quite similar to each other when just using the 'title' as predictors. Hence, for apple to apple comparison with the parametric model, the model with TFIDF vectorizer is selected for enhancement.

|Estimator|Vectorizers|Train Accuracy|TrainCV Accuracy|Test Accuracy|Sensitivity|Specificity|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Random Forest|Tfidf Vectorizer|0.877125|0.835738|0.829644|0.912959|0.747126|
|Random Forest|Count Vectorizer|0.862368|0.830282|0.833494|0.920696|0.747126|

Similar to the parametric model, 4 additional predictors were added to the model to improve the accuracy score by ~15% between model 1 and model 6.
- selftext
- link_flair_text
- author_fullname
- whitelist_status

|Model No|Estimator|Vectorizers|Predictors|Train Accuracy|TrainCV Accuracy|Test Accuracy|Sensitivity|Specificity|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|Random Forest|Tfidf Vectorizer|title|0.877125|0.835738|0.829644|0.912959|0.747126|
|2|Random Forest|Tfidf Vectorizer|title<br>selftext|0.835097|0.824510|0.826756|0.972920|0.681992|
|3|Random Forest|Tfidf Vectorizer|title<br>selftext<br>link_flair_text|0.954122|0.955083|0.954764|0.969052|0.940613|
|4|Random Forest|Tfidf Vectorizer|title<br>selftext<br>link_flair_text<br>author_fullname|0.951876|0.943528|0.943214|0.951644|0.934865|
|5|Random Forest|Tfidf Vectorizer|title<br>selftext<br>link_flair_text<br>whitelist_status|0.997433|0.996149|0.999037|1.0|0.998084|
|6|Random Forest|Tfidf Vectorizer|title<br>selftext<br>link_flair_text<br>author_fullname<br>whitelist_status|0.993262|0.994224|0.985563|0.978723|0.992337|


For the test data, we can see from the confusion matrix that there are 15 misclassifications whereby 11 TrashTaste posts were misclassified as LinusTechTips and vice versa for the other 4 posts.
<img src='imgs\Tfidf_rf_confusion_matrix.PNG'>


### Conclusion
It is possible to use data from the subreddit posts to infer which subreddit they came from with high accuracy. While both models are usable in predicting which subreddit the posts came from, the 1st model (parametric model) is still preferred as compared to the 2nd model (random forest) due to its slightly higher accuracy. When using the 2 models to classify posts from late 2020, the accuracy is 99% for the 1st model and 98% for the 2nd model.

||Logistic Regression|Random Forest|
|:---:|:---:|:---:|
||<img src='imgs\Tfidf_logisitic_confusion_matrix_validation.PNG'>|<img src='imgs\Tfidf_rf_confusion_matrix_validation.PNG'>|
|Accuracy|0.990209|0.982661|
|Sensitivity|0.986329|0.977351|
|Specificity|0.994086|0.987969|


### Potential Improvements
As the reason this current production model is showing very high accuracy is due to the significant difference between the whitelist_status of the 2 subreddit chosen, when training model for other subreddits, we could:-
- Include comments from each subreddit post
- author_flair_richtext if the usage rate is high for other subreddits
- Increasing the amount of dataset used for training
- Checking if there are images in the post
    - Certain subreddits tend to have more image in post
    
    
### Business Recommendations
- Reddit could show a list of subreddits to the users which the posts would be suitable to be posted in
- With the logistic regression model, since there are coefficients that indicate what are the key terms for the subreddits, Reddit could list down the current trend or hot words in the subreddit