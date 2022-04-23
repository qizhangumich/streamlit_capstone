import streamlit as st
import pandas as pd


list_a = [['reviewerID',	0],
['reviewTime',	0],
['asin',	0],
['title',	0],
['brand',	0],
['overall',	0],
['imageURLHighRes',	0]]
df = pd.DataFrame(list_a, columns=['Features', 'Number of missing values across columns'])

list_a = [['Total no of ratings', 	518714],
['Total no of users', 	465432],
['Total no of products',	98043],
['Total no of review',	518714]]


df2 = pd.DataFrame(list_a, columns=['Features', 'Number of Data'])

list_a = [['rating',	'float64'],
['user_id',	'object'] ,
['product_id',	'object'],
['image',	'object']]


df3 = pd.DataFrame(list_a, columns=['Features', 'DType'])



list_a = [
['B00YP2TNZ2',	3.849057,	636,	1.0],
['B000PHANNM',	4.704787,	376,	2.0],
['B0183JQHCO',	3.267030,	367,	3.0],
['B00XTM0ZPG',	4.203647,	329,	4.0],
['B00ZW3SCF0',	3.917683,	328,	5.0],
['B00UPN42RY',	4.264984,	317,	6.0],
['B011MP1ODS',	4.442675,	314,	7.0],
['B01CDV7TNE',	4.694444,	288,	8.0],
['B00LKWYX2I',	4.367133,	286,	9.0],
['B00RLSCLJM',	4.838596,	285,	10.0]]

df4 = pd.DataFrame(list_a, columns=['Product_id', 'Rating', 'Counts', 'Rank'])

list_a = [
['SVDpp',	1.129775,	3.052613,	0.206412],
['KNNBaseline',	1.149961,	1.921499,	0.548754],
['SVD',	1.153295,	1.307575,	0.077567],
['KNNBasic',	1.196293,	1.947534,	0.497492],
['BaselineOnly',	1.19978,	0.094804,	0.057198],
['KNNWithMeans',	1.239685,	1.950055,	0.746992],
['SlopeOne',	1.245341, 3.55694,	0.225664],
['NMF',	1.343935,	3.79399,	0.105652,],
]

df5 = pd.DataFrame(list_a, columns=['Algorithm', 'Test_RMSE', 'Fit_Time', 'Test_Time'])


list_a = [['Q1', '32%', '68%'],
['Q2', '24%', '76%'],
['Q3', '35%', '65%'],
['Q4', '27%', '73%'],
['Q5', '19%', '81%'],
['Q6', '24%', '76%'],
['Q7', '13%', '87%'],
['Q8', '23%', '77%'],
['Q9', '36%', '64%'],
['Q10', '37%', '63%'],
['Q11', '28%', '72%'],
['Q12', '23%', '77%'],
['Q13', '18%', '82%'],
['Q14', '19%', '81%']]

df6 = pd.DataFrame(list_a, columns=['Popularity', 'Collaborative', 'Filtering'])

list_a = [
['Random', '9.217172513383314'],
['Popularity-based', '4.3923459172729755'],
['Collaborative Filtering', '11.17091035112718']]
df7 = pd.DataFrame(list_a, columns=['Method', 'Score'])


st.write("""


    # Fashion Recommender System




Advisor
Qiaozhu Mei




Team Members
ZhiPeng, Luo, ChihShen, Hsu, Qi, Zhang




##	Abstract
    
There are three major approaches in building recommender systems in this project: content-based recommendation, popularity-based recommendation and collaborative filtering methods. Content-based recommenders are developed based on the top fashion dataset abstracted by Amazon Ads API. Bag of Words and TF-IDF are applied on product title to find the products with most similar titles. In addition, the image of the product can be input for recommendation by object detection and calculation. The quality of content-based recommenders is measured by the real users participating in the evaluation of recommending output using metrics including Accuracy, Diversity, User Satisfaction and Novelty based on designed questionnaire. For accuracy, bag of words and TF-IDF perform better than image-based method. Bag of words outperforms the others on Diversity. TF-IDF achieves higher user satisfaction. Bag of words surprisingly collect feedbacks higher on novelty. Popularity-based recommendation and collaborative filtering systems are developed based on Amazon Fashion Review data, evaluating Mean Average Recall, Coverage and Novelty on the test set with random recommender together. Collaborative-filtering methods achieves higher mean average recall and novelty performance. Random recommender has the largest coverage over 90%, which followed by collaborative-filtering methods achieving above 40%. Undoubtedly, popularity-based recommender is the one with smallest coverage as it recommends the same popular items to all users.

## 1.	Introduction
    
Fashion is a big market! From P.Smith, the revenue of the global apparel market was calculated to amount to some 1.5 trillion US and was predicted to achieve about 2 trillion dollars by 2026. Ian Mackenzie highlighted that about 35 percent of what consumers purchase on Amazon and 75 percent of what they watch on Netflix come from product recommendations. In a data-driven business model, an accurate recommendation system can help companies boost sales. 

In this project, we build content-based (text, image), popularity-based, and collaborative filtering models. These models have different roles, of which the first two can be used for cold-start scenarios with relatively little data, while collaborative filtering, especially the item-item approach, is more applicable to companies with amounts of data accumulation.


### 1.1	 Recommender system
    
E-commerce recommender systems can provide product recommendations to their customers and suggest what they might like to buy based on their past histories of purchases, reviews, and/or product searches. There are three basic architectures for a recommender system introduced in our project:

 > A.	Content-based systems:

In the content-based system, our main goal is to make recommendations based on the user's search input, which can be either text or images. In the text part, we will organize the text into a vector based on the product title that the user is currently reading, and compare it with the existing title's vector to find similar results. On the other hand, in the image part, considering that the images input by users may be very complicated, we use object detection model to find out the location of clothes and other clothing accessories in the image, and crop the location of clothes, and then use the crop result as a search to find similar clothes.

The main idea of these algorithms is to recommend items that are similar to those that a customer rated highly in the past. User profiles and item profiles are created to capture unique characteristics used by the recommender system. We then use user-item profiles to predict the heuristics by computing the similarity scores between the user’s and item’s vectors.

 > B.	Popularity-based filtering systems:

It is another method to deal with the "cold start" problem. Popularity based recommendation system works with the trend. It basically uses the items which are in trend right now. For example, if any product which is usually bought by every new user, then there are chances that it may suggest that item to the user who just signed up. This approach eliminates the need for knowing other factors like user’s behavior, user preferences and other factors. Hence, the single-most factor considered is the rating to generate a scalable recommendation system. This increases the chances of user engagement as compared to when there was no recommendation system.

The problem with popularity-based recommendation system is that the personalization is not available with this method, namely even though you know the behavior of the user, you cannot recommend items accordingly.


 > C.	Collaborative filtering systems:

Recommending the new items to users based on the interest and preference of other similar users is basically collaborative-based filtering, shown as the image below. The key advantage of this approach is that it does not require user profiles or item profiles, which can be challenging to build. This overcomes the disadvantage of content-based filtering as it will use the user Interaction instead of content from the items used by the users.""")

st.image('1_3._introdiction.jpg')

st.write("""
Figure 1: Collaborative-filtering mechanism

The drawback of this approach is that it is suffers from what is known as the cold start problem. The model performances heavily depend on the “density” of user-item interaction. This means that when a new user or a new item comes into the system the model’s predictions can deteriorate substantially. There are two major methods in collaborative filtering: neighborhood methods and latent factor models:

 >> 	Neighborhood methods: These techniques perform recommendation in terms of user/user and item/item similarity. User similarity can be measured in terms of the items they purchased. Item similarity can be measure in terms of the users who make the purchase. Similarity measures such as cosine similarity, Jaccard similarity, and Pearson correlation can be used to gauge the similarity between users and items.
 
 >> 	Latent factor models: These methods try to explain user behavior by inferring the “factors” based upon user and item interactions. One way to implement this is by using matrix factorization which maps users and items into a K-dimension space. The resulting matrix can be estimated by the dot product of user vectors and item vectors shown below.


### 1.2	Question Formulation and Our work
We organize the works for recommender exploration and study by tackling the following questions:

 > How to build a recommendation system with different kinds of data?
 
 > How about the pre-processing steps for each recommender design?
 
 > How to evaluate the performance of recommenders?

### 1.3	Pipeline of Recommendation System
The pipeline of a recommendation system has the following five phases 
 > Pre-processing 
 
 > Model Training 
 
 > Hyper Parameter Optimization 
 
 > Post Processing 
 
 > Evaluation.




## 3.	Data Source
    
    
#### 3.1.	Amazon review dataset
    
We use Amazon review dataset (2018),which is an updated version of the Amazon review dataset released in 2014. This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). It can be downloaded from the following website: http://deepyeti.ucsd.edu/jianmo/amazon/index.html. 
For our team, we only use the sub-category AMAZON FASHION, which includes:
A.	Reviews:
In reviews, it include:
 > reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
 
 > asin - ID of the product, e.g. 0000013714 
 
 > reviewerName - name of the reviewer
 
 > vote - helpful votes of the review
 
 > reviewText - text of the review
 
 > overall - rating of the product
 
 > summary - summary of the review
 
 > reviewTime - time of the review (raw)
 
 > image - images that users post after they have received the product

B.	Metadata:
In the meta transaction metadata for each review shown on the review page. Such information includes: Product information, e.g. color (white or black), size (large or small), package type (hardcover or electronics), etc. Product images that are taken after the user received the product. 

#### 3.2	Amazon api datasets
    
Since the review data is obtained by ucsd directly in the amazon crawler, although the crawler data includes the user's review data, but the crawler data includes some books, coats, jewelry, glasses, backpacks and other accessories in addition to clothing data, and the lack of label data to label these contents, so in the absence of the user's past browsing history It is easy to confuse the classification and difficult to recommend the content. Therefore, instead of using the data from ucsd, the analysis was conducted using the data obtained from the api. The content of the data is similar to the above uscd, including asin, title, image hyperlink, but the related comment data is missing.

#### 3.3	Deep Fashion
    
We found that the deep fashion dataset includes these data, so we used the data to train the yolo model with a total of about 350,000 images.


### 3.4	Data Preparation
    
We selected metadata with 186,637 records and review with 883,636 records. We first merged the data, we used asin as the primary key to merge the two data sets, and the merged table with 875,121 records. 518,714 records were retained after removing duplicates. We keep the following columns: reviewerID, reviewTime, asin, title, brand,overall, imageURLHighRes for further exploration. 

The results of the pre-processing of the dataset can be observed from the table 1.

Number of missing values across columns :
""")




# 	Number of missing values across columns
# reviewerID	0
# reviewTime	0
# asin	0
# title	0
# brand	0
# overall	0
# imageURLHighRes	0

st.dataframe(df)

st.write("""
table 1: Results of Pre-Processing data.

To make the program more readable, we modified the original column names, changing 'overall' to 'rating', 'reviewerID' to 'user_id', 'asin' to 'product_id', and 'imageURLHighRes' to 'image'.

table 2 displays the total number of ratings , users and number of products in the dataset.
""")

st.dataframe(df2)


# 	Number of Data
# Total no of ratings 	518,714
# Total no of users 	465,432
# Total no of products	98,043
# Total no of review	518,714
st.write("""
By using the .info() and .dtypes we can get a information about index dtype,column dtype and usage of the memory. The results can be seen in the table 3.
""")

st.dataframe(df3)

st.write("""
Table 3:Concise summary of the dataframe


After reading the dataset and finding all the details, we move to the ratings column in the dataset. Since it uses the ratings from different users, we plot the overall ratings to see if they are well distributed as shown in Figure 4 .The plot shows that five star ratings are given the most by the users to different products and lowest number of users rated the products with two star ratings . To plot this graph we have used the seaborn library which is built on top of maltplotlib and is integrated with the pandas.
""")

st.image('./fig1.png')
# Figure 1: Total number of ratings
st.write("""
We further calculated the joint probability distribution of rating and rating_count:
 """)

# Figure 2: Joint distribution for number of ratings & rating
st.image('./fig2.png')
st.write("""
For API data, we first remove null value from the api data, and remove the duplicate title data, and finally keep the following five columns: asin, title, image url, brand, color, with a total data volume of about 17,000, and crawl the image url to get the corresponding 17,000 images.


## 4.	Model Analysis
To build a recommender engine for a E-commerce giant like Amazon, there are many practical factors to consider:
 > Generalization: With the diverse categories of items that are available for sale the recommender engine needs to be able to recommend items across different categories.
 
 > Cold start/sparsity problems: How does the models handle new users and new items that do not exist in the current system? Moreover, what can you do to improve the model performance when the user and/or item have limited feedback information?
 
 > Scalability: When new data is available, how do you retrain the model? 

### 4.1	Content-based Recommendation
    
We have two sections for content-based recommendations: text-based recommendations with database titles, and user-image based recommendations:



#### 1.	Searching with images can be divided into the following steps:
    
We consider that when we get a picture of a potential customer from social media, we want to find the right match for that customer from our product catalog. However, the background of the image on social media may be too complicated, so using the image directly for recommendation may affect the recommendation result due to the background. Therefore, we need to train a model to find the coordinates of the clothes and then filter the effect of the background on the recommendation system by cutting it, so that the recommendation system can focus on the clothes only.""")

st.image('./yolo.png')

st.write("""
Therefore, we trained a set of object detection model using the data in the deep fashion dataset, and the process is as follows: the data already includes the bounding box content, but some of the categories and bounding box range is wrong, for example, the position of the bounding box exceeds the length and width of the image, so we first use opencv and logic comparison to filter out the images that do not match the length and width, and we found that 36 categories are closer to the content of clothing or accessories. Therefore, we annotate the proportion and category of the bounding box, which is about 350,000 images in total, and train the model with yolo v5l to get a set of object detection model.

Using the object detection model helps us to know the position of the clothes and crop them. After the image is cut, we can know the position of the clothes in it, and we use VGG16 to find out the images that are similar to the images in our database and use these images as a basis for recommendation.""")

st.image('./deepimage.png')

st.write("""
#### 2.	Searching by text can be separated into the following steps:

Considering that after reading an item in the system, users may be interested in products similar to the title of that item, we tried three nlp models to calculate the similarity between user input and the products there.

Before introducing the nlp model, we would like to explain our process of processing the data. Among the top-fashion data we use, some of them are listed with multiple contents because of different colors or sizes. To calculate the similarity between the user input and the title in the top fashion data.
Our input title and image is:
""")

st.image('./fig7.png')



st.write("""
 > bag of words: Each word is computed as one-hot encoding, focusing only on the keyword itself and ignoring the context and syntax for comparison.""")
st.write('And top 3 of bag of words is:')
st.image('./wob1.png')
st.image('./wob2.png')
st.image('./wob3.png')

st.write("""
 > idf: If the fewer documents containing term, that is, the smaller the n, the larger the IDF, it means that term has good category differentiation ability.
 """)
st.write('And top 3 of idf is:')
st.image('./idf1.png')
st.image('./idf2.png')
st.image('./idf3.png')
         
         
st.write("""
 > tf-idf: add term frequency to idf for weighting, and use tf * idf to find the most representative titles""")
st.write('And top 3 of tf-idf is:')
st.image('./tf-idf1.png')
st.image('./tf-idf2.png')
st.image('./tf-idf3.png')

st.write("""
### 4.2	Popularity-based Recommendation

A common (and usually hard-to-beat) baseline approach is the Popularity model. This model is not actually personalized it simply recommends to a user the most popular anime that the user has not previously consumed.

We have packaged a separate module based on the popularity-base recommendation system. In this module, we build a dataframe to establish the rating and count for each product. And then sort dataframe by counts in descending order, extracting the top 10. 
For example, when we recommend a user's product, we recommend the Top 10 most popular items at the moment, regardless of whether the user has a difference. 
""")

st.dataframe(df3)
# Product_id	Rating	Counts	Rank
# B00YP2TNZ2	3.849057	636	1.0
# B000PHANNM	4.704787	376	2.0
# B0183JQHCO	3.267030	367	3.0
# B00XTM0ZPG	4.203647	329	4.0
# B00ZW3SCF0	3.917683	328	5.0
# B00UPN42RY	4.264984	317	6.0
# B011MP1ODS	4.442675	314	7.0
# B01CDV7TNE	4.694444	288	8.0
# B00LKWYX2I	4.367133	286	9.0
# B00RLSCLJM	4.838596	285	10.0

st.write("""
### 4.3	Collaborative Filtering
In the Collaborative Filtering method, we build a user-item matrix. The matrix is a typical sparse matrix because there is a large amount of missing data about the rating of the item by the user.


A item-item or user-user similarity matrix is constructed based on the user-item matrix. We use to method in calculating similarity: 

 > Pearson correlation — The most well-known similarity metric for the linear relation is person correlation. It measures how similar two samples are based on the direction of how the value changes.
 
 > Cosine similarity — As the name mentioned, It measures the cosine angle of the two vectors in the multi-dimensional space. Two things can be similar together in terms of direction rather than magnitude.

Instead of direct computation with the user-item interaction matrix. We will decompose the user-item interaction matrix into the latent factors matrix representing the lower-dimensional space that is more useful. The idea of decomposing is we believe that the observed user-item rating matrix is constructed from the underlying user and item latent factor matrix.

Suppose we can extract the best underlying latent factor matrix that minimizing the loss between the reconstructed matrix and the original matrix. Then we can use the inner product of the user and item latent factor matrix for inferencing an unobserved rating.
We use Matrix Factorization approach. There are several kinds of matrix factorization techniques, and each of them provides a different set of results, leading to different recommendations. This is the place where classic methods like Singular Value Decomposition, Principal Component Analysis. We mainly use SVD, SVDpp，and KNN. In this case, we use the Surprise package to help us control over the experiments and debug the different prediction algorithms. In Surprise package, there are various built-in and ready-to-use prediction algorithms such as baseline algorithms, neighborhood methods, matrix factorization-based ( SVD, PMF, SVD++, NMF), and many others. It also, various similarity measures (cosine, MSD, pearson) are built-in. 

### 4.4	Hyper Parameter Optimization
In collaborative filtering recommender, we use Surprise to analyse and compare the algorithms’ performance. Cross-validation procedures can be run very easily using powerful CV iterators (inspired by scikit-learn excellent tools), as well as exhaustive search over a set of parameters.
""")

st.dataframe(df4)

st.write("""
## 5.	We optimize the parameters by using Root Mean Square Error (RMSE). 

Finally we find SVDpp is the best model for collaborative filtering. Performance and Evaluation

### 5.1	Off-line evaluation metrics & result

The evaluation questionnaire consisted of fourteen questions. It was based on previous work in the movie recommender domain from Ekstrand et al. (2014), and was adapted to the fashion recommendation. Per question, users needed to select one recommendation list that would contain either the best (e.g., having the most attractive suggestions) or the worst recommendations (e.g., having the least appealing suggestions), in relation to different evaluation metrics. This setup allowed for asymmetrical user preferences, in the sense that the least chosen “best option” may not be the worst.

Different subsets of questions addressed different evaluations metrics. To address a user's evaluation of our algorithms, we measured the perceived Accuracy of a recommendation list, the perceived Diversity within a list, whether a user perceived that a list was personalized toward her preferences (i.e., Understands Me), the experienced level of Satisfaction, and the perceived level of Novelty. The list of questions was as follows, noting that some questions were formulated positively, while others were formulated negatively:

 > Accuracy: Q1. Which list has more selections that you find appealing? [positive]

 > Accuracy: Q2. Which list has more obviously bad suggestions for you? [negative]

 > Diversity: Q3. Which list has more products that are similar to each other? [negative]

 > Diversity: Q4. Which list has a more varied selection of products? [positive]

 > Diversity: Q5. Which list has products that match a wider variety of preferences? [positive]

 > Understands Me: Q6. Which list better reflects your preferences in products? [positive]

 > Understands Me: Q7. Which list seems more personalized to your university ratings? [positive]

 > Understands Me: Q8. Which list represents mainstream ratings instead of your own? [negative]

 > Satisfaction: Q9. Which list would better help you find products to consider? [positive]

 > Satisfaction: Q10. Which list would you likely to recommend to your friends? [positive]

 > Novelty: Q11. Which list has more products you did not expect? [positive]

 > Novelty: Q12. Which list has more products that are familiar to you? [negative]

 > Novelty: Q13. Which list has more pleasantly surprising products? [positive]

 > Novelty: Q14. Which list provides fewer new suggestions? [negative].

We compared how users evaluated different university recommendation lists, which were generated by different algorithms. We outlines per question the percentage of instances in a which recommendation list was chosen, designated by the algorithm generating it. Some questions contributed positively to a specific metric (e.g., Q1 to Accuracy), while those denoted in italics contributed negatively to that metric (e.g., Q2).""")

st.dataframe(df6)


# 	% of chosen
# 	Popularity	Collaborative Filtering
# Q1	32%	68%
# Q2	24%	76%
# Q3	35%	65%
# Q4	27%	73%
# Q5	19%	81%
# Q6	24%	76%
# Q7	13%	87%
# Q8	23%	77%
# Q9	36%	64%
# Q10	37%	63%
# Q11	28%	72%
# Q12	23%	77%
# Q13	18%	82%
# Q14	19%	81%

st.write("""
To examine which algorithm had the best performance per metric, we performed pairwise t-tests per questionnaire item. And calculate the t-statistics, while the p-values are indicated by asterisks in superscript. The tests were performed by creating dummy variables for each algorithm, assigning the value 1 to an algorithm if its recommendation list was chosen by a user for a specific item. According to the p value, we found the recommender algorithms are evaluated differently across different metrics. And from the % of choosen, we found collaborative filtering is much better than popularity.


### 5.2	The other evaluation methods & result


The Long Tail plot is used to explore popularity patterns in user-item interaction data. Typically, a small number of items will make up most of the volume of interactions and this is referred to as the "head". The "long tail" typically consists of most products, but make up a small percent of interaction volume.
""")

st.image('./fig3.png')

st.write("""

We made MAR@K available in recmetrics, and MAR@K gives insight into how well the recommender is able to recall all the items the user has rated positively in the test set.
""")

st.image('./fig33.png')

st.write(""" 
From the figure, we can find that by MAR@K, the collaborative filter is able to recall the relevant items for the user better than the other models.

Besides MAR@K, we also explored the Coverage metrics, which is the percent of items in the training data the model is able to recommend on a test set. In this example, the popularity recommender has less than 1% coverage. The random recommender has nearly 100% coverage as expected. Surprisingly, the collaborative filter is able to recommend 42% of the items it was trained on.""")

st.image('./fig4.png')

st.write("""
Finally, we also use the following formula to measure novelty.""")

st.image('./form1.png')

st.write("""
We found that """)

st.dataframe(df7)

# 	Score
# Random	9.217172513383314
# Popularity-based	4.3923459172729755
# Collaborative Filtering	11.17091035112718

st.write("""
In novelty, Collaborative Filtering is also best in all recommender system.

## 6.	Conclusion
The primary goal of this project is to provide recommendations to the user in a e-commerce website by making use of machine learning algorithms. We have designed and implemented the system using collaborative filtering and Pearson correlation coefficient. The dataset considered has the ratings given by the other users to a specific product and depending on the similarity between the rated product we try to recommend the products to our current user. Through a comprehensive comparison, we conclude that collaborative filtering is the best recommended effect, and among collaborative filtering, SVDpp algorithm is the optimal one.

The future work of the project includes improving the efficiency of the system. And it should also be able to give appropriate recommendations to the users who don’t have any previous purchase history or to the new users. In future we can try to use recurrent neural networks and deep learning. With the help of deep learning techniques we can overcome some of the drawbacks of the matrix factorization technique. Deep learning uses recurrent neural networks to accommodate time in the recommender system which is not possible in the matrix factorization method. We can also work on providing sub-optimal recommendations to the user and record the reaction of the user and it can be used in the future by the system.

## 7. State of work


> Qi Zhang: Chief Marketing Officer

> Chihshen: Chief Technology Officer

> Zhipeng: Chief Product Officer


The whole project is really a team effort. At the beginning, all of us were involved in data collection and exploration to choose the most suitable fashion data source. Qi Zhang arranged some meetings with experts in the fashion domain to understand more about the topic. Both Zhipeng and Qi Zhang are participated in collaborative-filtering recommender study and design. Chihshen is mainly responsible for image-based model training, image crawler, image search system and recommender design, and streamlit report format preparation. Zhipeng is responsible for driving the project discussion and monitoring the project progress, aligning the internal meeting time and the meeting with our project instructor. Zhipeng mainly focuses on data manipulation, visualiztion, recommender design and evaluation. Along the project progress, all of us work close to each other in brainstorming, discussion, and report writing.


Reference
    https://medium.com/@cfpinela/recommender-systems-user-based-and-item-based-collaborative-filtering-5d5f375a127f
    https://www.kaggle.com/datasets/ajaysh/women-apparel-recommendation-engine-amazoncom?sort=recent-comments
    https://github.com/ultralytics/yolov5
    
    https://github.com/Abhinav1004/Apparel-Recommendation
    
    .
""")
