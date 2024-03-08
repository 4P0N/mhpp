# A Multi-Modal Deep Learning Based Approach for House Price Prediction



Accurate prediction of house price, a vital aspect of the residential real estate sector, is of substantial interest for a wide range of stakeholders. However, predicting house prices is a complex task due to the significant variability influenced by factors such as house features, location, neighborhood, and many others. Despite numerous attempts utilizing a wide array of algorithms, including recent deep learning techniques, to predict house prices accurately, existing approaches have fallen short of considering a wide range of factors such as textual and visual features. This paper addresses this gap by comprehensively incorporating attributes, such as features, textual descriptions, geo-spatial neighborhood, and house images, typically showcased in real estate listings in a house price prediction system. Specifically, we propose a multi-modal deep learning approach that leverages different types of data to learn more accurate representation of the house. In particular, we learn a joint embedding of raw house attributes, geo-spatial neighborhood, and most importantly from textual description and images representing the house; and finally use a downstream regression model to predict the house price from this jointly learned embedding vector. Our experimental results with a real-world dataset show that the text embedding of the house advertisement description and image embedding of the house pictures in addition to raw attributes and geo-spatial embedding, can significantly improve the house price prediction accuracy.
## Dataset:

We conducted our experiments on the house transaction records obtained from a real-estate website2 for Melbourne, which is the second largest city in Australia by population.We extracted a total of the 52,851 house transaction records of years from 2013 to 2015. Our dataset also includes the three types of POIs: regions, schools, and train stations and their corresponding features. Houses are situated in regions which capture the geographical contextual information about houses. Intuitively, information about nearby schools and train stations may influence house prices. Our dataset contains information of the 13,340 regions, 709 schools, and 218 train stations.

![dataset](https://github.com/4P0N/mhpp/assets/70822909/3d6208d7-0265-49e1-90f6-3a4f5987b266)


**Housing Features**: Our dataset contains information about a wide range of housing features. In
total, we consider 43 housing features for each house for in depth exploration of the effect of GSNE. To the best of our knowledge, none of the prior works considered such a wide range of feature sets in a large dataset like ours for house price prediction task. Although the dataset in Kaggle competition has 86 features, it has only 3000 samples in total and lots of columns are highly sparse rendering only a few of those columns truly useful. Besides, no information regarding neighbourhood amenities is available in that dataset. In our dataset, each house record contains information ranging from basic housing features like number of bedrooms, number of bathrooms, number of parking spaces, location, type of property, etc. to detailed facility features like air-conditioning, balcony, city-view,river-view, swimming, tennis-court, etc.

**Region Features**: Our dataset contains Melbourne region information at SA1 level3. SA1 is the
most granular unit for the release of census data of Australia. The SA1 data typically has a
population of 200 to 800 people with an average of 400 people per region. For each region, our
dataset contains comprehensive information about the number of residents, average age, median personal income, percentage of Australian citizens, educational qualification, median house rent,location as the centroid of the region, etc. Since these aspects can be useful for determining house prices, we consider all of them as the features for regions.

**School Features**: The schools in our dataset are attributed with the type of school (primary
or secondary), school category by gender(single gender or co-ed), ranking, location, number of
students, zone restrictions, number of students enrolled in Victorian Certificate of Education(VCE), percentage of students securing 40% marks, etc


**Train Stations**: The train stations in the dataset contain information about their location and
average time to reach to other stations.


<!-- You can get the experiment dataset [here](https://drive.google.com/drive/folders/1ssAjH8b8GGVlYohIdhyZPKje2sGeXggB?usp=sharing).

## Project Structure:

1. Repository
   - GSNE_Boosting_House_Price
     - Dataset
       - Check_Performance_Dataset
         -	*** contains check performance required datasets  ***
       - Preprocessed_Dataset
         -	*** Main raw data ***
       - Processed_Dataset
         -	*** Processed npz files ***
     - Data_Preprocessing
       - 	*** data-preprocessing codes ***
     - Graph_embedding
       - Code
         - utils.py
       	 - model.py
         - train.py
       - job.sh
       - requirements.txt
     - Checking_performance 
       - embedding_1
         - 	*** contains embedded pickle file ***
       - embedding_2
	 - 	*** contains embedded pickle file ***
       - check_performance.py
   - ReadMe.md


### To run the the project,you have the followings to do-
1.	Check the requirements in the requirements.txt in the Graph_embedding folder 3 and install them, Then just do `python train.py ‘cora-ml’ ‘glace’ `
or simply run the command `./job.sh` in linux system.
2.	Now for the price prediction from the embedded pickle files in folder 4.Chechking_performance, run `python chechk_performance.py`



 -->
