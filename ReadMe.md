# A Multi-Modal Deep Learning Based Approach for House Price Prediction

Accurate prediction of house price, a vital aspect of the residential real estate sector, is of substantial interest for a wide range of stakeholders. However, predicting house prices is a complex task due to the significant variability influenced by factors such as house features, location, neighborhood, and many others. Despite numerous attempts utilizing a wide array of algorithms, including recent deep learning techniques, to predict house prices accurately, existing approaches have fallen short of considering a wide range of factors such as textual and visual features. This paper addresses this gap by comprehensively incorporating attributes, such as features, textual descriptions, geo-spatial neighborhood, and house images, typically showcased in real estate listings in a house price prediction system. Specifically, we propose a multi-modal deep learning approach that leverages different types of data to learn more accurate representation of the house. In particular, we learn a joint embedding of raw house attributes, geo-spatial neighborhood, and most importantly from textual description and images representing the house; and finally use a downstream regression model to predict the house price from this jointly learned embedding vector. Our experimental results with a real-world dataset show that the text embedding of the house advertisement description and image embedding of the house pictures in addition to raw attributes and geo-spatial embedding, can significantly improve the house price prediction accuracy.



## Dataset:

Our experimentation was conducted using a dataset comprising records of house transactions sourced from a prominent real estate [website](https://www.realestate.com.au/). The dataset encompasses real estate transactions in Melbourne, Australia's second-largest city in terms of population. It encompasses a total of 52,851 house transaction records from year 2013 to 2015. Furthermore, the dataset includes valuable details about nearby Points of Interest (POIs), encompassing regions, schools and train stations. This dataset comprehensively covers information related to 13,340 regions, 709 schools and 218 train stations. Additionally, each record within the dataset includes a concise textual description and images of the houses.
<!-- We conducted our experiments on the house transaction records obtained from a real-estate website2 for Melbourne, which is the second largest city in Australia by population.We extracted a total of the 52,851 house transaction records of years from 2013 to 2015. Our dataset also includes the three types of POIs: regions, schools, and train stations and their corresponding features. Houses are situated in regions which capture the geographical contextual information about houses. Intuitively, information about nearby schools and train stations may influence house prices. Our dataset contains information of the 13,340 regions, 709 schools, and 218 train stations. -->

![dataset](https://github.com/4P0N/mhpp/assets/70822909/3d6208d7-0265-49e1-90f6-3a4f5987b266)


**House and POI Features**: Our dataset encompasses an extensive array of house attributes for each property, totaling 43 distinct features. These features span a spectrum from fundamental characteristics such as the number of bedrooms and bathrooms to intricate facility-related attributes like the presence of air-conditioning and tennis courts. Moreover, the dataset offers comprehensive insights into Melbourne's various regions at the SA1 level, incorporating details such as population count, average age, median personal income, and educational qualifications.

Furthermore, the dataset is enriched with information regarding different Points of Interest (POIs), including schools and train stations. These POIs are further categorized and accompanied by timetable data and precise location information.

**House Descriptions**: The dataset also provides textual descriptions for each of the houses, capturing various aesthetics and features that may not be readily quantified. These descriptions exhibit varying lengths, with some extending up to a maximum of 280 words. An illustrative example of a textual description from the dataset is shown in Figure \ref{fig:data}(d).

**House Images**: Each property in the dataset typically features an average of five distinct images. These images collectively portray both the interior and exterior aspects of the houses. It is  worth noting that while some houses may have been missing one image within this five-image set, we mitigated this by duplicating one of the four available images to maintain consistency. In our setting, we used the collage of five distinctive images to learn the correlation between house images and descriptions.




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
