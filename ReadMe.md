# Boosting House Price Predictions using Multi-Modal House Price Predictor



Real estate business is the profession of buying, selling, or renting real estate(land, buildings, or housing).1 In this world of depereciated assets, real estate business provides the foresight of a long lasting savings in current the world economy. Thus it is a major concern for not only the clientele like buyers or tenants but also for sellers ranging from different different stakeholders and companies. So far, a lot of techniques leveraging different sets of algorithms
including machine learning has been applied considering different sets of attributes to predict the house price as precise as possible. However, all of them
have failed to take into account all the possible attributes ranging from their raw features, geo-spatial properties, textual description to even their visual representations. In real world, all of them have significant contributions in correctly determining determining house price as well as interest of the house
buyers. In this paper, we take into account all these attributes of different domains used in advertising a house fro a real life house flyer. We proposed a
Multi-Modal House Price Predictor (MHPP) that captures embedding form geo-spatial context considering different point of interrests (POIs), textual
embedding from the description of the house present and image embedding form different interior and exterior visuals of the house. Our extensive experimentation shows that embedding captured in our Multi-Modal House Price Predictor from all these possible attributes for a particular house significantly
improves the house price prediction task , irrespective of the choice of downstream regression model.
## Dataset:

We conducted our experiments on the house transaction records obtained from a real-estate website2 for Melbourne, which is the second largest city in Australia by population.We extracted a total of the 52,851 house transaction records of years from 2013 to 2015. Our dataset also includes the three types of POIs: regions, schools, and train stations and their corresponding features. Houses are situated in regions which capture the geographical contextual information about houses. Intuitively, information about nearby schools and train stations may influence house prices. Our dataset contains information of the 13,340 regions, 709 schools, and 218 train stations.

![dataset](https://github.com/4P0N/mhpp/assets/70822909/fcdb9e76-ccfb-44d8-afe9-53fe31408eef)


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


You can get the experiment dataset [here](https://drive.google.com/drive/folders/1ssAjH8b8GGVlYohIdhyZPKje2sGeXggB?usp=sharing).



