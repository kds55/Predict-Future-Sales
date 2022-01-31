Predict-Future-Sales
-------------------------


Contents:
- Repository Summary
- Competition/Goal 
- EDA
- Features Added
- Final Model
- Other Features Tried
- Things to Consider

Repository Summary
-------------------------
- See "EDA & Model.py" for EDA and modeling done in python
- See "Data Visulization - Client Dashboard - PowerBi.pbix" for a example PowerBi dashboard I create that would provide valuable reporting to the client

Competition/Goal 
-------------------------
- Goal: To predict the number of units sold of certain item ID's per shop for 1 month after our data (date block 34 or Nov 2015)
- For general description of data and the competition see: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview

EDA
-------------------------
- item_categories data - Per review of data it appeared that within the item category name it contained two different parts. One being a more generic descriptor and the other being more specific (Ex. Video Games - Xbox). This should be broken out. Additionally it appeared as though certain item categories appeared to be similar in nature, however as I already broke out the item categories into their two parts as described above, I elected to just delete the original item category as I believe it was too generic and would add noise to the model.
- items data - Per review of the data there appears to be additional information stored in the item names. Some of the items have information kept in rounded or square brackets at the end. This information appears to be generic (Ex. dvd). This information should also be broken out and shown separate as it will further allow us to classify these items into more specific categories then before.
- shops data - Per review of the data (shop names) with Google translate there appears to be a few things that the name consists of. 1) the city 2) the type of store 3) the name of store location. Only points 1 and 2 appear important, so this information should also be broken out. Additionally it appears that there are a few stores that have very similar names with small differences. These look to just be typo's, so these stores are corrected.
- sales_train data - Per review of the data there appears to be a variety of outliers. There appears to be individual days where there is a large quantity of items sold that is far outside the mean as well as items that sell for a price far from the mean. These transactions are eliminated from the training data to avoid any noise. There appears to be more outliers to deal with as there is certain months or days where an item sells a significant amount more then the norm (Ex. the launch of the PS4 had a large amount of sales recorded in Nov 2013 when it was released). There was also items where the min items sold appeared low or there was 1 item returned for a large price. Neither of these potential outliers were accounted for however definitely something to consider.
- test - Per review of the test data (the items we have to predict with our model for date_block_num 34) there is 363 items that we have to predict for that we have no transaction history in the sales_train. As such we will need to figure out how to adjust for these.

Features Added
-------------------------
The below features were included in our final model:
- df - created a base dataframe consisting of all possible item, shop, 
- item category and subcategory - as broken out from the original category names (see above in the EDA section)
- item feature 1 and item feature 2 - as broken out from the items data (see above in the EDA section)
- shop ID, city, and shop type - as broken out from the original shop names (see above in the EDA section)
- month and year - added in the month and year that each transaction occured as this was not originally included in our data. Should help identify trends and seasonality
- historical item price - adding in the historical average item price. For items in the test data that were not included in our sales train we calculated the average for them based off of 1) the average for the category, subcategory, item feature 1 and item feature 2. 2) If 1 yielded no results used the average for there category and subcategory. After reviewing the data this wasn't to accurate but did get it in the right ball park.
- average shop sales per month of year (Ex Jan, Feb...) - This would appeared to be a fairly accurate prediction for the year over year sales by each store as the data appeared fairly stationary. This is using the $ sales and not units sold, as $ sales appeared more stationary and also produced a better outcome in the model.
- average units sold per shop for history - used to help get an idea of the size of the store
- average units sold per shop per category for history - used to help determine sales mix of the store. Which item categories are most popular at each store
- average units sold per category for history - used to determine on an overall level which item categories are most popular
- average units sold per item for history - used to determine on an overall level which items are most popular

The below lag features were added. The numbers in brackets represent the lags used - these lags were determined to be the most beneficial to the model without adding any unnecessary noise:
- item id sold per shop per date_block_num (1,2,3,6,9,12)- this is what we are actually predicting
- units sold per item category per date_block_num (1,3,12)
- units sold per shop per date_block_num (1,3,12)
- units sold per item id per date_block_num (1,3,12)

Final Model
-------------------------
A XGBM model was used as it will be able to handle the multiple inputs well. I tried my best to stop the model from overfitting but it appeared to be something I still need to tweak as my training data was much stronger fit then my test data. See below for results:
- Training data (prior to date block 33) - 0.64698 RMSE
- Test data (date block 33) - 0.81820 RMSE
- Final prediction to actual (date block 34)- 1.00537 RMSE (5,587/12,589 or top 44.38% at the time of this being posted)

As you can see I was able to achieve a significantly lower RMSE on the training data showing my model was likely over fitted. The only thing odd is that the model did not take long to run as it only took 88 iterations to finish which seemed fairly quick given the amount of data. It did however start getting small performance gains on the test data after the first 16-20 iterations, which at this point the training data RMSE was around 0.75. I beleive my model can acheive a much better result with some tweaking. 

Other Features Tried
-------------------------
Below summarizes other features that were added to the model but removed due to either making the model worse or just slowing it down/adding noise for little performance gain:
- average units sold per shop per month (Jan, Feb...) 
- sales mix for overall company based on item categories
- sales mix per store by item categories
- average items sold per item ID per store
- lag - items sold per shop (specific item ID)
- lag - units sold per item subcategory
- lag - shops sales

Things to Consider
-------------------------
Below summarizes other things I should consider that could improve the model:
- Fine tuning the model parameters to avoid overfitting 
- Add delta (change) for lag features. This might help identify trends more easily for the models by showing the trend from lag 3 to lag 1 was increasing/decreasing.
- Add rolling averages. Again this might help make it identify trend easier as it will show trend better then just 1 historical average
- Add back the original item category ID. Not sure if this will help but something to consider.
- Clean the training data for more of the outliers as identified above in the EDA section. 
