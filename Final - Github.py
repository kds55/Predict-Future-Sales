import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
from itertools import product

item_categories = pd.read_csv ("item_categories.csv")
items = pd.read_csv ('items.csv')
sales_train = pd.read_csv ('sales_train.csv')
shops = pd.read_csv ('shops.csv')
test = pd.read_csv ('test.csv')

#####################################################
#EDA & Some pre-processeing over the DF's
#####################################################

#--------------------------------
#item_categories
#--------------------------------

#adding in id category to sales train
sales_train = pd.merge(sales_train, items, on = 'item_id', how = "left")

item_categories
#we can see that the item category name is broken into two parts by the '-':
# - First part apears to be a more general category
# - Second part appears to be more specific category
#These should be broken out and the original column dropped

#There appears to be some with teh same name but include another word in ( )'s. 


def compare(x,y,filt):
    group = sales_train.groupby(['date_block_num', filt]).sum('item_cnt_day').reset_index()
    group = group[['date_block_num', filt,'item_cnt_day']]
    group_filtered_x = group.loc[(group[filt] == x)]
    group_filtered_y = group.loc[(group[filt] == y)]
    plt.plot(group_filtered_x['date_block_num'], group_filtered_x['item_cnt_day'])
    plt.plot(group_filtered_y['date_block_num'], group_filtered_y['item_cnt_day'])
    
    group_filtered_x = sales_train.loc[sales_train[filt] == x]
    group_filtered_y = sales_train.loc[sales_train[filt] == y]
    mean_x = group_filtered_x['item_price'].mean()
    mean_y = group_filtered_y['item_price'].mean()
    min_x = group_filtered_x['item_price'].min()
    min_y = group_filtered_y['item_price'].min()
    max_x = group_filtered_x['item_price'].max()
    max_y = group_filtered_y['item_price'].max()
    
    print('Mean, min, max for', x, '=', mean_x, min_x, max_x )
    print('Mean, min, max for', y, '=', mean_y, min_y, max_y )
    
compare(33, 34, 'item_category_id') #appear to be similar - atleast price wise. Per Google transalte it is for payment cards live however one is numeral
compare(43,44, 'item_category_id') #do not appear to similar in either way. Per google translate they are for audio books however one is for figure
compare(64,65, 'item_category_id') #appear to be the same type of items - similar trend however price wise differs. Per google translate one appears to be compact (travel games) where as the other is normal board games
compare(69,70, 'item_category_id') #appear to be the same type of items - similar trend however price wise differs. Per google transalte both are souvenirs one is weighed in. Not sure what the difference is.
compare(75,76, 'item_category_id') #appear to be similar - atleast price wise, somewhat similar trend. Per google translate it is home & office however one is digital. 
compare(77,78, 'item_category_id') #appear to be similar - atleast price wise, somewhat similar trend. Per google transalte it is educational however one is modified with figure
compare(81,82, 'item_category_id') #do not appear to similar in either way. Per google translate it is nat carriers for spire or piece. Not suire difference.
#all are some what modifications of one another however a few pairings seem more simliar 33/34, 64/65, 69/70, 75/76, 77/78

#checking below to see if these are even in the items test set
temp = pd.merge(test, items, on = 'item_id', how = "left")
temp.isnull().sum() #ensuring no null values

for i in [33,34,43,44,64,65,69,70,75,76, 77,78,81,82]:
    group = temp.loc[temp['item_category_id'] == i]
    print("Number of items in category", i, "=", len(group.index))
#81 and 82 are not in the test set so can be ignored

#conclusion:
#1 - breakout the category names into the two parts - more general vs specific
#2 - should set the following categories equal to one another. 33=34, 64=65, 69=70, 75=76, 77=78 and explore if better results. Also consider 43=44

item_categories['details'] = item_categories.item_category_name.str.split('-',1)
item_categories['category'] =''
item_categories['subcategory'] =''
for i in range(0,len(item_categories)):
    item_categories['category'][i] = item_categories['details'][i][0]
    if len(item_categories['details'][i]) > 1:
        item_categories['subcategory'][i] = item_categories['details'][i][1]
    else:
        item_categories['subcategory'][i] = item_categories['details'][i][0]      

item_categories = item_categories[['item_category_id', 'category', 'subcategory']]
item_categories['category'] = LabelEncoder().fit_transform( item_categories.category )
item_categories['subcategory'] = LabelEncoder().fit_transform( item_categories.subcategory )



#--------------------------------
#items
#--------------------------------
#we can immediately note that some of the items have round or square brackets after the name, indicating that there may be additional informaiton here
#nothing else to note

items['temp'], items['feature_1'] =  items.item_name.str.split('(',1).str
items['temp'], items['feature_2'] =  items.item_name.str.split('[',1).str

items['feature_1'], items['temp '] = items.feature_1.str.split(')',1).str
items['feature_2'], items['temp '] = items.feature_2.str.split(']',1).str

items = items.fillna('0')

items.feature_1 = LabelEncoder().fit_transform(items.feature_1)
items.feature_2 = LabelEncoder().fit_transform(items.feature_2)

items = items[['item_id','item_category_id', 'feature_1', 'feature_2']]


#--------------------------------
#shops
#--------------------------------
#shop name apperas to be made up of 3 different parts. seperated by spaces. Per Google translate they appear to be
#1 - city
#2 - type of store/identifier?
#3 - name of store/mall/location
#only 1 & 2 appear important. These are important as there appears to be multiple stores in same cities as well as multiple stores sharing the same type/identifier

shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"' #extra space will screw up out code below for taking out store type
shops['details'] = shops.shop_name.str.split(' ')
shops['city'] =''
shops['type'] =''
for i in range(0,len(shops)):
    shops['city'][i] = shops['details'][i][0]
    shops['type'][i] = shops['details'][i][1]
shops.loc[shops.city == "!Якутск", "city"] = "Якутск" #cleaning data and getting rid of exclamation point.

#As there is a wide variety of types, only kept types where there was 5 or more occurences
type = []
for cat in shops.type.unique():
    if len(shops[shops.type == cat]) >= 5:
        type.append(cat)    
shops.type = shops.type.apply( lambda x: x if (x in type) else "other" )

#label encoding
shops['shop_type'] = LabelEncoder().fit_transform( shops.type )
shops['shop_city'] = LabelEncoder().fit_transform( shops.city )

shops = shops[['shop_id', 'shop_type', 'shop_city']]

#below shops appear to be duplicates/same shop just named different. adjusted them to actual
sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

#--------------------------------
#sales_train
#--------------------------------
#adding in total sale amount
sales_train['sale_total'] = sales_train['item_price'] * sales_train['item_cnt_day']


#looking at individual sales
sales_train['item_cnt_day'].describe() #appears to be outliers, max value well exceeds average. 
sales_train['sale_total'].describe() #appears to be outliers, max value well exceeds average
sales_train['item_price'].describe() #appears to be outliers, max value well exceeds average

plt.plot(sales_train['item_cnt_day']) #appears to be outliers, max value well exceeds average. over 1,500 
plt.plot(sales_train['sale_total']) #outliers dont appear as apparent here
plt.plot(sales_train['item_price']) #appears to be outliers, max value well exceeds average. over 100,000

sales_train['item_cnt_day'].argmax() #row 2909818 is where the max value exists
sales_train['sale_total'].argmax() #row 1107225 is where the max value exists
sales_train['item_price'].argmax() #row 1163158 is where the max value exists

sales_train_temp = sales_train[sales_train['item_cnt_day'] > 1500] #appears to be one occurence. safe to say its an outlier
sales_train_temp = sales_train[sales_train['item_price'] > 100000] #appears to be one occurence. safe to say its an outlier


#looking at monthly groupings and nothing apperas unusual
sales_train_temp = sales_train.groupby('date_block_num').sum('item_cnt_day')
plt.plot(sales_train_temp['item_cnt_day'])
plt.plot(sales_train_temp['sale_total'])
plt.xticks(np.arange(0,33,1.0),  rotation = 'vertical', size =8)

    
#looking at daily groupings
sales_train_temp = sales_train.groupby('date').sum('item_cnt_day')

sales_train_temp['item_cnt_day'].describe() 
plt.plot(sales_train_temp['item_cnt_day']) #looks like a few days (5) that are outliers with over 10k sold in one day

sales_train_temp['sale_total'].describe() 
plt.plot(sales_train_temp['sale_total']) #looks like one or two outliers with over 1.7e7 for the day

sales_train_temp = sales_train.groupby(['date', 'date_block_num', 'item_id']).sum('item_cnt_day') #.item_cnt_day.sum() #sum('item_cnt_day')
sales_train_temp = sales_train_temp.sort_values(by=['item_cnt_day'], ascending = False)

sales_train_temp = sales_train_temp[sales_train_temp['item_cnt_day'] >10000]
sales_train_temp = sales_train_temp[sales_train_temp['sale_total'] > 1.7e7]


#removing outliers
sales_train = sales_train[sales_train.item_cnt_day < 1500]
sales_train = sales_train[sales_train.item_price < 100000]

#--------------------------------
#test
#--------------------------------

#there are no  new shops in the test df that aren't in our sales_train
list(set(test['shop_id'].unique()).difference(sales_train['shop_id'].unique()))

#there is 363 items that appear in the test df that aren't in our sales_train. We will have to figure out how to address these
#all these items do exist in the items DF so we at least know categories 
len(list(set(test['item_id'].unique()).difference(sales_train['item_id'].unique())))
list(set(test['item_id'].unique()).difference(items['item_id'].unique()))

max(list(set(test['item_id'].unique()).difference(sales_train['item_id'].unique())))


#####################################################
#Additional data prep
#####################################################
 
#--------------------------------
#creating base DF
#--------------------------------

#created a new df will all possible shop/item/month combinatoins in the sales train
#this is done as it will now include items where 0 were sold in that month/shop - the normal sales_train wouldnt show that
df = []
cols  = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
    temp = sales_train[sales_train.date_block_num == i]
    df.append( np.array(list( product( [i], temp.shop_id.unique(), temp.item_id.unique() ) ), dtype = np.int16) )
df = pd.DataFrame( np.vstack(df), columns = cols )


#adding in item_cnt_month into the df
sales_train_temp = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':['sum']})
sales_train_temp.columns = ["item_cnt_month"]
sales_train_temp = sales_train_temp.reset_index()
df = pd.merge(df, sales_train_temp, how = 'left')
df["item_cnt_month"] = df["item_cnt_month"].fillna(0)
df.item_cnt_month = df['item_cnt_month'].astype(np.int16)


#adding in data for the test frame/ones we are predicting
test['date_block_num'] = 34
df = pd.concat([df, test.drop(["ID"],axis = 1)], ignore_index=True, keys=cols)
df.fillna( 0, inplace = True )


#--------------------------------
#adding in features
#--------------------------------

#adding in item category/subcategory & item feature
temp = pd.merge(items, item_categories, how = 'left')
temp = temp[['item_id', 'category', 'subcategory', 'feature_1', 'feature_2']]
df = pd.merge(df, temp, how = 'left')
df.category = df['category'].astype(np.int16)
df.subcategory = df['subcategory'].astype(np.int16)


#adding in city and shop type
df = pd.merge(df, shops, how = 'left')
df.shop_type = df['shop_type'].astype(np.int16)
df.shop_city = df['shop_city'].astype(np.int16)


#adding in month and year
df['month'] = df['date_block_num'] % 12 +1
temp = [list(range(0,35)),[2013]*12 + [2014]*12+  [2015]*11]
temp = pd.DataFrame(temp).transpose()
temp.columns = ['date_block_num', 'year']
df = pd.merge(df, temp, how = 'left')
df.year = df['year'].astype(np.int16)


#adding in historical item price
#as there was some items in the test set that were not sold in the year, created an average for them by:
#1- taking a historical average for the category, subcategory, item feature 1, and item feature 2
#2- if #1 doesn't yield a result take a historical average for category and subcategory
temp = sales_train.groupby( ['item_id'] ).agg({'item_price': ['mean']})
temp.columns = ['avg_item_price']
temp = temp.reset_index()
df = pd.merge(df, temp, how = 'left')
df.avg_item_price = df['avg_item_price'].astype(np.float32)

temp = df[df['date_block_num']!= 34]
temp = temp.groupby(['category', 'subcategory', 'feature_1', 'feature_2']).agg({'avg_item_price': ['mean']})
temp.columns = ['avg_item_price_2']
temp = temp.reset_index()
df['isnull'] = df['avg_item_price'].isnull()
df = pd.merge(df, temp, how = 'left')
for i in range(0, len(df)):
    if df['isnull'][i] == True:
        df['avg_item_price'][i] = df['avg_item_price_2'][i]       
df.drop(['isnull', 'avg_item_price_2'], axis = 1, inplace = True)
df.isna().sum() #still some null values so simpliefied to be just avg price for category and subcategory

df['isnull'] = df['avg_item_price'].isnull()
temp = df[df['date_block_num']!= 34]
temp = temp.groupby(['category', 'subcategory']).agg({'avg_item_price': ['mean']})
temp.columns = ['avg_item_price_2']
temp = temp.reset_index()
df = pd.merge(df, temp, how = 'left')
for i in range(0, len(df)):
    if df['isnull'][i] == True:
        df['avg_item_price'][i] = df['avg_item_price_2'][i]
df.drop(['isnull', 'avg_item_price_2'], axis = 1, inplace = True)
df.isna().sum() #no null values exist now


#adding in average shop sales per month
#have to do some additional work for null values for shop 36 as only 1 month of sale
#used historical average for shop of the same type to create these averages
temp = sales_train.groupby( ['shop_id', 'date_block_num'] ).agg({'sale_total': ['sum']})
temp.columns = ['sale_total']
temp = temp.reset_index()
temp['month'] = temp['date_block_num'] % 12 +1
temp = temp.groupby(['shop_id', 'month']).agg({'sale_total':['mean']})
temp.columns = ['avg_shop_sales_per_month']
temp = temp.reset_index()
df = pd.merge(df, temp, how = 'left')

df.isna().sum() #5100 occurences of 0 sales per month
df['isnull'] = df['avg_shop_sales_per_month'].isnull() #looks like it is for shop 36 - it only opened last month of year so no sales historically for month 11 (aka for block 34)
temp = df[df['shop_type']==3] #use shop type 3 as the indicator for it
temp = temp[temp['month']==11] #use month 11 as thats the month thats missing
temp.dropna(inplace = True) #drops for block 34 thats still included
temp = temp['avg_shop_sales_per_month'].unique() #gets unique sales - that way it is taking a true average of all of them
temp = temp.mean()
df.fillna(temp, inplace = True)


#adding in average shop units sold
temp = df[['date_block_num', 'shop_id', 'item_cnt_month']]
temp = temp.groupby(['shop_id']).agg({'item_cnt_month': ['mean']})
temp.columns = ['avg_shop_cnt']
temp = temp.reset_index()
df = pd.merge(df, temp, how = 'left')


#adding in hist stores units per category ID - mean
temp = df[['date_block_num', 'shop_id', 'category', 'item_cnt_month']]
temp = temp.groupby(['shop_id', 'category']).agg({'item_cnt_month': ['mean']})
temp.columns = ['avg_shop_cnt_category']
temp = temp.reset_index()
df = pd.merge(df, temp, how = 'left')


#adding in average units sold per category
temp = df[['date_block_num', 'category', 'item_cnt_month']]
temp = temp.groupby(['category']).agg({'item_cnt_month': ['mean']})
temp.columns = ['avg_category_cnt']
temp = temp.reset_index()
df = pd.merge(df, temp, how = 'left')


#adding in average units sold per item_id
temp = df[['date_block_num', 'item_id', 'item_cnt_month']]
temp = temp.groupby(['item_id']).agg({'item_cnt_month': ['mean']})
temp.columns = ['avg_item_id_cnt']
temp = temp.reset_index()
df = pd.merge(df, temp, how = 'left')


#--------------------------------
#adding in lags
#--------------------------------
#adding in lag for item_id per shop per date_block_num
temp = df[['date_block_num','shop_id','item_id','item_cnt_month']]
temp['date_block_num']= temp['date_block_num']+1
temp.rename(columns={'item_cnt_month': 'lag_1_cnt'}, inplace = True)
df = pd.merge(df, temp, how = 'left')
df["lag_1_cnt"] = df["lag_1_cnt"].fillna(0)
df.lag_1_cnt = df['lag_1_cnt'].astype(np.int16)

temp = df[['date_block_num','shop_id','item_id','item_cnt_month']]
temp['date_block_num']= temp['date_block_num']+2
temp.rename(columns={'item_cnt_month': 'lag_2_cnt'}, inplace = True)
df = pd.merge(df, temp, how = 'left')
df["lag_2_cnt"] = df["lag_2_cnt"].fillna(0)
df.lag_2_cnt = df['lag_2_cnt'].astype(np.int16)

temp = df[['date_block_num','shop_id','item_id','item_cnt_month']]
temp['date_block_num']= temp['date_block_num']+3
temp.rename(columns={'item_cnt_month': 'lag_3_cnt'}, inplace = True)
df = pd.merge(df, temp, how = 'left')
df["lag_3_cnt"] = df["lag_3_cnt"].fillna(0)
df.lag_3_cnt = df['lag_3_cnt'].astype(np.int16)

temp = df[['date_block_num','shop_id','item_id','item_cnt_month']]
temp['date_block_num']= temp['date_block_num']+6
temp.rename(columns={'item_cnt_month': 'lag_6_cnt'}, inplace = True)
df = pd.merge(df, temp, how = 'left')
df["lag_6_cnt"] = df["lag_6_cnt"].fillna(0)
df.lag_6_cnt = df['lag_6_cnt'].astype(np.int16)

temp = df[['date_block_num','shop_id','item_id','item_cnt_month']]
temp['date_block_num']= temp['date_block_num']+9
temp.rename(columns={'item_cnt_month': 'lag_9_cnt'}, inplace = True)
df = pd.merge(df, temp, how = 'left')
df["lag_9_cnt"] = df["lag_9_cnt"].fillna(0)
df.lag_9_cnt = df['lag_9_cnt'].astype(np.int16)

temp = df[['date_block_num','shop_id','item_id','item_cnt_month']]
temp['date_block_num']= temp['date_block_num']+12
temp.rename(columns={'item_cnt_month': 'lag_12_cnt'}, inplace = True)
df = pd.merge(df, temp, how = 'left')
df["lag_12_cnt"] = df["lag_12_cnt"].fillna(0)
df.lag_12_cnt = df['lag_12_cnt'].astype(np.int16)



#adding in lag for item category per month
temp = df[['date_block_num','category','item_cnt_month']]
temp = temp.groupby(['date_block_num','category']). agg({'item_cnt_month' : ['sum']})
temp.columns = ['lag_1_cnt_cat']
temp = temp.reset_index()
temp['date_block_num']= temp['date_block_num']+1
df = pd.merge(df, temp, how = 'left')
df["lag_1_cnt_cat"] = df["lag_1_cnt_cat"].fillna(0)

temp = df[['date_block_num','category','item_cnt_month']]
temp = temp.groupby(['date_block_num','category']). agg({'item_cnt_month' : ['sum']})
temp.columns = ['lag_3_cnt_cat']
temp = temp.reset_index()
temp['date_block_num']= temp['date_block_num']+3
df = pd.merge(df, temp, how = 'left')
df["lag_3_cnt_cat"] = df["lag_3_cnt_cat"].fillna(0)

temp = df[['date_block_num','category','item_cnt_month']]
temp = temp.groupby(['date_block_num','category']). agg({'item_cnt_month' : ['sum']})
temp.columns = ['lag_12_cnt_cat']
temp = temp.reset_index()
temp['date_block_num']= temp['date_block_num']+12
df = pd.merge(df, temp, how = 'left')
df["lag_12_cnt_cat"] = df["lag_12_cnt_cat"].fillna(0)




#adding in lag for shop per month
temp = df[['date_block_num','shop_id','item_cnt_month']]
temp = temp.groupby(['date_block_num','shop_id']). agg({'item_cnt_month' : ['sum']})
temp.columns = ['lag_1_cnt_shop']
temp = temp.reset_index()
temp['date_block_num']= temp['date_block_num']+1
df = pd.merge(df, temp, how = 'left')
df["lag_1_cnt_shop"] = df["lag_1_cnt_shop"].fillna(0)

temp = df[['date_block_num','shop_id','item_cnt_month']]
temp = temp.groupby(['date_block_num','shop_id']). agg({'item_cnt_month' : ['sum']})
temp.columns = ['lag_3_cnt_shop']
temp = temp.reset_index()
temp['date_block_num']= temp['date_block_num']+3
df = pd.merge(df, temp, how = 'left')
df["lag_3_cnt_shop"] = df["lag_3_cnt_shop"].fillna(0)

temp = df[['date_block_num','shop_id','item_cnt_month']]
temp = temp.groupby(['date_block_num','shop_id']). agg({'item_cnt_month' : ['sum']})
temp.columns = ['lag_12_cnt_shop']
temp = temp.reset_index()
temp['date_block_num']= temp['date_block_num']+12
df = pd.merge(df, temp, how = 'left')
df["lag_12_cnt_shop"] = df["lag_12_cnt_shop"].fillna(0)



#adding in lag for item_id per month (overall level)
temp = df[['date_block_num','item_id','item_cnt_month']]
temp = temp.groupby(['date_block_num','item_id']). agg({'item_cnt_month' : ['sum']})
temp.columns = ['lag_1_cnt_item']
temp = temp.reset_index()
temp['date_block_num']= temp['date_block_num']+1
df = pd.merge(df, temp, how = 'left')
df["lag_1_cnt_item"] = df["lag_1_cnt_item"].fillna(0)

temp = df[['date_block_num','item_id','item_cnt_month']]
temp = temp.groupby(['date_block_num','item_id']). agg({'item_cnt_month' : ['sum']})
temp.columns = ['lag_3_cnt_shop']
temp = temp.reset_index()
temp['date_block_num']= temp['date_block_num']+3
df = pd.merge(df, temp, how = 'left')
df["lag_3_cnt_shop"] = df["lag_3_cnt_shop"].fillna(0)

temp = df[['date_block_num','item_id','item_cnt_month']]
temp = temp.groupby(['date_block_num','item_id']). agg({'item_cnt_month' : ['sum']})
temp.columns = ['lag_12_cnt_item']
temp = temp.reset_index()
temp['date_block_num']= temp['date_block_num']+12
df = pd.merge(df, temp, how = 'left')
df["lag_12_cnt_item"] = df["lag_12_cnt_item"].fillna(0)


#dropping info where lags could not be calculated (less then 12 months of data)
df = df[df['date_block_num'] > 11]

#ensuring no null values in final df
df.isna().sum()
df.fillna(0, inplace = True)


#####################################################
#model & predictions
#####################################################

from xgboost import XGBRegressor
from matplotlib.pylab import rcParams

data = df.copy()
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
Y_train = Y_train.clip(0, 20)
Y_valid = Y_valid.clip(0, 20)
del data

model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1, 
    seed=42) 


model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 20)

rcParams['figure.figsize'] = 12, 4

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('submission_4.csv', index=False)

from xgboost import plot_importance
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
plot_features(model, (10,14))





