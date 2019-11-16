# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Code starts here
plt.figure(figsize=[10,8])
data = pd.read_csv(path)
data.hist(column = 'Rating', bins = 10)

data = data[data['Rating'] <= 5]
data.hist(column = 'Rating', bins = 10)

#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()

percent_null = (total_null/data.isnull().count())

missing_data = pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])
print(missing_data)

data.dropna(inplace=True)
total_null_1 = data.isnull().sum()

percent_null_1 = (total_null_1/data.isnull().count())

missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])
print(missing_data_1)


# code ends here


# --------------

#Code starts here
p = sns.catplot(x="Category", y="Rating", data=data,kind="box",height=10);
p.set_xticklabels(rotation=90)
p.fig.suptitle('Rating vs Category [BoxPlot]')

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].value_counts())

data['Installs'] = data['Installs'].str.replace('+','0')
data['Installs'] = data['Installs'].str.replace(',','0')
data['Installs'] = data['Installs'].str.replace('Free','0')
data['Installs'] = data['Installs'].astype('int64')
print(data['Installs'].head())

le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])

p = sns.regplot(x="Installs", y="Rating", data=data);
p.set_title('Rating vs Installs [RegPlot]')


#Code ends here



# --------------
#Code starts here
data['Price'] = data['Price'].str.replace('$','0')
data['Price'] = data['Price'].astype('float')
print(data['Price'].head())

p = sns.regplot(x="Price", y="Rating", data=data);
p.set_title('Rating vs Price [RegPlot]')


#Code ends here


# --------------

#Code starts here
data['Genres'] = [genre.split(';')[0] for genre in data['Genres']]

gr_mean = data.groupby('Genres',as_index=False)['Rating'].mean()
print(gr_mean.describe())

gr_mean = gr_mean.sort_values(by = ['Rating'])
print(gr_mean.head(2))
print(gr_mean.tail(2))



#Code ends here


# --------------

#Code starts here
print(data['Last Updated'])

data['Last Updated'] = pd.to_datetime(data['Last Updated'], errors='coerce')
print(data['Last Updated'])

max_date = data['Last Updated'].max()
print(max_date)

data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days
print(data['Last Updated Days'])

p = sns.regplot(x="Last Updated Days", y="Rating", data=data);
p.set_title('Rating vs Last Updated Days [RegPlot]')
#Code ends here


