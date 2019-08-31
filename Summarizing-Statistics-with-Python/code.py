# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path

#Code starts here 
data = pd.read_csv(path)

data['Gender'].replace('-','Agender', inplace = True)

gender_count = data['Gender'].value_counts()
gender_count.plot(kind = 'bar', figsize = (7,6))


# --------------
#Code starts here
alignment = data['Alignment'].value_counts()

plt.pie(alignment,labels =  data['Alignment'].unique())
plt.title('Character Alignment')



# --------------
#Code starts here
#Pearson's correlation between Strength and Combat
sc_df = data[['Strength','Combat']].copy()
sc_covariance = sc_df['Strength'].cov(sc_df['Combat'])
sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()
sc_pearson = round(sc_covariance / (sc_strength * sc_combat), 2)
print("Pearson's correlation coefficient between Strength and Combat : ")
print(sc_pearson)

#Pearson's correlation between Intelligence and Combat
ic_df = data[['Intelligence','Combat']].copy()
ic_covariance = ic_df['Intelligence'].cov(ic_df['Combat'])
ic_intelligence= ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()
ic_pearson = round(ic_covariance / (ic_intelligence * ic_combat), 2)
print("Pearson's correlation coefficient between Intelligence and Combat : ")
print(ic_pearson)



# --------------
#Code starts here
total_high = data['Total'].quantile(0.99)

super_best = data[data['Total'] > total_high]

super_best_names = super_best['Name'].unique().tolist()
print(super_best_names)


# --------------
#Code starts here
fig, [ax_1, ax_2, ax_3] = plt.subplots(1,3, figsize=(20,10))

ax_1.set_title('Intelligence')
ax_1.boxplot(data['Intelligence']);

ax_2.set_title('Speed')
ax_2.boxplot(data['Speed']);

ax_3.set_title('Power')
ax_3.boxplot(data['Power']);








