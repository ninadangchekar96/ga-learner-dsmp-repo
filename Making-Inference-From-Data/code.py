# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]
data = pd.read_csv(path)

# Create a sample of 'data' using "sample()"
data_sample = data.sample(n=sample_size, random_state = 0)

# the mean of installment column of 'sample_data'
sample_mean = np.mean(data_sample['installment'])

# the standard deviation of installment column of 'sample_data'
sample_std = round(data_sample['installment'].std(),2)

# Calculate Margin error
margin_of_error = z_critical * (sample_std/math.sqrt(sample_size))

# Find the confindence interval
confidence_interval = (round(sample_mean - margin_of_error,2),
                       round(sample_mean + margin_of_error,2))  

# the mean of installment column
true_mean = np.mean(data['installment'])

print(sample_std)
print(confidence_interval[0])

if true_mean > confidence_interval[0] and true_mean < confidence_interval[0] :
    print('True mean falls in the range of confidence interval')
else :
    print('True mean does not fall in the range of confidence interval')


#Code starts here



# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig,axes = plt.subplots(3,1)

for i in range(len(sample_size)) :
    m = []
    for j in range(1000) :
        m.append(data['installment'].sample(n=sample_size[i], random_state=0).mean())
        mean_series = pd.Series(m)
    axes[i].hist(mean_series)


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = data['int.rate'].apply(lambda rate : rate.replace('%','')).astype(float)

data['int.rate'] = data['int.rate'] / 100

z_statistic,p_value = ztest(x1 = data[data['purpose']=='small_business']['int.rate'], value = data['int.rate'].mean(), alternative = 'larger')

# check the p-value
if p_value < 0.05 :
   print('There is no difference in interest rate being given to people with purpose as small_business')
else :
   print('Interest rate being given to people with purpose as small_business is higher than the average interest rate')




# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic,p_value = ztest(x1 = data[data['paid.back.loan']=='No']['installment'], x2 = data[data['paid.back.loan']=='Yes']['installment'])

# check the p-value
if p_value < 0.05 :
   print('There is no difference in installments being paid by loan defaulters and loan non defaulters')
else :
   print('There is difference in installments being paid by loan defaulters and loan non defaulters')


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()

no = data[data['paid.back.loan']=='No']['purpose'].value_counts()

observed = pd.concat([yes.transpose(),no.transpose()],axis=1,keys = ['Yes','No'])

chi2, p, dof, ex  = chi2_contingency(observed)

# check the p-value
if chi2 > critical_value :
   print('Distribution of purpose across all customers is same.')
else :
   print('Distribution of purpose for loan defaulters and non defaulters is different.')


