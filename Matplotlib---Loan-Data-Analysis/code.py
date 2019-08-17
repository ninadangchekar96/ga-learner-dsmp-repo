# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Code starts here
data = pd.read_csv(path)
loan_status = data['Loan_Status'].value_counts()

loan_status.plot(kind = 'bar')


# --------------
#Code starts here

#Plotting an unstacked bar plot
property_and_loan=data.groupby(['Property_Area', 'Loan_Status'])
property_and_loan=property_and_loan.size().unstack()
property_and_loan.plot(kind='bar', stacked=False, figsize=(15,10))

#Changing the x-axis label
plt.xlabel('Property_Area')

#Changing the y-axis label
plt.ylabel('Loan_Status')

#Rotating the ticks of X-axis
plt.xticks(rotation=45)

#Code ends here


# --------------
#Code starts here

#Plotting an unstacked bar plot
education_and_loan=data.groupby(['Education', 'Loan_Status']).size().unstack()
education_and_loan.plot(kind='bar', stacked=False, figsize=(15,10))

#Changing the x-axis label
plt.xlabel('Education Status')

#Changing the y-axis label
plt.ylabel('Loan Status')

#Rotating the ticks of X-axis
plt.xticks(rotation=45)

#Code ends here


# --------------
#Code starts here
graduate = data[data['Education'] == 'Graduate']

not_graduate = data[data['Education'] == 'Not Graduate']

#Plot Density dsitribution
graduate['LoanAmount'].plot(kind = 'density', label = 'Graduate')
not_graduate['LoanAmount'].plot(kind = 'density', label = 'Not Graduate')

#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
fig, (ax_1, ax_2, ax_3) = plt.subplots(3,1)

ax_1.scatter(x = data['ApplicantIncome'],y = data['LoanAmount'])
ax_1.set_title('Applicant Income')

ax_2.scatter(x = data['CoapplicantIncome'],y = data['LoanAmount'])
ax_2.set_title('Coapplicant Income')

data['TotalIncome'] = data['ApplicantIncome'] +  data['CoapplicantIncome']

ax_3.scatter(x =data['TotalIncome'],y = data['LoanAmount'])
ax_3.set_title('Total Income')

plt.show()


