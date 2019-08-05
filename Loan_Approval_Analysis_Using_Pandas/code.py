# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path, delimiter = ',')

categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)

numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)

# code ends here


# --------------
# code starts here
banks = bank.drop(['Loan_ID'], axis = 1)
print("Check Null Values : ")
print(banks.isnull().sum())
print("="*50)

bank_mode = banks.mode().iloc[0]

print(bank_mode)
print("="*50)

banks.fillna(bank_mode, inplace = True)

print("Check cleaned dataframe : ")
print(banks.isnull().sum())
#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks, values='LoanAmount', 
        index=['Gender', 'Married','Self_Employed'],
        aggfunc=np.mean)

print(avg_loan_amount)
# code ends here



# --------------
# code starts here
loan_approved_se = len(banks[banks['Self_Employed'] == 'Yes'][banks['Loan_Status'] == 'Y'])

loan_approved_nse = len(banks[banks['Self_Employed'] == 'No'][banks['Loan_Status'] == 'Y'])

percentage_se = loan_approved_se / 614 * 100

percentage_nse = loan_approved_nse / 614 * 100

print("Percentage of loan approval for self employed people : ")
print(percentage_se)

print("Percentage of loan approval for non self employed people : ")
print(percentage_nse)

# code ends here


# --------------
# code starts here
#Function to convert Loan_Amount_Term which is in months to year 
loan_term = banks['Loan_Amount_Term'].apply(lambda months : months / 12)
print(loan_term)

#Find Applicants having loan amount term greater than or equal to 25 years
big_loan_term = len(loan_term[loan_term >= 25])
print(big_loan_term)
# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby(by='Loan_Status')

loan_groupby = loan_groupby[['ApplicantIncome','Credit_History']]

mean_values = loan_groupby.mean() 
print(mean_values)

# code ends here


