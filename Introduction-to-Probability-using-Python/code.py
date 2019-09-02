# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)

p_a = df[df['fico'].astype(float) >700].shape[0]/df.shape[0]

p_b = df[df['purpose'].astype(str) == 'debt_consolidation'].shape[0]/df.shape[0]

df1 = df[df['purpose'].astype(str) == 'debt_consolidation']

p_a_b = df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]

result = (p_a_b == p_a)

print(result)
# code ends here


# --------------
# code starts here
prob_lp = df[df['paid.back.loan'].astype(str) == 'Yes'].shape[0] /df.shape[0]

prob_cs = df[df['credit.policy'].astype(str) == 'Yes'].shape[0] /df.shape[0]

new_df = df[df['paid.back.loan'].astype(str) == 'Yes']

prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0] / new_df.shape[0]

bayes = (prob_pd_cs * prob_lp) / prob_cs

print(bayes)
# code ends here


# --------------
# code starts here
df['purpose'].value_counts(normalize=True).plot(kind = 'bar', figsize = (20,10))

df1 = df[df['paid.back.loan'].astype(str) == 'No']

df1['purpose'].value_counts(normalize=True).plot(kind = 'bar', figsize = (20,10))
# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()

inst_mean = df['installment'].mean()

df.hist(bins = 10, column = 'installment')

df.hist(bins = 10, column = 'log.annual.inc')

# code ends here


