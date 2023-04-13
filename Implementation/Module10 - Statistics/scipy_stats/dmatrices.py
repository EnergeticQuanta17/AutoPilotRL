from util import *

import numpy as np
from patsy import dmatrices, dmatrix, demo_data

study = study_loader('study-26')
trials_df = change_column_names(preprocess_df(study, normalize=True))

print(trials_df)

# data = trials_df
# print(dmatrices("value ~ learning_rate + n_steps", data))

formula = 'value ~ ' + ' + '.join(trials_df.columns.drop(['value', 'number']))
y, X = dmatrices(formula, trials_df, return_type='dataframe')

print(y)
print(X)

print()
betas, residual, rank_of_X, singular_vectors_X = np.linalg.lstsq(X, y)
betas = betas.ravel()
print(betas)

for name, beta in zip(X.design_info.column_names, betas):
    print("%s: %s" % (name, beta))

if(not(residual)):
    print("\nThe number of regressors are greater than sample size !")
    print("OR Rank deficiency")
else:
    print(f"\nThe residual is: {residual}")

print(f"\nThe rank of X is {rank_of_X}")

print("\nSingluar values of X:", singular_vectors_X)