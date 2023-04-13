import statsmodels.api as sm
import pandas
from patsy import dmatrices

from util import *

study = study_loader('study-26')
trials_df = change_column_names(preprocess_df(study, normalize=True))

print(trials_df)

formula = 'value ~ ' + ' + '.join(trials_df.columns.drop(['value', 'number']))
y, X = dmatrices(formula, trials_df, return_type='dataframe')

mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())