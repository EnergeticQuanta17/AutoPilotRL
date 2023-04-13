import statsmodels.api as sm
import pandas as pd
import numpy as np
from patsy import dmatrices

df = pd.DataFrame({'x': np.arange(1, 11),
                   'y': np.arange(2, 22, 2)})

formula = 'y ~ x'
y, X = dmatrices(formula, df, return_type='dataframe')

mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())