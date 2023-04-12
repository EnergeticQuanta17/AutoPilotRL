
import pandas as pd

def normalize_df(df, normalize=True):

    for i in ['datetime_start', 'datetime_complete', 'state']:
        df = df.drop(i, axis=1)

    df['duration'] = df['duration'].apply(lambda x: x.total_seconds())
    
    if(normalize):
        df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df

def significant_digits(df, n_digits=4):
    if(n_digits==2):
        return df.applymap(lambda x: f"{x:.2g}")    
    return df.applymap(lambda x: f"{x:.4g}")

def change_column_names(df):
    old_cols = df.columns.to_list()
    new_cols = [col.replace("params_", "") if col.startswith("params_") else col for col in old_cols]
    return df.rename(columns=dict(zip(old_cols, new_cols)))

def reorder_cols_signif(df):
    desired_order = ['number',
                    'value',
                    'duration',
                    'learning_rate',
                    'gamma',
                    'n_epochs',
                    'n_steps',
                    'batch_size',
                    'ent_coef',
                    'bvtradeoff',
                    'clip_range',
                    'clip_range_vf',
                    'vf_coef',
                    'target_kl',
                    'max_grad_norm',                
    ]
    return df.reindex(columns=desired_order)
