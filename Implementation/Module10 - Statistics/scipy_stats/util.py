import optuna
import urllib

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

def study_loader(study_name):
    # storage_name may be a problem
    # url need to updated to relative path
    # deactivated getting rawdatafromgithub for now

    url = "https://raw.githubusercontent.com/SBhat2615/AutoPilotRL/main/Implementation/version8%20-%20data%20and%20Graphs/study-26.db"
    storage_name = f"sqlite:///{study_name}.db"

    # with urllib.request.urlopen(url) as response:
    #     with open(study_name + ".db", 'wb') as f:
    #         f.write(response.read())

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )

    return study

def preprocess_df(study, normalize=True):
    trials_df = study.trials_dataframe()

    for i in ['datetime_start', 'datetime_complete', 'state']:
        trials_df = trials_df.drop(i, axis=1)

    trials_df['duration'] = trials_df['duration'].apply(lambda x: x.total_seconds())
    
    if(normalize):
        trials_df = trials_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return trials_df
