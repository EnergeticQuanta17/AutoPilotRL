import pandas as pd
import optuna


def study_loader():
    study_name = "study-26"
    storage_name = f"sqlite:///C:/Users/mpree/Downloads/{study_name}.db"

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )

    return study


study = study_loader()

trials_df = study.trials_dataframe()
print(trials_df.head())
print("\n")
print(trials_df.columns.to_list())