import pandas as pd
import joblib
import os
import sys

def transfo_survie(df, file_path):
    df = df.sort_values(by=['encounterId', 'delta_hour'])

    roll_mean = lambda x: x.rolling(window=24, min_periods=1).mean()
    roll_std  = lambda x: x.rolling(window=24, min_periods=1).std()

    df['pam_moyenne'] = df.groupby('encounterId')['pam'].transform(roll_mean)
    df['pam_sd']      = df.groupby('encounterId')['pam'].transform(roll_std)
    df['pam_sd']      = df['pam_sd'].fillna(0) #une seule valeur

    #gestion heure (heure max = 0)
    df['max_hour'] = df.groupby('encounterId')['delta_hour'].transform('max')
    df['heure'] = df['delta_hour'] - df['max_hour']


    #appel algo
    pack = joblib.load(file_path)

    model = pack['model']
    scaler = pack['scaler']
    colonnes_attendues = pack['features']

    X_service = df[colonnes_attendues]
    X_service_scaled = scaler.transform(X_service)

    df['proba_survie'] =model.predict_proba(X_service_scaled)[:, 1]


    #tableau et tri
    df = df.rename(columns={'encounterId': 'patient'})

    df_long = pd.melt(
        df,
        id_vars=['patient', 'age', 'heure', 'non_survival'],     
        value_vars=['proba_survie'],              
        var_name='score_name',                    
        value_name='valeur'                      
    )

    #tri
    df_long = df_long.sort_values(by=['patient', 'heure'])
    print(f"Données transformées : {len(df_long)} lignes.")
    return df_long