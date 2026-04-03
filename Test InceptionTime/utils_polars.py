import json
import pandas as pd
import numpy as np
import polars as pl
# from sqlalchemy import create_engine

# --- CONFIGURATION ET CONNEXION ---

def load_thesaurus(filepath: str = 'thesaurus.json') -> dict:
    """Charge la configuration des variables cliniques."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# def get_db_engine(server: str, database: str):
#     """Génère le moteur de connexion SQL Server."""
#     return create_engine(f'mssql+pymssql://@{server}/{database}')

# --- EXTRACTION ---

def extract_standard(engine, encounter_id: int, config: dict) -> pd.DataFrame:
    """Extrait les séries temporelles numériques (Vitals, Labs, Meds)."""
    col_name = config["column"]
    table = config.get("table", "PtAssessment")
    codes = config.get("codes", [])
    
    is_prioritized = isinstance(codes, dict)
    all_codes = codes.get("INV", []) + codes.get("NON_INV", []) if is_prioritized else codes
    codes_sql = "'" + "','".join(all_codes) + "'"
    
    if table == "ptLabResult":
        # Jointure sur attributeId pour l'extraction des laboratoires
        query = f"""
        SELECT pr.utcChartTime, pr.valueNumber 
        FROM {table} pr
        JOIN M_dictionary md ON pr.attributeId = md.attributeId AND pr.interventionId = md.interventionId
        WHERE pr.encounterId = {encounter_id} 
          AND md.dictionaryPropName IN ({codes_sql})
          AND pr.valueNumber IS NOT NULL
        """
    elif table == "patientMedication":
        query = f"""
        SELECT pm.utcChartTime, pm.valueNumber 
        FROM DAR.{table} pm
        JOIN M_dictionary md ON pm.attributeId = md.attributeId AND pm.interventionId = md.interventionId
        WHERE pm.encounterId = {encounter_id} 
          AND md.dictionaryLabel IN ({codes_sql})
          AND pm.valueNumber IS NOT NULL
        """
    else: # PtAssessment
        query = f"""
        SELECT pa.utcChartTime, md.dictionaryPropName, pa.valueNumber 
        FROM {table} pa 
        JOIN M_dictionary md ON pa.attributeId = md.attributeId AND pa.interventionId = md.interventionId
        WHERE pa.encounterId = {encounter_id} 
          AND md.dictionaryPropName IN ({codes_sql}) 
          AND pa.valueNumber IS NOT NULL
        """

    try:
        df = pd.read_sql_query(query, engine)
        if df.empty: return pd.DataFrame()
        
        df['utcChartTime'] = pd.to_datetime(df['utcChartTime'])
        if is_prioritized:
            df['is_inv'] = df['dictionaryPropName'].isin(codes.get("INV", []))
            df['h_round'] = df['utcChartTime'].dt.floor('h')
            has_inv = df.groupby('h_round')['is_inv'].transform('max')
            df = df[df['is_inv'] | (~has_inv)]
            
        df_var = df.groupby('utcChartTime')['valueNumber'].median().reset_index()
        df_var.columns = ['utcChartTime', col_name]
        return df_var.set_index('utcChartTime')
    except Exception as e:
        print(f" [ERREUR SQL STANDARD {col_name}] {e}")
        return pd.DataFrame()

def extract_regex_match(engine, encounter_id: int, config: dict) -> pd.DataFrame:
    """Détecte la ventilation invasive.
    Règle de priorité : si O2DeliveryInt indique un support non-invasif (lunettes, masque, OHD)
    à un timestamp donné, is_ventilated = 0 même si VentMode dit VS ou VEILLE.
    VAC / VSAI / SPN-VSPEP / APNEEVENT sans O2Delivery non-invasif = is_ventilated = 1.
    """
    col_name = config["column"]
    excl = config.get("exclusion_pattern", "")
    query = f"""
    SELECT utcChartTime, dictionaryPropName, valueString FROM DAR.PatientAssessment 
    WHERE encounterId = {encounter_id} AND valueString IS NOT NULL 
      AND (dictionaryPropName LIKE '%VentMode%' OR dictionaryPropName LIKE '%Mode_ventilatoire%' 
           OR dictionaryPropName LIKE '%Respiratory_Support%' OR dictionaryPropName LIKE '%O2_Delivery%'
           OR dictionaryPropName LIKE '%O2DeliveryInt%')
    """
    try:
        df = pd.read_sql_query(query, engine)
        if df.empty: return pd.DataFrame()
        df['utcChartTime'] = pd.to_datetime(df['utcChartTime'])
        df['search_text'] = df['dictionaryPropName'].astype(str) + " " + df['valueString'].astype(str)
        df['is_o2delivery'] = df['dictionaryPropName'].str.contains('O2DeliveryInt', case=False, na=False)
        df['is_non_invasive'] = excl != "" and df['search_text'].str.contains(excl, case=False, na=False)

        # Timestamps où O2Delivery dit explicitement non-invasif → override total
        ts_non_invasive = set(df.loc[df['is_o2delivery'] & df['is_non_invasive'], 'utcChartTime'])

        df[col_name] = 1
        # Forcer 0 sur tous les timestamps non-invasifs (O2Delivery prioritaire)
        df.loc[df['utcChartTime'].isin(ts_non_invasive), col_name] = 0
        # Pour les timestamps sans O2Delivery, appliquer exclusion classique
        mask_no_o2 = ~df['utcChartTime'].isin(ts_non_invasive) & ~df['is_o2delivery']
        df.loc[mask_no_o2 & df['is_non_invasive'], col_name] = 0

        return df.groupby('utcChartTime')[col_name].max().reset_index().set_index('utcChartTime')
    except Exception as e:
        print(f" [ERREUR regex_match {col_name}] {e}")
        return pd.DataFrame()

def extract_categorical_mapping(engine, encounter_id: int, config: dict) -> pd.DataFrame:
    """Pivote les états cliniques (ex: Neurologie)."""
    codes_sql = "'" + "','".join(config["codes"]) + "'"
    query = f"SELECT utcChartTime, dictionaryPropName FROM dar.PatientAssessment WHERE encounterId = {encounter_id} AND dictionaryPropName IN ({codes_sql})"
    try:
        df = pd.read_sql_query(query, engine)
        target_cols = config["columns"]
        if df.empty: return pd.DataFrame()
        df['utcChartTime'] = pd.to_datetime(df['utcChartTime'])
        df_res = df.groupby('utcChartTime')['dictionaryPropName'].last().reset_index()
        mapping = config["mapping"]
        df_res['status'] = df_res['dictionaryPropName'].map(mapping)
        for col in target_cols:
            df_res[col] = (df_res['status'] == col).astype(int)
        return df_res[['utcChartTime'] + target_cols].set_index('utcChartTime')
    except Exception:
        return pd.DataFrame()

def extract_column_value(engine, encounter_id: int, config: dict) -> pd.DataFrame:
    """Extrait une valeur de colonne directe (ex: AGE) depuis PtBedStay."""
    col_name = config["column"]
    table = config.get("table", "PtBedStay")
    query = f"SELECT {col_name}, utcInTime FROM {table} WHERE encounterId = {encounter_id}"
    try:
        df = pd.read_sql_query(query, engine)
        if df.empty: return pd.DataFrame()
        val = df.iloc[0][col_name]
        t0 = pd.to_datetime(df.iloc[0]['utcInTime'])
        return pd.DataFrame({'utcChartTime': [t0], col_name: [float(val)]}).set_index('utcChartTime')
    except Exception:
        return pd.DataFrame()

def extract_fio2_corrected(engine, encounter_id: int, config: dict) -> pd.DataFrame:
    """
    Calcule la FiO2 corrigée à partir des mesures directes et des débits d'O2 (L/min).
    Formule : FiO2 = 21 + (4 * débit_Lmin).
    """
    col_name = config["column"] # "fio2_corr"
    
    # Codes pour FiO2 directe et Débit O2
    fio2_codes = config.get("codes", [])
    o2_flow_codes = [
        'O2FlowInt.O2Flowmsmt', 'O2FlowInt.O2Flowmsmt.ptAdult', 
        'Reglage_Debit_O2.O2FlowMsmt', 'O2_V2.Flow'
    ]
    
    all_codes_sql = "'" + "','".join(fio2_codes + o2_flow_codes) + "'"
    
    query = f"""
    SELECT pa.utcChartTime, md.dictionaryPropName, pa.valueNumber 
    FROM PtAssessment pa JOIN M_dictionary md ON pa.attributeId = md.attributeId 
    WHERE pa.encounterId = {encounter_id} AND md.dictionaryPropName IN ({all_codes_sql}) 
      AND pa.valueNumber IS NOT NULL
    """
    
    try:
        df = pd.read_sql_query(query, engine)
        if df.empty: return pd.DataFrame()
        
        df['utcChartTime'] = pd.to_datetime(df['utcChartTime'])
        
        # Séparation deux types de mesures
        mask_fio2 = df['dictionaryPropName'].isin(fio2_codes)
        
        # Traitement FiO2 directe
        df_fio2 = df[mask_fio2].groupby('utcChartTime')['valueNumber'].median().reset_index()
        df_fio2.columns = ['utcChartTime', 'val_fio2']
        
        # Traitement Débit O2 -> Conversion en FiO2
        df_flow = df[~mask_fio2].groupby('utcChartTime')['valueNumber'].median().reset_index()
        df_flow['val_from_flow'] = 21.0 + (df_flow['valueNumber'] * 4.0)
        df_flow = df_flow[['utcChartTime', 'val_from_flow']]
        
        # Fusion et priorité à la FiO2 directe
        df_merged = pd.merge(df_fio2, df_flow, on='utcChartTime', how='outer').set_index('utcChartTime')
        df_merged[col_name] = df_merged['val_fio2'].fillna(df_merged['val_from_flow'])       
        return df_merged[[col_name]]
    except Exception:
        return pd.DataFrame()

def extract_dialyse(engine, encounter_id: int, config: dict) -> pd.DataFrame:
    """Détecte la dialyse (HDI ou CVVHF) et formate en tenseur exclusif."""
    cols = config.get("columns", ["abs_dialyse", "dialyse_hdi", "dialyse_cvvhf"])
    table = config.get("table", "PtAssessment")

    codes = config.get("codes", {})
    hdi_list   = codes.get("hdi",   []) if isinstance(codes, dict) else []
    cvvhf_list = codes.get("cvvhf", []) if isinstance(codes, dict) else []

    if not table or not hdi_list or not cvvhf_list:
        print(f" [WARN DIALYSE] Table ou codes manquants — dialyse initialisée à 0")
        return pd.DataFrame() 

    hdi_codes   = "'" + "','".join(hdi_list)   + "'"
    cvvhf_codes = "'" + "','".join(cvvhf_list) + "'"

    query = f"""
    SELECT pa.utcChartTime, md.dictionaryPropName, pa.valueNumber
    FROM {table} pa
    JOIN M_dictionary md ON pa.attributeId = md.attributeId
    WHERE pa.encounterId = {encounter_id}
      AND (md.dictionaryPropName IN ({hdi_codes}) OR md.dictionaryPropName IN ({cvvhf_codes}))
      AND pa.valueNumber > 0
    """
    try:
        df = pd.read_sql_query(query, engine)
        if df.empty:
            return pd.DataFrame() 

        df['utcChartTime'] = pd.to_datetime(df['utcChartTime'])
        df['is_hdi']   = df['dictionaryPropName'].isin(hdi_list)
        df['is_cvvhf'] = df['dictionaryPropName'].isin(cvvhf_list)

        df_res = df.groupby('utcChartTime')[['is_hdi', 'is_cvvhf']].max().reset_index()
        df_res[cols[1]] = df_res['is_hdi'].astype(int)
        df_res[cols[2]] = (~df_res['is_hdi'] & df_res['is_cvvhf']).astype(int)
        df_res[cols[0]] = 0  

        return df_res[['utcChartTime'] + cols].set_index('utcChartTime')
    except Exception as e:
        print(f" [ERREUR SQL DIALYSE] {e}")
        return pd.DataFrame()
    

def extract_drip_medication(engine, encounter_id: int, config: dict) -> pd.DataFrame:
    """Extrait une perfusion IVSE depuis DAR.patientMedication.
    DAR.patientMedication est une vue qui contient déjà dictionaryPropName —
    filtre direct sans jointure sur attributeId (NULL dans M_dictionary).
    """
    col_name            = config["column"]
    intervention_labels = config.get("intervention_labels", [])
    prop_names          = config.get("prop_names", [])

    if not intervention_labels or not prop_names:
        print(f" [WARN DRIP] intervention_labels ou prop_names manquants pour {col_name}")
        return pd.DataFrame()

    int_sql  = "'" + "','".join(intervention_labels) + "'"
    prop_sql = "'" + "','".join(prop_names)          + "'"

    query = f"""
    SELECT utcChartTime, valueNumber
    FROM DAR.patientMedication
    WHERE encounterId = {encounter_id}
      AND interventionId IN (
          SELECT interventionId FROM M_dictionary
          WHERE dictionaryLabel IN ({int_sql})
      )
      AND dictionaryPropName IN ({prop_sql})
      AND valueNumber IS NOT NULL
      AND valueNumber > 0
    """
    try:
        df = pd.read_sql_query(query, engine)
        if df.empty:
            return pd.DataFrame()
        df['utcChartTime'] = pd.to_datetime(df['utcChartTime'])
        df_var = df.groupby('utcChartTime')['valueNumber'].sum().reset_index()
        df_var.columns = ['utcChartTime', col_name]
        return df_var.set_index('utcChartTime')
    except Exception as e:
        print(f" [ERREUR SQL DRIP {col_name}] {e}")
        return pd.DataFrame()

def extract_feature(engine, encounter_id: int, config: dict) -> pd.DataFrame:
    """Dispatcher clinique."""
    f_type = config.get("type", "standard")

    if f_type in ("source_only", "derived", "static"):
        return pd.DataFrame()

    if f_type == "fio2_corrected":
        return extract_fio2_corrected(engine, encounter_id, config)
    elif f_type == "column_value":
        return extract_column_value(engine, encounter_id, config)
    elif f_type == "regex_match":
        return extract_regex_match(engine, encounter_id, config)
    elif f_type == "categorical_mapping":
        return extract_categorical_mapping(engine, encounter_id, config)
    elif f_type == "dialyse":
        return extract_dialyse(engine, encounter_id, config)
    elif f_type == "drip_medication":
        return extract_drip_medication(engine, encounter_id, config)
    else:
        return extract_standard(engine, encounter_id, config)

# --- PRÉTRAITEMENT ---
import polars as pl
import pandas as pd
 
 
def apply_generic_aggregation(
    df: pl.DataFrame,
    thesaurus: dict,
    id_col: str,
    time_col: str
) -> pl.DataFrame:
    """Agrège les données selon les méthodes définies (median, sum, max)."""
 
    agg_map = {}
 
    for feat_id, config in thesaurus["features"].items():
        if config.get("type") in ("static", "source_only", "derived"):
            continue
 
        cols = config["column"] if "column" in config else config.get("columns", [])
        method = config.get("agg_method", "median")
 
        if isinstance(cols, list):
            for c in cols:
                agg_map[c] = method
        else:
            agg_map[cols] = method
 
    # Ajouter les colonnes manquantes
    missing_cols = [col for col in agg_map if col not in df.columns]
    if missing_cols:
        df = df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(col) # constante qui vaut None pour toutes les lignes (lit pour littéral)
            for col in missing_cols
        ])
 
    # Construire les expressions d'agrégation
    agg_exprs = []
    for col, method in agg_map.items():
        if method == "median":
            agg_exprs.append(pl.col(col).median().alias(col))
        elif method == "sum":
            agg_exprs.append(pl.col(col).sum().alias(col))
        elif method == "max":
            agg_exprs.append(pl.col(col).max().alias(col))
        elif method == "first":
            agg_exprs.append(pl.col(col).fill_nan(None).drop_nulls().first().alias(col))
        else:
            raise ValueError(f"Méthode d'agrégation non supportée pour {col}: {method}")
 
    return df.group_by([id_col, time_col]).agg(agg_exprs)
 
def apply_generic_imputation(
    df: pl.DataFrame,
    thesaurus: dict,
    id_col: str,
) -> pl.DataFrame:
    """Impute les colonnes selon la stratégie définie dans le thesaurus."""
 
    exprs = []
 
    for feat_id, config in thesaurus["features"].items():
        if config.get("type") in ("static", "source_only", "derived"):
            continue
 
        cols = config["column"] if "column" in config else config.get("columns", [])
        method = config.get("imputation_method")
        default = config.get("default_value", 0)
 
        if not isinstance(cols, list):
            cols = [cols]
 
        for col in cols:
            # Si la colonne n'existe pas, on la crée remplie avec la valeur par défaut
            if col not in df.columns:
                exprs.append(
                    pl.lit(default).cast(pl.Float64).alias(col)
                )
                continue
 
            expr = pl.col(col)
 
            if method == "interpolate_ffill_bfill":
                expr = (
                    expr
                    .interpolate()
                    .over(id_col)
                    .forward_fill()
                    .over(id_col)
                    .backward_fill()
                    .over(id_col)
                )
            # pas d'équivalent polars. Je dois donc temporairement passer en pandas..
            elif method == "interpolate_limit":
                pdf = df.select([id_col, col]).to_pandas()
            
                pdf[col] = (
                    pdf.groupby(id_col)[col]
                    .transform(lambda x: x.interpolate(method="linear", limit=6).ffill().bfill())
                )
            
                df = df.with_columns(
                    pl.Series(col, pdf[col])
                )
            
                continue  # important pour éviter de faire ce qui se trouve après les branchements conditionnels
 
            elif method == "ffill":
                expr = (
                    expr
                    .forward_fill()
                    .over(id_col)
                    .fill_null(default)
                )
 
            elif method == "ffill_zero":
                expr = (
                    expr
                    .forward_fill()
                    .over(id_col)
                    .fill_null(0)
                )
 
            elif method == "ffill_bfill":
                expr = (
                    expr
                    .forward_fill()
                    .over(id_col)
                    .backward_fill()
                    .over(id_col)
                )
 
            elif method == "fillna_zero":
                expr = expr.fill_null(0)
 
            expr = expr.fill_null(default)
 
            exprs.append(expr.alias(col))
 
    return df.with_columns(exprs)