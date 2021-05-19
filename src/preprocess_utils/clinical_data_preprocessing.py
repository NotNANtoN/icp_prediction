import json
import os

import numpy as np
import pandas as pd

from src.preprocess_utils.preprocessing_utils import add_prefix, set_nans
from src.preprocess_utils.plot_utils import make_plots


remove_list = ['complication', 'LOSdays', 'localisation', 'icd0300_v1', 'admscore_v2', 'admscore_v3', 'admscore_v4', 'admscore_v5', 'admscore_v6', 'admscore_v7', 'admscore_v8', 'admscore_v9', 'pain_yes_no', 'GDS_imputed_T1_trial', 'GDS_imputed_T2_trial', 'ICUdays', 'inhouse_mortality_yes_no', 'anesthComb', 'cc_score_post', 'OP_Dauer_min', 'DreiMonatsmortalität', 'ÜberlebenBis90Tage', 'Status_3Monate', 'DeceasedBeforeFU', 'LackOfCompliance_ed2']

# patient specifics:
remove_list_new = ['vs0040_v1', 'clinic', 'explor']
# pre-operatives:
remove_list_new.extend(['AnaesthType', 'LOSdays', 'complication',
                        'Mortality90days', 'Survival90days',
                        'Lack_of_Compliance', 'DeceasedBeforeFU', 'GDS_imputed_T1_trial', 'GDS_imputed_T2_trial',
                        'cc_score_post', 'inhouse_mortality_yes_no', 'ICUdays', 'premedi_benzo_yes_no_v2',
                        'premedi_any_yes_no_v2'])
precipital_factors = ['surgeryDuration',
                      'op0270_v2',  # anaesthesia duration
                      'pain_yes_no',
                      'AnticholMed_OP', 'AnticholMed_POD1',
                      'AnticholMed_POD2', 'AnticholMed_POD3', 'AnticholMed_POD4', 'AnticholMed_POD5',
                      'AnticholMed_POD6', 'AnticholMed_POD7']

# alternatives are used:
remove_list_new.append('ISCED_binary')  # alternative used: ISCED_three_categories
remove_list_new.append('ASA_bin')  # alternative used: icd0300_v1 (more levels than binary)
remove_list_new.append('sm0031_v1')  # smoking history is encoded in other variables
# brain params:
remove_list_new.extend(['BrainVol_cm3_pre', 'BFCS_Vol_mm3_pre', 'NBM_Vol_mm3_pre'])


paper_rename = {'dm0020_v1': 'Sex',
                'dm0030_v1': 'Age',
                'dm0041_v1': 'Height',
                'dm0042_v1': 'Weight',
                'ie0072_v1': 'MMSE score',
                'icd0300_v1': 'ASA physical status',
                'icd0031_v1': 'Arterial hypertension',
                'icd0041_v1': 'Coronary artery disease',
                'icd0121_v1': 'non-insulin dependent diabetes mellitus',
                'icd0111_v1': 'insulin dependent diabetes mellitus',
                'icd0171_v1': 'History of transient ischaemic attack',
                'icd0191_v1': 'History of stroke',
                'BDZ_preop_longterm': 'Preoperative longterm medication with benzodiazepines',
                'AnticholMed_preop': 'Preoperative medication with anticholinergic drugs',
                'ADL_impaired': 'Activities of Daily Living Barthel Index)',
                'IADL_impaired': 'Instrumental activities of daily living (Lawton and Brody)',
                'falling': 'Falling incidences within the last year',
                'TUG': 'Timed up & go test result',
                'fraility': 'Frailty',
                'AUDIT_cat': 'Hazardous alcohol consumption (AUDIT score)',
                'sm0021_v1': 'Current smoker',
                'MNA_cat': 'Mini-Nutritional Assessment - short form',
                'sm0043_v1': 'SPY (smoking pack years)',
                #'sm0031_v1': 'History of smoking',
                'dm0043_v1': 'BMI',
                'ISCED_three_categories': 'ISCED level',
                'PreCI_dichotomous_T0': 'Preoperative cognitive impairment',
                'PastSurgery': 'Any past surgeries',
                'GDS_imputed_T0_trial': 'Geriatric Depression Scale',
                'cc_score_pre': 'Charlson comorbidity index',
                'TumorLymphomaLeukemia': 'Tumor - Lymphoma or Leukemia',
                'EQ5D_Index_baseline': 'EQ5D index'}

json.dump(paper_rename, open("data/renaming_for_paper_clinical.json", 'w+'))


def _clean_clinical_columns(df, lists_path, remove_precipitals=True):
    df = set_nans(df)
    # Pre-filter features
    #allowlist = np.load(os.path.join(lists_path, "feature_lists", "allowlist_clinical_data.npy"))
    #df = df[allowlist]

    # Conditionally remove precipital factors:
    if remove_precipitals:
        df = df.drop(columns=precipital_factors)
    else:
        # surgery or op duration greater than 4 hours instead of raw
        df["op0270_v2"] = df["op0270_v2"] > 4 * 60
        df["surgeryDuration"] = df["surgeryDuration"] > 4 * 60
        antichol_med_names = [col for col in df.columns if "AnticholMed" in col]
        df[antichol_med_names] = df[antichol_med_names] - 1  # for some reason encoded as (1,2) instead of binary (0,1)
        df["Any anticholinergic medication until postoperative day 7"] = df[antichol_med_names].sum(axis=1) > 0
        df = df.drop(columns=antichol_med_names)
        # rename precipital factors
        prec_factor_rename = {"op0270_v2": "Anaesthesia duration over 4h",
                              "surgeryDuration": "Surgery duration over 4h",
                              "pain_yes_no": "Any uncontrolled pain until postoperative day 7",
                              #"AnticholMed_OP": "Anticholinergic medication on day of surgery",
                              #"AnticholMed_POD1": "Anticholinergic medication on day 1 after surgery",
                              #"AnticholMed_POD2": "Anticholinergic medication on day 2 after surgery",
                              #"AnticholMed_POD3": "Anticholinergic medication on day 3 after surgery",
                              #"AnticholMed_POD4": "Anticholinergic medication on day 4 after surgery",
                              #"AnticholMed_POD5": "Anticholinergic medication on day 5 after surgery",
                              #"AnticholMed_POD6": "Anticholinergic medication on day 6 after surgery",
                              #"AnticholMed_POD7": "Anticholinergic medication on day 7 after surgery",
                              }
        df = df.rename(columns=prec_factor_rename)
    df = df.drop(columns=remove_list_new)
    # Change some feature names
    #with open(os.path.join(lists_path, "feature_lists", "clinical_feature_name_remapping.json")) as json_file:
    #    name_remap = json.load(json_file)
    #df = df.rename(columns=name_remap)

    # rename cols:
    # rename cols
    paper_rename = json.load(open("data/renaming_for_paper_clinical_lammers.json"))
    paper_rename.update({'POCD_dichotomous_T2': 'POCD'})
    df = df.rename(columns=paper_rename)

    print(df.columns)


    # Set male to 0 and female to 1:
    df.loc[df["Sex"] == "male", "Sex"] = 0
    df.loc[df["Sex"] == "female", "Sex"] = 1

    # Typecast all feature columns to numeric
    df = df.astype(float)
    #df.loc[:, df.columns != 'subject'] = df.loc[:, df.columns != 'subject'].astype(float)

    make_plots(df, 'clinical')

    return df


def _one_hot_encode(df):
    # map numeric values to string
    map_dict = {
                # 'AnaesthType': {1: 'general', 2: 'regional', 3: 'combined'},
                'localisation': {1: 'intracranial', 2: 'intrathoric|abdominal|pelvic', 3: 'peripheral'},
                }

    def _create_dummy_column(df, series, name):
        # get dummies for feature (=series) and append to df
        one_hot_feat = pd.get_dummies(series, prefix=name)
        one_hot_feat.loc[df[name].isnull(), :] = np.nan  # retain NaNs
        df = pd.concat([df, one_hot_feat], axis=1)
        # drop original column
        df = df.drop(name, axis=1)
        return df

    # get dummies per feature and append to df
    for feat_name in map_dict:
        feature = df[feat_name].map(map_dict[feat_name])
        df = _create_dummy_column(df, feature, feat_name)

    rename_cat = {"localisation_intracranial": "Intracranial surgery",
                  "localisation_peripheral": "Peripheral surgery",
                  "localisation_intrathoric|abdominal|pelvic": "Intrathoric, abdominal or pelvic surgery"
                  }
    df = df.rename(columns=rename_cat)

    return df


def _add_miss_features_for_clinical_data(df):
    # Add extra missingness feature here under special name such that we can filer out others but still keep this one
    df['TUG test result missing'] = df['TUG'].isna()
    
    # Don't encode targets and where there are no missings
    nans = df.isna().drop(columns=["POD", "POCD"])

    for col in nans.columns:
        if nans[col].sum() == 0:
            print("No missings for ", col)
            nans = nans.drop(columns=[col])

    # Missingness features get suffix for identification
    nans = nans.add_suffix("_nan").astype(float)
    df = pd.concat([df, nans], axis=1)
    return df


def load_clinical_data(path, lists_path, miss_feature=True, remove_precipitals=True):
    # Read raw data
    #clinical_df_old = pd.read_csv(os.path.join(path, "core_data_set_20200211_adalab_clinical.csv"), encoding="ISO-8859-1")
    
    clinical_df = pd.read_excel(os.path.join(path, "Core_Data_Set_12-06-2020.xlsx"))#, encoding="ISO-8859-1")
    

    blood_cols = [col for col in clinical_df if col.startswith("T1")]
    blood_cols.append('Final_T1_TP42_40')
    clinical_df = clinical_df.drop(columns=blood_cols)
    
    # Basic cleaning and renaming
    clinical_df = _clean_clinical_columns(clinical_df, lists_path, remove_precipitals=remove_precipitals)
    # One-hot encode categorical variables
    clinical_df = _one_hot_encode(clinical_df)

    # Add clinical data-specific missingness features
    if miss_feature:
        clinical_df = _add_miss_features_for_clinical_data(clinical_df)

    # Add prefix in order to identify the clinical data later
    prefixed_cols = list(set(clinical_df.columns) - set(["POD", "POCD", "Age", "Sex", "Height", "Weight"]))
    clinical_df = add_prefix(clinical_df, prefixed_cols, 'clinical_')

    return clinical_df
