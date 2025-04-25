# 1. Import und Setup
print('Importing Libraries...')
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


import sdv
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

import warnings
warnings.filterwarnings("ignore")

# Set global random seed
rng = np.random.default_rng(seed=187)

# 2. Load and prepare data
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

# Merge the features and targets into a single dataframe
df = pd.concat([X, y], axis=1)
# Replace '?' with NaN
df = df.replace('?', np.nan)

# Drop rows with missing values
df = df.dropna()

# Replace '.' in income
df['income'] = df['income'].replace('<=50K.', '<=50K')
df['income'] = df['income'].replace('>50K.', '>50K')

# Rpelace Target values with binary
df['income'] = df['income'].replace({'>50K': 1, '<=50K': 0}) # >50k = 1, <=50k = 0

# Group all Countries except the top 3 into 'Other'
top_countries = df['native-country'].value_counts().index[:3]
df['native-country'] = df['native-country'].apply(lambda x: x if x in top_countries else 'Other')

# Proprecess Catgorical Columns to minimze risk of missmatch in matrices shape after synthesizing
# drop rows where worklass = without-pay
df = df[df['workclass'] != 'Without-pay']
# replace 5th-6th , 1st-4th and Preschool with 'Primary'
df['education'] = df['education'].replace('5th-6th', 'Primary')
df['education'] = df['education'].replace('1st-4th', 'Primary')
df['education'] = df['education'].replace('Preschool', 'Primary')
# drop rows where Marital status = Married-AF-spouse
df = df[df['marital-status'] != 'Married-AF-spouse']
# drop rows wehere occupation = Armed-Forces
df = df[df['occupation'] != 'Armed-Forces']


# 3. Functions
# Real and Ref Data split (Real will be used for victim training)
def real_ref_split(df, local_seed):
    real, ref = train_test_split(df, test_size=0.5, random_state=local_seed)
    return real, ref

# In and Out Data split (In will be used for training the MIA)
def in_out_split(df, local_seed):
    real_in, real_out = train_test_split(real, test_size=0.5, random_state=local_seed)
    return real_in, real_out

# Funktion zum Data Preprocessing
def preprocess_data(df):
    # Split X and y
    X = df.drop("income", axis=1)
    y = df["income"]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 4. Preprocessing Pipelines definieren
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Preprocessing Pipeline wie gehabt:
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Fit + Transform auf Trainingsdaten
    X_preprocessed = preprocessor.fit_transform(X)
    X_preprocessed = X_preprocessed.toarray()
    return X_preprocessed, y


from scipy.stats import entropy
from sklearn.metrics import log_loss

# Extract richer features for MIA
def extract_attack_features(model, X, y_true=None):
    probs = model.predict_proba(X)
    confs = np.max(probs, axis=1)
    entropies = entropy(probs.T)  # careful: transpose
    if y_true is not None:
        losses = [log_loss([y], [p], labels=[0,1]) for y, p in zip(y_true, probs)]
    else:
        losses = np.zeros(len(confs))  # dummy
    return np.vstack([confs, entropies, losses]).T





# Main function
if __name__ == "__main__":
    print('Start MIA Experiment...')
    real_accuracy = []
    synth_accuracy = []

    #  CV loop
    for i in range(0,5): 
        print('Iteration:', i)
        print('Preparing Datasets')
        local_seed = rng.integers(0, 1000)

        real, ref = real_ref_split(df, local_seed)
        real_in, real_out = in_out_split(real, local_seed)

        print('Creating Synthetic Data')
        # load metadata
        metadata = Metadata.load_from_json(filepath='synth_data/metadata/adult_metadata_v1.json')
        # Copula
        #synthesizer = GaussianCopulaSynthesizer(metadata)
        
        # CTGAN
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(real_in) # Training
        synth_in = synthesizer.sample(num_rows=len(real_in)) # Generating synthetic data

        # Preprocess real and synthetic data
        X_preprocessed_real, y_real = preprocess_data(real_in)
        X_preprocessed_synt, y_synth = preprocess_data(synth_in)

        # Train Victim Model
        print('Training Victim Model')
        # on real data
        dtc_real = DecisionTreeClassifier(random_state=local_seed)
        dtc_real.fit(X_preprocessed_real, y_real)

        # for test
        y_pred = dtc_real.predict(X_preprocessed_real)
        print(f'Accuracy Train: {accuracy_score(y_pred, y_real)}')

        X_preprocessed_real_out, y_real_out = preprocess_data(real_out)
        y_pred_out = dtc_real.predict(X_preprocessed_real_out)
        print(f'Accuracy Test: {accuracy_score(y_pred_out, y_real_out)}')

        # on synthetic data
        dtc_synth = DecisionTreeClassifier(random_state=local_seed)
        dtc_synth.fit(X_preprocessed_synt, y_synth)

        # Train Shadow Model
        print('Training Shadow Model')
        ref_in, ref_out = in_out_split(ref, local_seed)
        X_preprocessed_ref_in, y_ref_in = preprocess_data(ref_in)
        X_preprocessed_ref_out, y_ref_out = preprocess_data(ref_out)
        
        # Train Shadow Model (ref_in)
        dtc_shadow = DecisionTreeClassifier(random_state=local_seed, max_depth=12)
        dtc_shadow.fit(X_preprocessed_ref_in, y_ref_in)

        # Building Trainset for Attack Model
        print('Train Attack Model')

        # Extrahiere Features für das Attack Model
        X_shadow_in = extract_attack_features(dtc_shadow, X_preprocessed_ref_in, y_ref_in)
        X_shadow_out = extract_attack_features(dtc_shadow, X_preprocessed_ref_out, y_ref_out)

        # Labels für in/out
        y_shadow_in = np.ones(len(X_shadow_in))
        y_shadow_out = np.zeros(len(X_shadow_out))

        # Merge für das Attack Model Training
        X_shadow = np.concatenate([X_shadow_in, X_shadow_out])
        y_shadow = np.concatenate([y_shadow_in, y_shadow_out])



        # Train Attack Model
        #attack_model = DecisionTreeClassifier(random_state=local_seed)
        #attack_model = RandomForestClassifier(n_estimators=100, random_state=local_seed)
        attack_model = XGBClassifier(random_state=local_seed)
        attack_model.fit(X_shadow, y_shadow)

        # MIA
        print('Executing MIA')
        # make y_label Vector for real_in and real_out
        y_label_in = np.ones(len(real_in))
        y_label_out = np.zeros(len(real_out))
        # Merge y_label_in and y_label_out
        y_label = np.concatenate([y_label_in, y_label_out]
        )
        # Data Prep real_in and real_out / will be used as test set
        # Preprocessing of both datasets
        X_preprocessed_real_in, y_real_in = preprocess_data(real_in)
        X_preprocessed_real_out, y_real_out = preprocess_data(real_out)

        # Attack on Model
        X_attack_in = extract_attack_features(dtc_real, X_preprocessed_real_in, y_real_in)
        X_attack_out = extract_attack_features(dtc_real, X_preprocessed_real_out, y_real_out)

        attack_pred_in = attack_model.predict(X_attack_in)
        attack_pred_out = attack_model.predict(X_attack_out)


        # Merge attack_pred_in and attack_pred_out
        y_pred_label = np.concatenate([attack_pred_in, attack_pred_out])
        # calculate accuracy
        real_accuracy.append(accuracy_score(y_label, y_pred_label))
        print("MIA Attack Accuracy ML(real Data):", accuracy_score(y_label, y_pred_label))

        # Prediction of the attack model
        X_attack_in_synth = extract_attack_features(dtc_synth, X_preprocessed_real_in, y_real_in)
        X_attack_out_synth = extract_attack_features(dtc_synth, X_preprocessed_real_out, y_real_out)

        attack_pred_in_synth = attack_model.predict(X_attack_in_synth)
        attack_pred_out_synth = attack_model.predict(X_attack_out_synth)


        # Merge attack_pred_in and attack_pred_out (synthetic data)
        y_pred_label_synth = np.concatenate([attack_pred_in_synth, attack_pred_out_synth])

        # calculate accuracy
        accuracy_score(y_label, y_pred_label_synth)
        synth_accuracy.append(accuracy_score(y_label, y_pred_label_synth))
        print("MIA Attack Accuracy ML(synth Data):", accuracy_score(y_label, y_pred_label_synth))

    # Save Results
    df_result = pd.DataFrame(list(zip(real_accuracy, synth_accuracy)), columns=['real_accuracy', 'synth_accuracy'])
    df_result.to_csv('results/MIA_Attack_Results_CTGAN.csv', index=False)


    print('MIA Experiment Finished')
    print('Real Data Accuracy:', np.mean(real_accuracy))
    print('Synthetic Data Accuracy:', np.mean(synth_accuracy))