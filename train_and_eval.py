# Ingham Medical Physics Coding Challenge - September 2020
# Author: <Daniel Al Mouiee>
# Python Script to train and evaluate different machine learning models to:
#
#   1)  perform regression and predict the value of the 'overall_survival_in_days' feature. The 
#       testing performance was evaluated using Mean Aboslute Error (MAE) and R-squared (R2)
#  
#   2)  perform classification and predict the class of the 'event_overall_survival' and 'ajcc_stage' 
#       features. The testing performance was evaluated using True accuracy (TA), F1 score (F1)
#
# Usage:        python train_and_eval.py PATH_TO_DATA_CSV_DIR
#           ie. python train_and_eval.py data
#
# Output:       Plots of the various metrics used to evaluate the testing performance of the models
#               Also, info relating to the state of the script is printed to STDOUT for the user to track if need be
#
# Assumptions:
#   1) The feature 'ajcc_stage' denotes the patient's cancer prognosis
#  
# Limitations:
#
#   1)  The datasets did not provide an explanation for the features and so
#       many were not explored for this challenge due to this ambiguity
#
#   2)  The hyperparameters choosen for the models are not necessarily optimal and so
#       model performance could be improved further by performing a search for an
#       optimal architecture
            
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Set the plots to be (7 inches x 7 inches)
rcParams['figure.figsize'] = 7,7

# Regression models
reg_models = {
    'lr'    : LinearRegression(),
    'rf'    : RandomForestRegressor(max_depth=10, random_state=0),
    'dt'    : DecisionTreeRegressor(),
    'mlp'   : MLPRegressor(hidden_layer_sizes=50, alpha=0.5, random_state=1, max_iter=1000)
}

# Classification models
class_models = {
    'logr'  : LogisticRegression(random_state=10)
}

# Regression output feature
reg_output = 'overall_survival_in_days'

# Classification output features
class_outputs = [   'event_overall_survival',
                    'ajcc_stage'  
                ]

# Method used to convert the 'ajcc_stage' Roman numeral values to integers
def convert_roman_nums_to_int(df_col):
    roman_nums = list(df_col.drop_duplicates())
    ints = range(len(roman_nums))
    return df_col.replace(roman_nums, ints)

# Method to obtain input data for training and testing
def split_X_train_test(df, features):
    return ( df[df["dataset"]=="train"][features], 
             df[df["dataset"]=="test"][features] )

# Method to obtain labels for training and testing
def split_y_train_test(df, out_feat):
    return ( df[df["dataset"]=="train"][out_feat], 
             df[df["dataset"]=="test"][out_feat] )

# Feature selection process that uses the correlation between the input and output via the chosen 
# features. The higher correlating features are chosen when considering a subset of these features. 
def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    
    # Calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    
    # Obtain feature names
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    return cor_feature, cor_list

# Method to plot bar charts which demonstrate the correlation cofficient values produed
# by each individual feature 
def plot_feat_corr_bar(features, coffs, output):
    plt.bar(features, coffs, width=0.7)
    plt.title('Feature Correlation Coefficient plot - output: '+ output)
    plt.xlabel('Features', fontsize=1)
    plt.ylabel('Correlation Coefficient', fontsize=8)
    plt.xticks(rotation=90)

    plt.show()
    plt.clf()

# Method to plot performance metrics across all different model for different number of
# 'significant' input features used for training the algorithms
def plot_all_metric_all_feat_combos(mae, metric, all_feats, models, output):
    plt.title('Testing ' + metric + ' for different models and features, output= '+ output)
    plt.xlabel('Model', fontsize=8)
    plt.ylabel(metric, fontsize=8)

    for li in mae:
        if metric == 'TA' or metric == 'F1':
            plt.scatter(list(models.keys()), li)
        else:
            plt.plot(list(models.keys()), li)
    
    plt.legend([(str(i)+' sign feat.') for i in range(1, len(all_feats)+1)])
    plt.show()

# Method to call respective plot methods based on task type
def plots(all_feats, cor_list, all_met_1, all_met_2, models, task, output):
    plot_feat_corr_bar(all_feats, cor_list, output)
    if task == 'Regression':
        met_1 = 'MAE'
        met_2 = 'R2'
    else:
        met_1 = 'TA'
        met_2 = 'F1'
    plot_all_metric_all_feat_combos(all_met_1, met_1, all_feats, models, output)
    plot_all_metric_all_feat_combos(all_met_2, met_2, all_feats, models, output)

# Method to train the different models and plot their testing performance
def train_and_eval(df_clinical_data, all_feats, out_feat, models, task):
    # Train/test split
    y_train, y_test = split_y_train_test(df_clinical_data, out_feat)

    # var <=> (regression_metric, classification_metric): 
    #   all_metric1 <=> (MAE,TA)
    #   all_metric2 <=> (R2,F1)
    all_met_1 = []
    all_met_2 = []

    for n_feats in range(1, len(all_feats)+1):
        temp_met_1 = []
        temp_met_2 = []

        # First produce input data for all considerable features
        X_train, X_test = split_X_train_test(df_clinical_data, all_feats)
        feats, cor_list = cor_selector(X_train, y_train, n_feats)

        print('Running on', str(len(feats)), 'features')
        for key, value in models.items():

            # Reproduce input data based on these new significant n_feat features 
            X_train, X_test = split_X_train_test(df_clinical_data, feats)

            # Fit model and predict...
            mod = value.fit(X_train, y_train)
            preds = mod.predict(X_test)
            
            if task == 'Regression':
                mae = mean_absolute_error(y_test, preds)
                temp_met_1.append(mae)

                r2 = r2_score(y_test, preds)
                temp_met_2.append(r2)
            else:
                ta = accuracy_score(y_test, preds)
                temp_met_1.append(ta)

                f1 = f1_score(y_test, preds, average='weighted')
                temp_met_2.append(f1)
        all_met_1.append(temp_met_1)
        all_met_2.append(temp_met_2)

    # Task Plots
    print('Producing' , task, 'plots...')
    plots(all_feats, cor_list, all_met_1, all_met_2, models, task, out_feat)

# Main method
def main():
    # Define paths to our data
    data_path = Path(sys.argv[1])
    radiomics_path = data_path.joinpath('HN_Radiomics.csv')
    clinical_data_path = data_path.joinpath('HN_ClinicalData.csv')

    # Load the data
    print('Loading data...')
    df_clinical_data = pd.read_csv(clinical_data_path)
    df_radiomics = pd.read_csv(radiomics_path)

    # Initially, I only consider the last 14 features from the 'Radiomics' dataset, as the rest don't seem to provide
    # meaningful information for the output feature I want to predict
    all_feats = (list(df_radiomics.columns)[-14:])

    # An additional feature named 'count_gtv' which is the number of GTV strcutures that appear for a single patient
    all_feats.append('count_gtv')
    df_gtv_radiomics = df_radiomics[df_radiomics['Structure'].str.startswith('GTV')]
    temp_list=df_gtv_radiomics.groupby('id').size()
    
    # The median is taken for all features (except the 'count_gtv') to get an aggregate score over all patients
    # for GTV strcutures only  
    df_gtv_radiomics = df_gtv_radiomics.groupby('id')[list(set(all_feats)-set(['count_gtv']))].median()
    df_gtv_radiomics['count_gtv'] = temp_list.values

    # Convert features to list to append to df
    for col in all_feats:
        df_clinical_data[col] = list(df_gtv_radiomics[col])
    
    # Convert the Roman numeral values to integers  
    df_clinical_data['ajcc_stage'] = convert_roman_nums_to_int(df_clinical_data['ajcc_stage']) 
    all_feats.append('ajcc_stage')

    # Train and evaluate models for regression task
    print('Training and Evaluating Regression models...')
    train_and_eval(df_clinical_data, all_feats, reg_output, reg_models, 'Regression')

    # Train and evaluate models for classification task
    # Remove 'ajcc_stage' from the trainable features as it is used as an output for a classifiaction
    # task
    all_feats.remove('ajcc_stage')
    for out in class_outputs:
        print('Training and Evaluating Classification models on output ', out, '...')
        train_and_eval(df_clinical_data, all_feats, out, class_models, 'Classification')

# Start of program
main()