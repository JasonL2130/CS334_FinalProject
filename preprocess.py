import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from glove import Corpus, Glove
from gensim.models import Word2Vec

# Drop Features 
def drop_features(df, cols):
    df.drop(columns = cols, inplace=True)
    return df

# One Hot Encoding Categorical Features
def one_hot_encode(df, cols):
    encoded_df = pd.get_dummies(df, columns=cols, dtype=int)
    return encoded_df

# Moving Label Features to End of DF
def label_end(df, cols):
    for feature in cols:
        label_col = df.pop(feature)
        df[feature] = label_col

    return df

# Extracting Date Values from 'trans_date_trans_time' 
def convert_dt(df, col, format):
    df[col] = pd.to_datetime(df[col], format=format)
    return df

# Converting Transaction Time into Ranges --> Morning (00:00-09:59), Afternoon (10:00-16:59), Evening (17:00 - 23:59)
def convert_time_ranges(df, col):
    range_vals = [float('-inf'), 10, 17, float('inf')]
    associated_labels = ["morning", "afternoon", "evening"]

    df[col] = pd.to_datetime(df[col], format='%H:%M')
    df['hour_value'] = df[col].dt.hour

    df[col] = pd.cut(df['hour_value'],
                        bins = range_vals,
                        right = True,
                        labels = associated_labels)
    
    df.drop(columns=['hour_value'], inplace=True)

    return df
    
# Min-Max Scaling --> Only for Training Data
def min_max_scale(train_df, test_df, cols):
    scaler = MinMaxScaler()
    scaler.fit(train_df[cols]) 
    
    train_df[cols] = scaler.transform(train_df[cols]) 
    test_df[cols] = scaler.transform(test_df[cols]) 

    return train_df, test_df

# Train-Test Split of Data
def data_split(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test

# Pearson Correlation Matrix
def pearson_matrix(df):
    corrDF = df.corr(method='pearson')
    return corrDF

# Pearson Correlation --> Delta Feature & Remover
def pearson_drop_delta(orig_df, corr_df, delta):
    def delta_search(correlation_matrix, delta_val):
        delta_list = [] 

        val_len = len(correlation_matrix)

        for row_val in range(val_len-1): 
            for col_val in range(row_val+1, val_len): 
                if np.abs(correlation_matrix.iloc[row_val, col_val]) > delta_val:
                    options = [correlation_matrix.index[row_val], correlation_matrix.columns[col_val]] 
                    delta_list.append(np.random.choice(options)) 

        return delta_list

    delta_list = delta_search(corr_df, delta) 
    final_drop_list = list(set(delta_list)) 
    print(final_drop_list)
    final_DF = orig_df.drop(final_drop_list, axis=1)

    return final_DF

# Pearson Correlation --> Gamma Feature & Remover
def pearson_drop_gamma(orig_df, corr_df, gamma):
    def gamma_search(correlation_matrix, gamma_val):
        gamma_list = [] 

        target_col = corr_df.columns[-1] # Gets DF w/ only Target Column to Focus on... converts into Series Pandas Object (items())

            # Iterates through every Cell from the Target Col
        for row_val, corr_value in corr_df[target_col].items():
            if np.abs(corr_value) < gamma: # Threshold Check
                gamma_list.append(row_val) # Add Val Feature to 'gamma_list'

        return gamma_list

    gamma_list = gamma_search(corr_df, gamma) 
    final_drop_list = list(set(gamma_list)) 
    print(final_drop_list)
    
    return final_drop_list 

# Update Sample Name --> Remove 'fraud_' from merchant
def remove_prefix(df, col_name):
    df[col_name] = df[col_name].str.replace('fraud_', '')
    return df


############################ Vectorizing Categorical Vaiables ############################

def apply_glove(df, value_list):
    corpus_model = Corpus()
    corpus_model.fit(value_list)

    glove_model = Glove(learning_rate = 0.05)
    glove_model.fit(corpus_model.matrix, epochs=30, verbose=True)

    return None