import pandas as pd

def create_dataframe(path):
    df = pd.read_excel(path)
    return df

def delete_duplicates(df):
    # Stergerea duplicatelor
    df.drop_duplicates(inplace = True) # inplace = True indica faptul ca fuctia va sterge duplicate din dataframe-ul dat si nu va return unul nou
    return df

def remove_unknown_gender_instances(df):
    # Eliminăm rândurile unde 'Gender' are valoarea 'Unknown'
    return df[df['Gender'] != 'Unknown']

def remove_unknown_race_instances(df):
    # Eliminăm rândurile unde 'Race' are valoarea 'Unknown'
    return df[df['Race'] != 'Unknown']

def replace_unknown_with_median(x):
    # Înlocuiește "Unknown" cu mediană, după ce valorile sunt convertite în numeric
    median_value = pd.to_numeric(x[x != "Unknown"]).median()
    return x.replace("Unknown", median_value)

# Funcția pentru aplicarea pe grupuri
def edit_unknown_values_for_natural_area(df):
    df["The abundance of natural areas"] = df.groupby("Race")["The abundance of natural areas"].transform(
        lambda x: replace_unknown_with_median(x)
    )

    # Convertim în int
    df["The abundance of natural areas"] = df["The abundance of natural areas"].astype(int)
    return df

def encode_dataset(df):
    # Create a copy of the DataFrame
    df_transformed = df.copy()
    
    # Age mapping
    age_mapping = {
        'Less than 1 year': 0.5,
        '1-2 years': 1.5,
        '2-10 years': 6,
        'More than 10 years': 12
    }
    
    # Apply age mapping
    df_transformed['Age'] = df_transformed['Age'].map(age_mapping)
    
    # One-hot encode categorical variables
    categorical_columns = ['Type of housing', 'Zone', 'Race', 'Gender']
    
    # Create one-hot encoded columns
    for column in categorical_columns:
        one_hot = pd.get_dummies(df_transformed[column], prefix=column)
        
        # Add one-hot encoded columns to the transformed DataFrame
        df_transformed = pd.concat([df_transformed, one_hot], axis=1)
        
        # Drop the original categorical column
        df_transformed = df_transformed.drop(column, axis=1)
    
    return df_transformed



def preprocessing():
    # Crearea dataframe-ului
    df = create_dataframe('./Dataset/Dataset.xlsx')

    #stergerea duplicatelor
    df = delete_duplicates(df)

    # Stergerea instantelor unde Gender = Unknown (6 instante)
    df = remove_unknown_gender_instances(df)

    # Stergerea instantelor unde Race = Unknown (79 instante)
    df = remove_unknown_race_instances(df)

    df = edit_unknown_values_for_natural_area(df)

    df = encode_dataset(df)

    return df

df = preprocessing()
df.info()