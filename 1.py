import pandas as pd

def clean_diagnosis(diagnosis_str):
    # Remove the record ID and split the diagnosis part
    diagnosis = diagnosis_str.split('[')[0].strip()
    if diagnosis == '-':
        return 'Ne'  # Replace '-' with 'Ne'
    return diagnosis
def clean_record_id(record_id_str):

    if pd.isna(record_id_str):
        return None
    cleaned_id = record_id_str.strip(' -[]')
    return cleaned_id

data_file = 'thyroid0387.data' 
df = pd.read_csv(data_file, delimiter=',', header=None)

# Assign column names based on the attributes you described
columns = [
    'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication',
    'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid',
    'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary',
    'psych', 'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured',
    'TT4', 'T4U measured', 'T4U', 'FTI measured', 'FTI', 'TBG measured',
    'TBG', 'referral source', 'diagnosis'
]



df.columns = columns

# Process the 'diagnosis' column
df['diagnosis'] = df['diagnosis'].apply(clean_diagnosis)
df['diagnosis'] = df['diagnosis'].apply(clean_record_id)



print(df.head())


df.to_csv('cleaned_thyroid_data.csv', index=False)
