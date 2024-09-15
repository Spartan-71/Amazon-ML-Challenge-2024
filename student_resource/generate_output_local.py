import os
import pandas as pd
import cv2
import re
from tqdm import tqdm
import src.single_units as ut


def loader(dataset_path, no_of_sample=10):
    # Load dataset and return a DataFrame with a specified number of samples.
    df = pd.read_csv(dataset_path)
    if no_of_sample >= 0:
        df = df.head(no_of_sample)
    return df

def preprocessing(img_path):
    # Preprocess the image for better OCR accuracy.
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add more preprocessing steps if needed
    return gray

def postprocessing(text: str) -> list:
    # Clean and extract relevant numerical data from OCR text.

    #removing all the spaces /tabs ...
    no_spaces=re.sub(r"\s+", "", text)
    #lowercasing
    lower_case = no_spaces.lower()
    symbols = r'[ !@#%^&*()$_\-\+\{\}\[\]\'\|:;"<>,/~?`=\"©™°®¢»«¥“”§—‘’é€]'
    # pattern = r'(\d+..)'
    # pattern = r'(\d+(\.\d+)?\w{2})'
    # pattern = r'(\d+(\.\d{1,})?\w{2})'
    pattern = r'(\d+(\.\d+)?\w)'
    cleaned_symbol = re.sub(symbols, ' ', lower_case)
    cleaned_symbol = cleaned_symbol.replace('"', 'inch')
    cleaned_text = re.findall(pattern, cleaned_symbol) # returns a list of tuples
    cleaned_text = [item for tup in cleaned_text for item in tup] # returns a list of strings
    print('----------------------------------------------------------------------------------------')
    print(f"clr_sym: {cleaned_symbol}")
    print(f"clr_text: {cleaned_text}")
    return cleaned_text

def match_units(input_list, units_dict, entity_name):
    # Match units from input_list using the provided units_dict and entity_name.
    abb_dict = {key: value 
                for category, sub_dict in units_dict.items()
                    if category == entity_name 
                         for key, value in sub_dict.items()}

    results = []
    for item in input_list:
        if len(item) >= 2:
            last_one_char = item[-1:]
            result = abb_dict.get(last_one_char)
            if result:
                results.append(f"{item[:-1]} {result}")
            # else:
            #     last_one_char = item[-1:]
            #     result=abb_dict.get(last_one_char)
            #     results.append(f"{item[:-2]} {result}")
    print(f"category: {entity_name}")
    print(f"matched units: {results}")
    return results

def get_max(units_list):
    if len(units_list) == 0:
        return ""    # print(f"fo: {final_output}")


    max_val = -10000000
    max_val_unit = units_list[0].split(" ")[1]
    for u in units_list:
        parts = u.split(' ')
        num = parts[0]
        if '.' in num:
            num = float(num)
        else:
            num = int(num)

        unit = ' '.join(parts[1:])
        if num > max_val:
            max_val = num
            max_val_unit = unit

    return f'{max_val} {max_val_unit}'

def predictor(image_link, category_id, entity_name, text):
    '''
    Call your model/approach here
    '''
    #TODO
    output = postprocessing(text)
    matched_units = match_units(output, ut.units_dict, entity_name)
    final_output = get_max(matched_units)
    # print(f"out: {output}")
    print(f"fo: {final_output}")
    return final_output

if __name__ == "__main__":
    DATASET_FOLDER = './dataset/'

    # Read the test file
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    data = pd.read_csv(os.path.join(DATASET_FOLDER, 'cleaned_data.csv'))

    start_index = 0
    end_index = test.shape[0]

    # Get all results
    # test['prediction'] = test.progress_apply( lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    for i, row in tqdm(test.iloc[start_index:end_index].iterrows(), total=end_index - start_index, desc="Processing..."):
        data_text = data.at[i, 'e']
        if pd.isna(data_text):
            continue
        test.at[i,'prediction'] = predictor(row['image_link'], row['group_id'], row['entity_name'], data_text)

    # Set output file
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')

    # Store in output file (only index and prediction)
    test[['index', 'prediction']].to_csv(output_filename, index=False)
