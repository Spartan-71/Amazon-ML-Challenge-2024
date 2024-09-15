import os
import pandas as pd
import pytesseract
import cv2
import re
import easyocr
from tqdm import tqdm
import requests
import src.units as ut

reader = easyocr.Reader(['en'])

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

def ocr(preprocessed_img, ocr_method='e') -> str:
    # Perform OCR on the preprocessed image.
    if ocr_method == 't':
        tes_text = pytesseract.image_to_string(preprocessed_img, lang='eng')
        text = tes_text
    else:
        easy_text = reader.readtext(preprocessed_img, detail=0)
        text = ''.join(easy_text)
    return text

def postprocessing(text: str) -> list:
    # Clean and extract relevant numerical data from OCR text.
    lower_case = text.lower().replace('\n', '')
    symbols = r'[ !@#%^&*()$_\-\+\{\}\[\]\'\|:;"<>,/~?`=\"©™°®¢»«¥“”§—‘’é€]'
    alphabets = r'[jsxyz]'
    # pattern = r'(\d+..)'
    pattern = r'(\d+(\.\d+)?\w{2})'
    cleaned_symbol = re.sub(symbols, ' ', lower_case)
    cleaned_text = re.findall(pattern, cleaned_symbol)
    cleaned_text = [item for tup in cleaned_text for item in tup]
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
            last_two_chars = item[-2:]
            result = abb_dict.get(last_two_chars)
            if result:
                results.append(f"{item[:-2]} {result}")

    return results

def predictor(image_link, category_id, entity_name):
    '''
    Call your model/approach here
    '''
    #TODO
    link = image_link.split('/')[-1]
    file_name = re.findall(r'.*\.jpg', link)[0]
    img_data = requests.get(image_link).content

    with open(f'./downloaded_images/{file_name}', 'wb') as handler:
        handler.write(img_data)
    img_path = f'./downloaded_images/{file_name}'

    img = preprocessing(img_path)
    text = ocr(img, 'e')

    # Stopping postprocessing for now
    output = postprocessing(text)
    matched_units = match_units(output, ut.units_dict, entity_name)
    print(matched_units)
    return match_units
    # df.at[i, 'List'] = matched_units

if __name__ == "__main__":
    DATASET_FOLDER = './dataset/'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
