import cv2
import re
import pytesseract
import pandas as pd
import units as ut
from PIL import Image
from tqdm import tqdm

def loader(dataset_path,no_of_sample):
    df = pd.read_csv(dataset_path)
    df = df.head(no_of_sample)
    values = pd.Series(dtype='string')
    pre_values= pd.Series(dtype='string')
    contains = pd.Series(dtype='string')


def preprocessing(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #add more preprocessing if the quality increases
    return gray

def ocr(preprocessed_img) -> str:
    tes_text = pytesseract.image_to_string(preprocessed_img, lang='eng')
    return tes_text


def postprocessing(text: str) -> list:
    lower_case= text.lower()
    lower_case.replace('\n', '')
    symbols = r'[ !@#%^&*()$_\-\+\{\}\[\]\'\|:;"<>,./~?`=\"©™°®¢»«¥“”§—‘’é€]'
    alphabets = r'[jsxyz]'
    pattern = r'(\d+..)'
    cleaned_symbol = re.sub(symbols, '', lower_case)
    # cleaned_text = re.sub(alphabets, '', cleaned_symbol)
    cleaned_text = re.findall(pattern, cleaned_symbol)
    # print(cleaned_text)
    return cleaned_text

def match_units(input_list, units_dict, entity_name):
    # Create reverse dictionary for the specified entity_name
    abb_dict = {}
    for category, sub_dict in units_dict.items():
        if category == entity_name:
            for key, value in sub_dict.items():
                abb_dict[key] = value
    
    # List to store results
    results = []
    
    # Iterate through the input list and match last two characters
    for item in input_list:
        if len(item) >= 2:
            last_two_chars = item[-2:]
            result = abb_dict.get(last_two_chars)
            if result:
                print({item,item[:-2]+ " "+ result})
                results.append({item,item[:-2]+ " "+ result})
    return results


if __name__== "__main__" :

    # loader('../dataset/train.csv',100)
    df = pd.read_csv('../dataset/train.csv')
    df = df.head(10)
    values = pd.Series(dtype='string')
    list= pd.Series(dtype='object')
    contains = pd.Series(dtype='string')

    print("Data Loaded :)")

    # for i, row in df.itertuples(index=False):
    #     print(f"Image Link: {row.image_link}")
    #     print(f"Group ID: {row.group_id}")
    #     print(f"Entity Name: {row.entity_name}")
    #     print(f"Entity Value: {row.entity_value}")

    i=0

    for row in df.itertuples(index=False):

        link = row.image_link.split('/')[-1]
        file_name = re.findall(r'.*\.jpg', link)[0]

        img =preprocessing(f'../images/{file_name}')
        tes_text = ocr(img)
        output = postprocessing(tes_text)
        match_units(output,ut.units_dict,row.entity_name)
        list.at[i]=output
        i+=1

    # df['tessaract_values'] = values
    df['List']=list

    csv_file_path = 'output.csv'
    df.to_csv(csv_file_path, index=False)  

