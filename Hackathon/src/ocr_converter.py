import cv2
import re
import pytesseract
import pandas as pd
from PIL import Image
from tqdm import tqdm
import units as ut
#pytesseract.pytesseract.tesseract_cmd = r'F:\amazon\tes\tesseract.exe'
def loader(dataset_path, no_of_sample):
    # Load dataset and return a DataFrame with a specified number of samples.
    df = pd.read_csv(dataset_path)
    df = df.head(no_of_sample)
    return df

def preprocessing(img_path):
    # Preprocess the image for better OCR accuracy.
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add more preprocessing steps if needed
    return gray

def ocr(preprocessed_img) -> str:
    # Perform OCR on the preprocessed image.
    tes_text = pytesseract.image_to_string(preprocessed_img, lang='eng')
    return tes_text

def postprocessing(text: str) -> list:
    # Clean and extract relevant numerical data from OCR text.
    lower_case = text.lower().replace('\n', '')
    symbols = r'[ !@#%^&*()$_\-\+\{\}\[\]\'\|:;<>,./~?`=\©™°®¢»«¥“”§—‘’é€]' #removed " symbol
    alphabets = r'[jsxyz]'
    pattern = r'(\d+..)'
    cleaned_symbol = re.sub(symbols, '', lower_case)
    cleaned_text = re.findall(pattern, cleaned_symbol)

    for i, string in enumerate(cleaned_text):
        if '"' in string:
            cleaned_text[i] = string.replace('"', 'inch')

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

def main():
    dataset_path = 'path to dataset'
    no_of_sample = 50
    df = loader(dataset_path, no_of_sample)
    df['List'] = pd.Series(dtype='object')
    
    print("Data Loaded :)")
    
    # Iterate over the DataFrame with a progress bar
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
        link = row['image_link'].split('/')[-1]
        file_name = re.findall(r'.*\.jpg', link)[0]
        img_path = f'..\images\\{file_name}'
        #print(img_path)
        img = preprocessing(img_path)
        tes_text = ocr(img)
        output = postprocessing(tes_text)
        matched_units = match_units(output, ut.units_dict, row['entity_name'])
        df.at[i, 'List'] = matched_units
    
    csv_file_path = 'output.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"Results saved to {csv_file_path}")

if __name__ == "__main__":
    main()
