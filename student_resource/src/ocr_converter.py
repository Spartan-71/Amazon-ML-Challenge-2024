import cv2
import re
import pytesseract
import easyocr
import pandas as pd
from tqdm import tqdm
import os

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

def find_highest_ckpt_number(folder_path):
    # Regular expression to match files with the pattern ckpt###.csv
    pattern = re.compile(r'ckpt(\d+)\.csv')
    highest_num = 0
    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        match = pattern.match(file_name)
        if match:
            # Extract the number from the file name
            file_num = int(match.group(1))
            # Update the highest number if necessary
            if file_num > highest_num:
                highest_num = file_num

    return highest_num

def main():
    dataset_path = './dataset/test.csv'
    no_of_sample = -1
    save_interval = 200
    ocr_method = 'e' # t for tesseract, e for easyocr
    df = loader(dataset_path, no_of_sample)
    # df['List'] = pd.Series(dtype='object')
    start_index = find_highest_ckpt_number("./checkpoints")
    
    print("Data Loaded :)")
    
    # Iterate over the DataFrame with a progress bar
    for i, row in tqdm(df.iloc[start_index:].iterrows(), total=df.shape[0] - start_index, desc="Processing Images"):
        link = row['image_link'].split('/')[-1]
        file_name = re.findall(r'.*\.jpg', link)[0]
        img_path = f'./images/{file_name}'
        
        img = preprocessing(img_path)
        tes_text = ocr(img, ocr_method)

        # Stopping postprocessing for now
        # output = postprocessing(tes_text)
        # matched_units = match_units(output, ut.units_dict, row['entity_name'])
        # df.at[i, 'List'] = matched_units
        df.at[i, ocr_method] = tes_text
    
        if i % save_interval == 0:
            if i == 0 or i == start_index:
                continue
            csv_file_path = f'checkpoints/ckpt{i}.csv'
            df.iloc[i-save_interval:i].to_csv(csv_file_path, index=False)
            print(f"Checkpoint {i} saved to {csv_file_path}")

    csv_file_path = f'checkpoints/final.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"Final Output saved to {csv_file_path}")

if __name__ == "__main__":
    main()
