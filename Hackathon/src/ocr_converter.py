from PIL import Image
import pytesseract
import pandas as pd
import re
from tqdm import tqdm

def preprocess(text :str) -> str:
    lower_case= text.lower()
    symbols = r'[ !@#%^&*()$_\-\+\{\}\[\]\'\|:;"<>,./~?`=\"©™°®¢»«¥“”§—‘’é€]'
    alphabets = r'[jsxyz]'
    cleaned_symbol = re.sub(symbols, '', lower_case)
    cleaned_text = re.sub(alphabets, '', cleaned_symbol)
    print(cleaned_text)
    return cleaned_text


df = pd.read_csv('../dataset/train.csv')
df = df.head(100)

print("Loaded Data")

values = pd.Series(dtype='string')
pre_values= pd.Series(dtype='string')
contains = pd.Series(dtype='string')

for i, link in tqdm(enumerate(df['image_link']), total=len(df)):

    link = link.split('/')[-1]
    file_name = re.findall(r'.*\.jpg', link)[0]

    img = Image.open(f'../images/{file_name}')

    # Tessaract
    tes_text = pytesseract.image_to_string(img, lang='eng')
    tes_text = tes_text.replace(' ', '').replace('\n', '')
    pre_tes_text = preprocess(tes_text)
    # values.at[i] = tes_text
    pre_values.at[i]=pre_tes_text

# df['tessaract_values'] = values
df['preprocessed']=pre_values

csv_file_path = 'output.csv'
df.to_csv(csv_file_path, index=False)  

