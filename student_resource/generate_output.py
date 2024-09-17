import os
import pandas as pd
import pytesseract
import cv2
import re
import easyocr
import pytesseract
from tqdm.auto import tqdm
import requests
import src.single_units as ut

# Create Easy OCR Reader for english
reader = easyocr.Reader(["en"])

# Symbols to remove from ocr output
SYMBOLS = r"[ !@#%^&*()$_\-\+\{\}\[\]\'\|:;<>,/~?`=\©™°®¢»«¥“”§—‘’é€]"

# Patter to detect numbers and units
PATTERN = r"(\d+(\.\d+)?\w)"


# Load specified number of samples
def loader(dataset_path, no_of_sample=10):
    # Load dataset and return a DataFrame with a specified number of samples.
    df = pd.read_csv(dataset_path)
    if no_of_sample >= 0:
        df = df.head(no_of_sample)
    return df


# Preprocess the image for better OCR accuracy.
def preprocessing(img_path):
    img = cv2.imread(img_path)
    # Converting to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add more preprocessing steps if needed
    return gray


# Perform OCR on the preprocessed image.
def ocr(preprocessed_img, ocr_method="e") -> str:
    # If t, use pytesseract
    if ocr_method == "t":
        tes_text = pytesseract.image_to_string(preprocessed_img, lang="eng")
        text = tes_text
    # Else use easyocr
    else:
        easy_text = reader.readtext(preprocessed_img, detail=0)
        text = "".join(easy_text)
    return text


# Clean and extract relevant numerical data from OCR text.
def postprocessing(text: str) -> list:
    # removing all the spaces/tabs ...
    no_spaces = re.sub(r"\s+", "", text)
    # lowercasing
    lower_case = no_spaces.lower()
    # Remove symbols
    cleaned_symbol = re.sub(SYMBOLS, " ", lower_case)
    # Replace " by inch
    cleaned_symbol = cleaned_symbol.replace('"', "inch")
    # Find all numbers and units
    cleaned_text = re.findall(PATTERN, cleaned_symbol)  # returns a list of tuples
    # return list of cleaned text
    cleaned_text_list = [
        item for tup in cleaned_text for item in tup
    ]  # returns a list of strings
    return cleaned_text_list


# Match units based on the units dictionary
def match_units(input_list, units_dict, entity_name):
    # Match units from input_list using the provided units_dict and entity_name.
    abb_dict = {
        key: value
        for category, sub_dict in units_dict.items()
        if category == entity_name
        for key, value in sub_dict.items()
    }

    results = []
    for item in input_list:
        if len(item) >= 2:
            last_one_char = item[-1:]
            result = abb_dict.get(last_one_char)
            if result:
                results.append(f"{item[:-1]} {result}")

    return results


# Get maximum unit
def get_max(units_list):
    if len(units_list) == 0:
        return ""
    max_val = -10000000
    max_val_unit = units_list[0].split(" ")[1]
    for u in units_list:
        parts = u.split(" ")
        num = parts[0]
        if "." in num:
            num = float(num)
        else:
            num = int(num)

        unit = " ".join(parts[1:])
        if num > max_val:
            max_val = num
            max_val_unit = unit

    return f"{max_val} {max_val_unit}"


# Main Predictor function
def predictor(image_link, category_id, entity_name):

    # Download image from link
    img_data = requests.get(image_link).content

    # Store image data in temp image.jpg
    img_path = "image.jpg"
    with open(img_path, "wb") as handler:
        handler.write(img_data)

    # preprocess image
    img = preprocessing(img_path)

    # running ocr (easyocr) on image
    text = ocr(img, "e")

    # postprocessing the text
    output = postprocessing(text)

    # Converting text to actual required output
    matched_units = match_units(output, ut.units_dict, entity_name)

    # Getting the maximum value of output if multiple
    final_output = get_max(matched_units)

    # return final_output
    return final_output


# Running this file standalone
if __name__ == "__main__":
    # Dataset
    DATASET_FOLDER = "./dataset/"

    # tqdm setup to show progress bar in pandas
    tqdm.pandas()

    # Load in the test directory
    test = pd.read_csv(os.path.join(DATASET_FOLDER, "test.csv"))

    # Get all predictions in prediction Series by using pandas progress_apply function
    test["prediction"] = test.progress_apply(
        # Lambda function to get prediction
        lambda row: predictor(row["image_link"], row["group_id"], row["entity_name"]),
        axis=1,
    )

    # Setting output file name
    output_filename = os.path.join(DATASET_FOLDER, "final_output.csv")
    # Saving output to DATASET_FOLDER
    test[["index", "prediction"]].to_csv(output_filename, index=False)
