# Captcha processor

The objective of this project is to create a CAPTCHA processor based on a YOLO model from Ultralytics.

## Structure

The project is organized as follows:

- `app.py` - The streamlit app that uses the final model to predict the images, and also shows the original image, the expected characters and the YOLO boxes prediction.
- `explore.py` - A notebook tha contains all the steps, This file contains all the steps, from downloading the data from Kaggle to creating the trained model, including data cleaning and validation.
- `requirements.txt` - This file contains the list of necessary python packages.
- `data/` - This directory contains the following subdirectories:
  - `interin/` - For intermediate data that has been transformed.
  - `processed/` - For the final data to be used for modeling.
  - `raw/` - For raw data without any processing.
 
    
## Setup

**Prerequisites**

The maximum accepted Python version is 3.9, make sure you have it already installed on your computer. You will also need pip for installing the Python packages.

**Installation**

Clone the project repository to your local machine.

Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

## Creating the YOLO model

Follow every step from the explore.ipynb.
You can skip the steps of checking the errors of files, or showing the images and boxes, if you mind.
These steps are necesary to undertand the flow of the proyect, and why decisions are made, but they're not needed in case you only want the model to be created to use it.

## Running the Application

When the model is created you can run the application, execute the app.py script from the "src" folder of the project directory:

```bash
streamlit run app.py
```

