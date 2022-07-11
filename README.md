# ML Pipeline

This webapp allows users to train a supervised machine learning model that supports modern algorithms such as simple configuration of K-folds, train test split ratio. It also supports modern algorithm training such as XGboost and Random Forest, hyperparameters tuning.

The application is dockerized and deployed via AWS EC2 instance ( http://44.203.130.230:8501/) to support public usage.

![image](https://user-images.githubusercontent.com/66100446/177497797-8b2d18a4-2292-4b42-b1d2-1a578521bf34.png)


# How to use the application
## Application URL: 
* http://44.203.130.230:8501/  

## Requirement: 
* `wget` command (optional to download sample dataset)
* a CSV dataset with *label/target* on the **rightmost** column. example:
  * | Feature1 | Feature2 | Label | 
    | :---: | :---: | :---: |
    | 1 | 2 | label1 |
## Steps
1. Prepare a sample dataset, else you may use download the following dataset:
```angular2html
wget https://github.com/yiwei-ang/ML_pipeline/blob/main/tests/data/sample_winequality.csv
wget https://github.com/yiwei-ang/ML_pipeline/blob/main/tests/data/sample_iris.csv
```
2. Access the UR,L and upload the dataset, then your result should be ready by seconds!

# How to run locally
1. Install `git`, and the latest anaconda/miniconda from: https://www.anaconda.com/products/distribution
2. Clone repository:
```bash
git clone git@github.com:yiwei-ang/ML_pipeline.git
```
3. Open the anaconda prompt/terminal that has `conda`, run the following to prepare a conda environment:
```bash
conda env create -f environment.yml
conda activate ml_pipeline
```
4. Run the application:
```bash
streamlit run "C:\Users\User\PycharmProjects\ML_pipeline\engine\frontend\main.py"
```
5. (Optional) To run a quick python test:
```bash
from engine.model.model import *
import pandas as pd

file = "tests\\data\\sample_winequality.csv"
df= pd.read_csv(file)
self = SupervisedModels(input_data=df)
result = self.run_pipeline()
```
# To-Do
* Add more analysis to UI:
  * Feature importance
  * AUC/ROC curve
  * Learning curve
* Techdebt:
  * Add make arguments for flexible based on problem (binary/classification)
* Add more features:
  * Config before run
    * Model Type - optional
    * Problem Type
    * Train Test Split ratio
  * Tuning (yes/no)
  * Download results
    * Whether a DB is required for long running training service.
  * Running time and UI
* Schema validator (We can use Pydantic) on POST request.
* Preview Dataset before processing
* Dockerize the process and deployment.
