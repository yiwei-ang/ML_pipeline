# ml_pipeline

```angular2html
from engine.model.model import *
from fastapi_csv import FastAPI_CSV
import pandas as pd

file = "tests\\data\\sample_winequality.csv"
df= pd.read_csv(file)
self = SupervisedModels(input_data=df)
result = self.run_pipeline()

file = "tests\\data\\sample_winequality.csv"
app = FastAPI_CSV(file)


```

# How to Run
Run backend service (ignore): 
```angular2html
uvicorn engine.backend.main:app --host 0.0.0.0 --port 8080 --reload
```
Run frontend service:
```angular2html
streamlit run "C:\Users\User\PycharmProjects\ML_pipeline\engine\frontend\main.py"
```

![image](https://user-images.githubusercontent.com/66100446/177497797-8b2d18a4-2292-4b42-b1d2-1a578521bf34.png)

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
