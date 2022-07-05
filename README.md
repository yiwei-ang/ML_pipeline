# ml_pipeline

```angular2html
from engine.model.model import *
import pandas as pd

df= pd.read_csv("tests\\data\\sample_winequality.csv")
self = SupervisedModels(input_data=df)
result = self.run_pipeline()


```

# How to Run
Run backend service: 
```angular2html
uvicorn engine.backend.main:app --host 0.0.0.0 --port 8080 --reload
```
Run frontend service:
```angular2html
streamlit run "C:\Users\User\PycharmProjects\ML_pipeline\engine\frontend\main.py"
```

# To-Do
* Add more analysis to UI:
  * Feature importance
  * AUC/ROC
  * Model Type
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