# ml_pipeline

```angular2html
from engine.model.model import *
import pandas as pd

df= pd.read_csv("tests\\data\\sample_winequality.csv")
self = SupervisedModels(input_data=df)
result = self.run_pipeline()


```