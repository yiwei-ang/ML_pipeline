# frontend/main.py

import requests
import streamlit as st
import pandas as pd
from engine.model.model import SupervisedModels
import seaborn as sns
import matplotlib.pyplot as plt


# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Machine Learning Pipeline")
st.markdown("This webapp allows users to train a supervised machine learning model that supports modern algorithms such as simple configuration of K-folds, train test split ratio."  \
            " It also supports modern algorithm training such as XGboost and Random Forest, hyperparameters tuning. \
            ")


# displays a file uploader widget
file = st.file_uploader("Choose an CSV File")

# displays a button
if st.button("Train!"):
    if file:
        df = pd.read_csv(file)
        self = SupervisedModels(input_data=df)
        res_json = self.run_pipeline()

        if res_json:
        # json_data = {"file": pd.read_csv(file).dropna().head(50).to_dict()}
        # res = requests.post(f"http://localhost:8080/", json=json_data)
        # res_json = res.json()
        #
            # CSS to inject contained in a string
            hide_dataframe_row_index = """
                        <style>
                        .row_heading.level0 {display:none}
                        .blank {display:none}
                        </style>
                        """

            # Inject CSS with Markdown
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

            st.header("Best Fitted Model on Training", anchor=None)
            st.text(res_json['model_name'])

            st.header("Evaluation Metrics", anchor=None)
            df = pd.DataFrame.from_dict(res_json['metrics'], orient='index').T
            st.dataframe(df)

            st.header("Confusion Matrix", anchor=None)
            fig = plt.figure(figsize=(10, 4))
            plt.title('Confusion Matrix')
            sns.heatmap(res_json['confusion_matrix'] / np.sum(res_json['confusion_matrix']), annot=True, fmt='.2%', cmap='Blues')
            st.pyplot(fig)
