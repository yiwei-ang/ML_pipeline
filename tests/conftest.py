import json
import pytest
import pandas as pd

from engine.model.model import SupervisedModels


@pytest.fixture
def SupervisedModelsFixture(input_data, output_data):
    parsed_dataframe = pd.read_csv(input_data)
    with open(output_data, 'r') as f:
        expected_output = json.load(f)
    return SupervisedModels(input_data=parsed_dataframe).run_pipeline(), expected_output
