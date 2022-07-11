import os
import pytest
import pathlib

from deepdiff import DeepDiff

TEST_DIR = pathlib.Path(__file__).resolve().parent
INPUT_FILE = os.path.join(TEST_DIR, "data", "sample_winequality.csv")
OUTPUT_FILE = os.path.join(TEST_DIR, "data", "expected_output_winequality.json")


@pytest.mark.parametrize("input_data, output_data", [(INPUT_FILE, OUTPUT_FILE)])
def test_integration(SupervisedModelsFixture):
    actual, expected = SupervisedModelsFixture
    actual = {k: v for (k, v) in actual.items() if k in ['metrics', 'model_name']}
    diff = DeepDiff(actual, expected, ignore_numeric_type_changes=True)
    assert diff == {}
