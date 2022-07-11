import os
import pytest
import pathlib

from deepdiff import DeepDiff

TEST_DIR = pathlib.Path(__file__).resolve().parent
TESTING_DATASET_NAME = ['iris', "winequality"]
INPUT_FILES = [os.path.join(TEST_DIR, "data", "sample_{}.csv").format(i) for i in TESTING_DATASET_NAME]
OUTPUT_FILES = [os.path.join(TEST_DIR, "data", "expected_output_{}.json").format(i) for i in TESTING_DATASET_NAME]
FILES_TUPLE = [(x, y) for x, y in zip(INPUT_FILES, OUTPUT_FILES)]


@pytest.mark.parametrize("input_data, output_data", FILES_TUPLE)
def test_integration(SupervisedModelsFixture):
    actual, expected = SupervisedModelsFixture
    actual = {k: v for (k, v) in actual.items() if k in ['metrics', 'model_name']}
    diff = DeepDiff(actual, expected, ignore_numeric_type_changes=True)
    assert diff == {}
