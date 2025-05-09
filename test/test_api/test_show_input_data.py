from gempy import generate_example_model
from gempy.core.data.enumerators import ExampleModel


def test_print_structural_frame():
    model = generate_example_model(ExampleModel.ONE_FAULT, compute_model=False)
    print(model.structural_frame)
