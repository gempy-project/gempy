import gempy as gp
from gempy.core.data.enumerators import ExampleModel


def test_gempy_to_subsurface():
    model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=True)
    
