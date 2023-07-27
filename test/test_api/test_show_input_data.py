from examples.examples.geometries.e05_fault import generate_fault_model


def test_print_structural_frame():
    model = generate_fault_model()
    print(model.structural_frame)
