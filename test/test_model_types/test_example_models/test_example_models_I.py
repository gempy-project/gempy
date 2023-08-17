import gempy as gp
# ! When importing the model is computed


def test_generate_horizontal_stratigraphic_model():
    from examples.examples.geometries.a01_horizontal_stratigraphic import geo_data as model
    print(model.structural_frame)
    assert isinstance(model, gp.data.GeoModel)


def test_generate_fold_model():
    from  examples.examples.geometries.b02_fold import generate_anticline_model
    model = generate_anticline_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_recumbent_fold_model():
    from examples.examples.geometries.c03_recumbent_fold import generate_recumbent_fold_model
    model = generate_recumbent_fold_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_pinchout_model():
    from examples.examples.geometries.d04_pinchout import generate_pinchout_model
    model = generate_pinchout_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_fault_model():
    from examples.examples.geometries.e05_fault import generate_fault_model
    model = generate_fault_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_unconformity_model():
    from examples.examples.geometries.f06_unconformity import generate_unconformity_model
    model = generate_unconformity_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_combination_model():
    from examples.examples.geometries.g07_combination import generate_combination_model
    model = generate_combination_model()
    assert isinstance(model, gp.data.GeoModel)
