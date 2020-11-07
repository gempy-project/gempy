import os

import pytest

import gempy as gp
import pooch


def test_load_model(recompute=False):
    """Load model from disk"""
    cwd = os.path.dirname(__file__)
    data_path = cwd + '/../../examples/'

    if recompute:
        geo_model = gp.load_model(r'Tutorial_ch1-8_Onlap_relations',
                                  path=data_path + 'data/gempy_models/Tutorial_ch1'
                                                   '-8_Onlap_relations',
                                  recompile=True)
        gp.compute_model(geo_model)
        gp.plot_3d(geo_model, image=True)

    else:
        geo_model = gp.load_model(r'Tutorial_ch1-8_Onlap_relations',
                                  path=data_path + 'data/gempy_models/Tutorial_ch1'
                                                   '-8_Onlap_relations',
                                  recompile=False)


def test_save_model(one_fault_model_no_interp, tmpdir):
    """Save a model in a zip file with the default name and path"""
    gp.save_model(one_fault_model_no_interp, path=tmpdir)


def test_save_model_solution(one_fault_model_topo_solution, tmpdir):
    """Save a model in a zip file with the default name and path"""
    gp.save_model(one_fault_model_topo_solution,
                  path=tmpdir,
                  solution=True)
    print('foo')

def test_load_model_compressed():
    geo_model = gp.load_model(name="one_fault_model")


def test_load_model_compressed_remote():
    model_file = pooch.retrieve(url="https://github.com/cgre-aachen/gempy_data/raw/master/"
                                    "data/gempy_models/viz_3d.zip",
                                known_hash=None)

    geo_model = gp.load_model(name='viz_3d', path=model_file)


def test_load_model_compressed_remote_fail():
    with pytest.raises(Exception):
        model_file = pooch.retrieve(url="https://nowhere.zip",
                                    known_hash=None)

        geo_model = gp.load_model(name='error', path=model_file)


def test_load_model_compressed_remote2():
    model_file = pooch.retrieve(url="https://github.com/cgre-aachen/gempy_data/raw/master/"
                                    "data/gempy_models/Onlap_relations.zip",
                                known_hash=None)

    geo_model = gp.load_model(name='Onlap_relations', path=model_file, recompile=True)
    gp.compute_model(geo_model)
    gp.plot_3d(geo_model, image=True)


def test_pooch():
    goodboy = pooch.create(
        # Use the default cache folder for the OS
        path=pooch.os_cache("plumbus"),
        # The remote data is on Github
        base_url="https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/data"
                 "/gempy_models/Tutorial_ch1-8_Onlap_relations/",
        # If this is a development version, get the data from the master branch
        version_dev="master",
        # We'll load it from a file later
        registry={
        "Tutorial_ch1-8_Onlap_relations_faults.csv": "19uheidhlkjdwhoiwuhc0uhcwljchw9ochwochw89dcgw9dcgwc"
    },
    )
    print(goodboy)
