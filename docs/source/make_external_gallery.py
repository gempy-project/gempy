"""
Modified after https://github.com/pyvista/pyvista/blob/ab70c26edbcfb107286c827bd4914562056219fb/docs/make_external_gallery.py

A helper script to generate the external 2-examples gallery.
"""
import os
from io import StringIO


def format_icon(title, description, link, image):
    body = r"""
   .. grid-item-card:: {}
      :link: {}
      :text-align: center
      :class-title: pyvista-card-title

      .. image:: {}
"""
    content = body.format(title, link, image)
    return content


class Example():
    def __init__(self, title, description, link, image):
        self.title = title
        self.description = description
        self.link = link
        self.image = image

    def format(self):
        return format_icon(self.title, self.description, self.link, self.image)


###############################################################################

articles = dict(
    _1=Example(
        title="Vector Model 1: Construction",
        description="Build a basic structural model",
        link="https://vector-raw-materials.github.io/vector-geology/examples/02_structural_modeling/01_model_1_gempy_step_by_step.html#sphx-glr-examples-02-structural-modeling-01-model-1-gempy-step-by-step-py",
        image="https://vector-raw-materials.github.io/vector-geology/_images/sphx_glr_01_model_1_gempy_step_by_step_001.png",
    ),
    _2=Example(
        title="Vector Model 1: Forward Gravity",
        description="Get the gravity response of the model",
        link="https://vector-raw-materials.github.io/vector-geology/examples/03_forward_engines/02_model_1_gempy_fw_gravity.html#sphx-glr-examples-03-forward-engines-02-model-1-gempy-fw-gravity-py",
        image="https://vector-raw-materials.github.io/vector-geology/_images/sphx_glr_02_model_1_gempy_fw_gravity_003.png",
    ),
    _3=Example(
        title="Spremberg: Importing Boreholes",
        description="Import data from well csv files",
        link="https://vector-raw-materials.github.io/vector-geology/examples/02_structural_modeling/03_model_spremberg_import.html#sphx-glr-examples-02-structural-modeling-03-model-spremberg-import-py",
        image="https://vector-raw-materials.github.io/vector-geology/_images/sphx_glr_03_model_spremberg_import_003.png",
    ),
    _4=Example(
        title="Spremberg: Constructing the Model",
        description="Build a structural model from borehole data",
        link="https://vector-raw-materials.github.io/vector-geology/examples/02_structural_modeling/04_model_spremberg_building.html#sphx-glr-examples-02-structural-modeling-04-model-spremberg-building-py",
        image="https://vector-raw-materials.github.io/vector-geology/_images/sphx_glr_04_model_spremberg_building_008.png",
    ),
)


###############################################################################

def make_example_gallery():
    """Make the example gallery."""
    path = "./external/external_examples.rst"

    with StringIO() as new_fid:
        new_fid.write(
            """
.. _external_examples:

External Examples
==================

Vector Examples
---------------
These are examples from the Vector project.

.. grid:: 3
   :gutter: 1

"""
        )
        # Reverse to put the latest items at the top
        for example in list(articles.values())[::-1]:
            new_fid.write(example.format())

        new_fid.write(
            """

.. raw:: html

    <div class="sphx-glr-clear"></div>


"""
        )
        new_fid.seek(0)
        new_text = new_fid.read()

    # check if it's necessary to overwrite the table
    existing = ""
    if os.path.exists(path):
        with open(path) as existing_fid:
            existing = existing_fid.read()

    # write if different or does not exist
    if new_text != existing:
        with open(path, "w", encoding="utf-8") as fid:
            fid.write(new_text)

    return
