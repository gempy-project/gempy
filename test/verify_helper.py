from typing import Literal

import subprocess

import numpy as np
from approvaltests.core import Comparator

import json
from approvaltests import verify, Options
from approvaltests.namer import NamerFactory
from approvaltests.reporters import GenericDiffReporter, GenericDiffReporterConfig

from gempy.core.data import GeoModel
from gempy.modules.serialization.save_load import _load_model_from_bytes, model_to_bytes

class WSLWindowsDiffReporter(GenericDiffReporter):
    def get_command(self, received, approved):
        # Convert WSL paths to Windows paths
        win_received = subprocess.check_output(['wslpath', '-w', received]).decode().strip()
        win_approved = subprocess.check_output(['wslpath', '-w', approved]).decode().strip()

        cmd = [self.path] + self.extra_args + [win_received, win_approved]
        return cmd

def verify_json(item, name: str):

    config = GenericDiffReporterConfig(
        name="custom",
        path=r"pycharm",
        extra_args= ["diff"]
    )

    parameters: Options = NamerFactory \
        .with_parameters(name) \
        .with_reporter(
        reporter=(WSLWindowsDiffReporter(config))
    )
    
    verify(item, options=parameters)


def gempy_verify_array(item, name: str, rtol: float = 1e-5, atol: float = 1e-5, ):
    # ! You will have to set the path to your diff tool
    reporter = GenericDiffReporter.create(
        diff_tool_path=r"/usr/bin/meld"
    )

    parameters: Options = NamerFactory \
        .with_parameters(name) \
        .with_comparator(
        comparator=ArrayComparator(atol=atol, rtol=rtol)
    ).with_reporter(
        reporter=reporter
    )

    verify(np.asarray(item), options=parameters)


class ArrayComparator(Comparator):
    # TODO: Make tolerance a variable
    rtol: float = 1e-05
    atol: float = 1e-05

    def __init__(self, rtol: float = 1e-05, atol: float = 1e-05):
        self.rtol = rtol
        self.atol = atol

    def compare(self, received_path: str, approved_path: str) -> bool:
        from approvaltests.file_approver import exists
        import filecmp
        import pathlib

        if not exists(approved_path) or not exists(received_path):
            return False
        if filecmp.cmp(approved_path, received_path):
            return True
        try:
            approved_raw = pathlib.Path(approved_path).read_text()
            approved_text = approved_raw.replace("\r\n", "\n")
            received_raw = pathlib.Path(received_path).read_text()
            received_text = received_raw.replace("\r\n", "\n")

            # Parse 2D matrices
            received = np.matrix(received_text)
            approved = np.matrix(approved_text)

            allclose = np.allclose(received, approved, rtol=self.rtol, atol=self.atol)
            self.rtol = 1e-05
            self.atol = 1e-05
            return allclose
        except BaseException:
            return False

class JsonSerializer:
    """Serializer that writes JSON with an indent and declares its own extension."""
    def get_default_extension(self) -> str:
        return "json"

    def write(self, received, received_path: str) -> None:
        with open(received_path, "w", encoding="utf-8") as f:
            json.dump(received, f, indent=2, ensure_ascii=False)


def verify_model_serialization(model: GeoModel, verify_moment: Literal["before", "after"], file_name: str):
    """
    Verifies the serialization and deserialization process of a GeoModel instance
    by ensuring the serialized JSON and binary data match during either the
    initial or post-process phase, based on the specified verification moment.

    Args:
        model: The GeoModel instance to be verified.
        verify_moment: A literal value specifying whether to verify the model
            before or after the deserialization process. Accepts "before"
            or "after" as valid inputs.
        file_name: The filename to associate with the verification process for
            logging or output purposes.

    Raises:
        ValueError: If `verify_moment` is not set to "before" or "after".
    """
    binary_file = model_to_bytes(model)

    original_model = model
    original_model.meta.creation_date = "<DATE_IGNORED>"

    if verify_moment == "before":
        verify_json(
            item=original_model.model_dump_json(by_alias=True, indent=4),
            name=file_name
        )
    elif verify_moment == "after":
        model_deserialized = _load_model_from_bytes(binary_file)
        model_deserialized.meta.creation_date = "<DATE_IGNORED>"
        verify_json(
            item=model_deserialized.model_dump_json(by_alias=True, indent=4),
            name=file_name
        )
    else:
        raise ValueError("Invalid model parameter")
