import numpy as np
from approvaltests import Options, verify
from approvaltests.core import Comparator
from approvaltests.namer import NamerFactory
from approvaltests.reporters import GenericDiffReporter


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
