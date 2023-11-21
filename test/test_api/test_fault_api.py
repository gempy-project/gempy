import numpy as np

import gempy as gp
from gempy.core.data.structural_group import FaultsRelationSpecialCase


def test_fault_api():
    frame = _create_structural_frame()
    gp.set_is_fault(
        frame=frame,
        fault_groups=['1', '3'],
        faults_relation_type=FaultsRelationSpecialCase.OFFSET_FORMATIONS,
    )

    print(frame)

    print(frame.fault_relations)
    np.testing.assert_array_equal(
        frame.fault_relations,
        np.array(
            [
                [False, True, False, True],
                [False, False, False, False],
                [False, False, False, True],
                [False, False, False, False]
            ]
        )
    )


def test_fault_api_setter():
    frame = _create_structural_frame()

    # Use the setter to set the fault_relations
    frame.fault_relations = np.array(
        [
            [False, True, False, True],
            [False, False, False, False],
            [False, False, False, True],
            [False, False, False, False]
        ]
    )

    # Now we verify if the StructuralGroup fault_relations have been updated correctly

    assert frame.structural_groups[0].fault_relations == [frame.structural_groups[1], frame.structural_groups[3]]
    assert frame.structural_groups[1].fault_relations == FaultsRelationSpecialCase.OFFSET_NONE
    assert frame.structural_groups[2].fault_relations == FaultsRelationSpecialCase.OFFSET_ALL
    # ! Last group does not matter really. assert frame.structural_groups[3].fault_relations == FaultsRelationSpecialCase.OFFSET_NONE


def _create_structural_frame():
    frame = gp.data.StructuralFrame(
        structural_groups=[
            gp.data.StructuralGroup(
                name='1',
                elements=[],
                structural_relation=gp.data.StackRelationType.ERODE
            ),
            gp.data.StructuralGroup(
                name='2',
                elements=[],
                structural_relation=gp.data.StackRelationType.ERODE
            ),
            gp.data.StructuralGroup(
                name='3',
                elements=[],
                structural_relation=gp.data.StackRelationType.ERODE
            ),
            gp.data.StructuralGroup(
                name='4',
                elements=[],
                structural_relation=gp.data.StackRelationType.ERODE
            ),
        ],
        color_gen=gp.ColorsGenerator()  # * Irrlevant for this test
    )
    return frame
