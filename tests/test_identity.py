"""Identity recovery after the tracker reassigns track IDs (review of commit 4a1a280)."""
import numpy as np
import pytest

from elder_monitor.features import Scene
from elder_monitor.vision import Person, TargetSelector

from test_units import LYING_ON_BED, STANDING_BESIDE, person

CAREGIVER = {k: (x + 560, y) for k, (x, y) in STANDING_BESIDE.items()}


def frame():
    """Resident's clothes blue (x < 700), the caregiver's red (x >= 700)."""
    f = np.zeros((1000, 1000, 3), np.uint8)
    f[:, :700], f[:, 700:] = (200, 40, 40), (40, 40, 200)
    return f


def resident(track_id):
    return person(LYING_ON_BED, track_id)


def caregiver(track_id):
    return person(CAREGIVER, track_id)


@pytest.fixture
def selector(cfg):
    sel = TargetSelector(cfg, Scene(cfg, 1000, 1000))
    assert sel.select(0.0, frame(), [resident(1), caregiver(2)])[2] == "selected"
    assert sel.select(0.2, frame(), [resident(1), caregiver(2)])[2] == "tracked" and 2 in sel.others
    return sel


def test_resident_given_the_caregivers_old_id_is_recovered(selector):
    assert selector.select(0.4, frame(), [resident(1)])[2] == "tracked"
    assert selector.select(0.6, frame(), [resident(2)])[1:] == (True, "reassociated")
    assert selector.track_id == 2 and 2 not in selector.others
    assert selector.select(0.8, frame(), [resident(2)])[2] == "tracked"


def test_caregiver_given_the_residents_id_is_not_adopted(selector):
    assert selector.select(0.4, frame(), [caregiver(1)]) == (None, False, "identity_uncertain")
    p, ok, reason = selector.select(0.6, frame(), [caregiver(1), resident(3)])
    assert reason == "reassociated" and p.track_id == 3 and 1 in selector.others
    assert selector.select(0.8, frame(), [caregiver(1), resident(3)])[0].track_id == 3


def test_ids_swapped_while_both_are_in_view(selector):
    p, ok, reason = selector.select(0.4, frame(), [resident(2), caregiver(1)])
    assert reason == "reassociated" and p.track_id == 2 and p.bbox == resident(2).bbox
    assert selector.select(0.6, frame(), [resident(2), caregiver(1)])[2] == "tracked"


def test_insufficient_evidence_stays_unknown(selector):
    assert selector.select(0.4, frame(), [caregiver(2)]) == (None, False, "identity_uncertain")
    ambiguous = Person(2, [620, 460, 780, 535], resident(2).keypoints, 0.9)
    assert selector.select(0.6, frame(), [ambiguous]) == (None, False, "identity_uncertain")
    assert selector.track_id == 1
