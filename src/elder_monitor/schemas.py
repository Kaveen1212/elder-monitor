from dataclasses import dataclass, field

LYING_IN_BED = "LYING_IN_BED"
SITTING_ON_BED = "SITTING_ON_BED"
SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
STANDING = "STANDING"
WALKING = "WALKING"
OUT_OF_BED = "OUT_OF_BED"
LYING_ON_FLOOR = "LYING_ON_FLOOR"
UNKNOWN = "UNKNOWN"

ACTIVITY_STATES = (
    LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING,
    WALKING, OUT_OF_BED, LYING_ON_FLOOR, UNKNOWN,
)
IN_BED_STATES = {LYING_IN_BED, SITTING_ON_BED}
LYING_STATES = {LYING_IN_BED, LYING_ON_FLOOR}

IN_BED = "IN_BED"
BED_STATES = (IN_BED, OUT_OF_BED, UNKNOWN)

NORMAL, MONITOR, ALERT = "NORMAL", "MONITOR", "ALERT"
DECISION_RANK = {NORMAL: 0, MONITOR: 1, ALERT: 2}

BED_EXIT = "bed_exit"
RETURN_TO_BED = "return_to_bed"
EVENT_TYPES = (BED_EXIT, RETURN_TO_BED)


def bed_status(activity):
    if activity == UNKNOWN:
        return UNKNOWN
    return IN_BED if activity in IN_BED_STATES else OUT_OF_BED


@dataclass
class Observation:
    t: float
    frame_idx: int = -1
    visible: bool = False
    identity_ok: bool = True
    reason: str = ""
    track_id: int | None = None
    n_persons: int = 0
    det_conf: float = 0.0
    bbox: list | None = None
    keypoints: list | None = None
    kp_conf: float = 0.0
    truncated: bool = False
    torso_angle: float | None = None
    torso_len: float | None = None
    shoulder_width: float | None = None
    leg_angle: float | None = None
    knee_angle: float | None = None
    thigh_ratio: float | None = None
    knee_drop: float | None = None
    thigh_angle: float | None = None
    anchor: list | None = None
    body_in_bed: float = 0.0
    hip_in_bed: bool = False
    feet_in_bed: bool = False
    bed_dist: float | None = None
    near_bed: bool = False
    edge_of_bed: bool = False
    in_chair: bool = False
    speed: float = 0.0
    box_speed: float = 0.0


@dataclass
class Proposal:
    t: float
    label: str
    confidence: float
    reason: str = ""
    source: str = "rules"


@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float = 1.0
    reasons: list = field(default_factory=list)

    @property
    def duration(self):
        return self.end - self.start


@dataclass
class BedEvent:
    event: str
    episode_id: int
    start_sec: float
    confirmed_sec: float
    previous_state: str
    current_state: str
    confidence: float
    decision: str = NORMAL
    evidence: list = field(default_factory=list)


@dataclass
class Verdict:
    time: float | None
    until: float
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
