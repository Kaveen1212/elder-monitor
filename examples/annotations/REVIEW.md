# Annotation review log

Every label in this folder is an AI-assisted draft (`provenance` in each file says how it was made). **No person has reviewed any of them yet**, so every row below is *pending human review*. The automated checks in `tests/test_annotations.py` only prove that the files are well formed: known labels, contiguous intervals from 0, events inside the clip, the duration equal to the video's. They do not show that a label matches what happens in the video. Only a person watching the footage can do that.

How to review a clip:

1. Play `examples/data/<clip>.mp4` (downloaded by `examples/run_examples.py`) at normal and at quarter speed.
2. For each interval and event below, check the label and the boundary time. The policy is: an exit starts when the resident stands up off the bed, and a return starts when they sit back on it.
3. Correct the JSON file if needed, and record the change in the *Corrections* column (old → new).
4. Fill in *Reviewer* and *Date*, and set *Status* to `reviewed` or `reviewed with corrections`.
5. After any correction, re-run `python examples/run_examples.py` (and `--heldout`) and update the reported numbers.

| Clip | Set | Intervals | Events to verify | Reviewer | Date | Corrections | Status |
|---|---|---|---|---|---|---|---|
| `pexels_10608105` | held-out | 3 | none | | | | pending human review |
| `pexels_4052925` | held-out | 3 | bed_exit 20.9 s | | | | pending human review |
| `pexels_5983705` | held-out | 2 | none | | | | pending human review |
| `pexels_6130024` | held-out | 3 | none | | | | pending human review |
| `pexels_7505324` | held-out | 2 | none | | | | pending human review |
| `pexels_7938959` | held-out | 4 | bed_exit 6.1 s | | | | pending human review |
| `pexels_8088284` | held-out | 1 | none | | | | pending human review |
| `pexels_8090730` | held-out | 4 | bed_exit 9.7 s | | | | pending human review |
| `pexels_8539659` | held-out | 4 | return_to_bed 7.2 s | | | | pending human review |
| `pexels_8539664` | held-out | 4 | return_to_bed 8.4 s | | | | pending human review |
| `pexels_8591515` | held-out | 2 | return_to_bed 7.9 s | | | | pending human review |
| `pexels_8862331` | held-out | 1 | none | | | | pending human review |
| `pexels_9615483` | held-out | 3 | none | | | | pending human review |
| `pexels_4049556` | development | 5 | bed_exit 13.8 s | | | | pending human review |
| `pexels_4052924` | development | 4 | bed_exit 23.0 s | | | | pending human review |
| `pexels_8090735` | development | 4 | bed_exit 4.3 s | | | | pending human review |
| `roundtrip_4049556` | constructed | 9 | bed_exit 13.8 s, return_to_bed 26.34 s | | | | pending human review |

## Intervals to verify

### `pexels_10608105` (held-out, 34.24 s)

- [ ] 0.0–6.9 s `LYING_IN_BED`
- [ ] 6.9–31.8 s `SITTING_ON_BED`
- [ ] 31.8–34.24 s `LYING_IN_BED`

### `pexels_4052925` (held-out, 21.64 s)

- [ ] 0.0–20.9 s `SITTING_ON_BED`
- [ ] 20.9–21.3 s `STANDING`
- [ ] 21.3–21.64 s `WALKING`
- [ ] event `bed_exit` at 20.9 s

### `pexels_5983705` (held-out, 23.08 s)

- [ ] 0.0–8.8 s `LYING_IN_BED`
- [ ] 8.8–23.08 s `SITTING_ON_BED`

### `pexels_6130024` (held-out, 13.5 s)

- [ ] 0.0–0.3 s `LYING_IN_BED`
- [ ] 0.3–3.6 s `SITTING_ON_BED`
- [ ] 3.6–13.5 s `LYING_IN_BED`

### `pexels_7505324` (held-out, 16.37 s)

- [ ] 0.0–6.5 s `LYING_IN_BED`
- [ ] 6.5–16.37 s `SITTING_ON_BED`

### `pexels_7938959` (held-out, 10.01 s)

- [ ] 0.0–6.1 s `SITTING_ON_BED`
- [ ] 6.1–6.9 s `STANDING`
- [ ] 6.9–7.3 s `WALKING`
- [ ] 7.3–10.01 s `UNKNOWN`
- [ ] event `bed_exit` at 6.1 s

### `pexels_8088284` (held-out, 25.79 s)

- [ ] 0.0–25.79 s `SITTING_ON_BED`

### `pexels_8090730` (held-out, 15.31 s)

- [ ] 0.0–9.7 s `SITTING_ON_BED`
- [ ] 9.7–10.4 s `STANDING`
- [ ] 10.4–12.5 s `WALKING`
- [ ] 12.5–15.31 s `UNKNOWN`
- [ ] event `bed_exit` at 9.7 s

### `pexels_8539659` (held-out, 15.2 s)

- [ ] 0.0–5.2 s `WALKING`
- [ ] 5.2–7.2 s `STANDING`
- [ ] 7.2–7.4 s `SITTING_ON_BED`
- [ ] 7.4–15.2 s `LYING_IN_BED`
- [ ] event `return_to_bed` at 7.2 s

### `pexels_8539664` (held-out, 15.92 s)

- [ ] 0.0–6.4 s `WALKING`
- [ ] 6.4–8.4 s `STANDING`
- [ ] 8.4–8.6 s `SITTING_ON_BED`
- [ ] 8.6–15.92 s `LYING_IN_BED`
- [ ] event `return_to_bed` at 8.4 s

### `pexels_8591515` (held-out, 13.68 s)

- [ ] 0.0–7.9 s `STANDING`
- [ ] 7.9–13.68 s `LYING_IN_BED`
- [ ] event `return_to_bed` at 7.9 s

### `pexels_8862331` (held-out, 35.4 s)

- [ ] 0.0–35.4 s `SITTING_ON_BED`

### `pexels_9615483` (held-out, 38.52 s)

- [ ] 0.0–15.0 s `LYING_IN_BED`
- [ ] 15.0–31.9 s `SITTING_ON_BED`
- [ ] 31.9–38.52 s `LYING_IN_BED`

### `pexels_4049556` (development, 17.57 s)

- [ ] 0.0–5.4 s `LYING_IN_BED`
- [ ] 5.4–13.8 s `SITTING_ON_BED`
- [ ] 13.8–15.0 s `STANDING`
- [ ] 15.0–16.0 s `WALKING`
- [ ] 16.0–17.57 s `UNKNOWN`
- [ ] event `bed_exit` at 13.8 s

### `pexels_4052924` (development, 27.88 s)

- [ ] 0.0–23.0 s `SITTING_ON_BED`
- [ ] 23.0–23.6 s `STANDING`
- [ ] 23.6–26.0 s `WALKING`
- [ ] 26.0–27.88 s `UNKNOWN`
- [ ] event `bed_exit` at 23.0 s

### `pexels_8090735` (development, 7.67 s)

- [ ] 0.0–4.3 s `SITTING_ON_BED`
- [ ] 4.3–4.6 s `STANDING`
- [ ] 4.6–6.6 s `WALKING`
- [ ] 6.6–7.67 s `UNKNOWN`
- [ ] event `bed_exit` at 4.3 s

### `roundtrip_4049556` (constructed, 44.13 s)

- [ ] 0.0–5.4 s `LYING_IN_BED`
- [ ] 5.4–13.8 s `SITTING_ON_BED`
- [ ] 13.8–15.0 s `STANDING`
- [ ] 15.0–16.0 s `WALKING`
- [ ] 16.0–24.14 s `UNKNOWN`
- [ ] 24.14–25.14 s `WALKING`
- [ ] 25.14–26.34 s `STANDING`
- [ ] 26.34–34.74 s `SITTING_ON_BED`
- [ ] 34.74–44.13 s `LYING_IN_BED`
- [ ] event `bed_exit` at 13.8 s
- [ ] event `return_to_bed` at 26.34 s
