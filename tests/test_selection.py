from simple_clipper.models import TranscriptSegment, TranscriptWord, format_timestamp
from simple_clipper.selection import group_adjacent_indices, selections_from_indices


def test_group_adjacent_indices_splits_non_adjacent_ranges() -> None:
    assert group_adjacent_indices([4, 2, 3, 9, 9, 10]) == [(2, 3, 4), (9, 10)]


def test_segment_selection_creates_one_clip_for_adjacent_rows() -> None:
    items = [
        TranscriptSegment(id=0, start=0.0, end=1.0, text="hello"),
        TranscriptSegment(id=1, start=1.0, end=2.5, text="world"),
        TranscriptSegment(id=2, start=4.0, end=5.0, text="later"),
    ]

    selections = selections_from_indices(items, [0, 1])

    assert len(selections) == 1
    assert selections[0].start == 0.0
    assert selections[0].end == 2.5
    assert selections[0].text == "hello world"
    assert selections[0].source_indices == (0, 1)


def test_non_adjacent_segment_selection_creates_multiple_clips() -> None:
    items = [
        TranscriptSegment(id=0, start=0.0, end=1.0, text="first"),
        TranscriptSegment(id=1, start=1.0, end=2.0, text="second"),
        TranscriptSegment(id=2, start=4.0, end=5.0, text="third"),
    ]

    selections = selections_from_indices(items, [0, 2])

    assert [(clip.start, clip.end, clip.text) for clip in selections] == [
        (0.0, 1.0, "first"),
        (4.0, 5.0, "third"),
    ]


def test_word_selection_uses_word_boundaries() -> None:
    items = [
        TranscriptWord(start=0.1, end=0.3, text=" Hello", segment_id=0),
        TranscriptWord(start=0.31, end=0.7, text=" world", segment_id=0),
        TranscriptWord(start=0.9, end=1.2, text=" again", segment_id=0),
    ]

    selections = selections_from_indices(items, [0, 1])

    assert len(selections) == 1
    assert selections[0].start == 0.1
    assert selections[0].end == 0.7
    assert selections[0].text == "Hello world"


def test_format_timestamp_handles_hour_and_millisecond_rollover() -> None:
    assert format_timestamp(59.9998) == "01:00.000"
    assert format_timestamp(62.5) == "01:02.500"
    assert format_timestamp(3661.9998) == "01:01:02.000"
