from __future__ import annotations

from collections.abc import Iterable, Sequence

from .models import ClipSelection, TranscriptSegment, TranscriptWord

TimedTranscriptItem = TranscriptSegment | TranscriptWord


def group_adjacent_indices(indices: Sequence[int]) -> list[tuple[int, ...]]:
    if not indices:
        return []

    sorted_unique = sorted(set(indices))
    groups: list[list[int]] = [[sorted_unique[0]]]
    for index in sorted_unique[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [tuple(group) for group in groups]


def selections_from_indices(
    items: Sequence[TimedTranscriptItem],
    indices: Sequence[int],
) -> list[ClipSelection]:
    selections: list[ClipSelection] = []
    for group in group_adjacent_indices(indices):
        selected_items = [items[index] for index in group]
        start = min(item.start for item in selected_items)
        end = max(item.end for item in selected_items)
        preserve_spacing = all(isinstance(item, TranscriptWord) for item in selected_items)
        text = join_transcript_text((item.text for item in selected_items), preserve_spacing=preserve_spacing)
        selections.append(
            ClipSelection(
                start=start,
                end=end,
                text=text,
                source_indices=group,
            )
        )
    return selections


def join_transcript_text(parts: Iterable[str], preserve_spacing: bool = False) -> str:
    materialized = [str(part) for part in parts]
    if not materialized:
        return ""

    if preserve_spacing:
        joined = "".join(materialized)
    else:
        joined = " ".join(part.strip() for part in materialized)
    return " ".join(joined.split())
