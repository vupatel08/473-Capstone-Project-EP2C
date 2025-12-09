## rlt_builder.py
import random
from typing import List, Callable, Tuple
from collections import deque

from dataset_loader import PreferenceDataset, Segment

class RLTBuilder:
    def __init__(self,
                 preference_data: 'PreferenceDataset',
                 dataset_segments: List['Segment'],
                 compare_fn: Callable[[Segment, Segment], float],
                 config: dict = None):
        """
        Initializes the RLTBuilder with preference data, dataset segments, and comparison function.
        Args:
            preference_data (PreferenceDataset): Collected preference labels among segments.
            dataset_segments (List[Segment]): List of all candidate segments to insert.
            compare_fn (Callable): Function taking two segments and returning preference label:
                                 0 - first preferred, 1 - second preferred, 0.5 - equal.
            config (dict): Optional configuration dictionary (not used explicitly here).
        """
        self.preference_data = preference_data
        self.segments = dataset_segments
        self.compare_fn = compare_fn
        # List of groups; each group is a set/list of segments with same preference level
        self.ranked_list: List[List[Segment]] = []

    def construct_rlt(self, seed_segment: 'Segment' = None) -> List[List['Segment']]:
        """
        Constructs the Ranked List of Trajectories (RLT) by sequential insertion.
        Args:
            seed_segment (Segment, optional): Segment to seed the list. Defaults to random.
        Returns:
            List[List[Segment]]: The fully constructed ranked list (ordered groups).
        """
        if not self.segments:
            return []

        # Initialize list with one seed segment in first group
        if seed_segment is None:
            seed_segment = random.choice(self.segments)
        self.ranked_list = [[seed_segment]]

        # Set of segments already inserted (for avoiding duplicates)
        inserted_segments = set([seed_segment.segment_id])

        # For each remaining segment, insert into list
        for segment in self.segments:
            if segment.segment_id in inserted_segments:
                continue  # skip already inserted
            # Insert current segment using binary search
            self._insert_segment(segment)

            inserted_segments.add(segment.segment_id)

        return self.ranked_list

    def _insert_segment(self, segment: 'Segment') -> None:
        """
        Insert a segment into the current RLT list using binary search and preference queries.
        Args:
            segment (Segment): The candidate segment to insert.
        """
        low = 0
        high = len(self.ranked_list) - 1

        # If list is empty, just add the segment
        if high < 0:
            self.ranked_list.append([segment])
            return

        while low <= high:
            mid = (low + high) // 2
            group = self.ranked_list[mid]
            # Select a representative segment from the group for comparison
            # Here, compare with the first segment in group for simplicity
            ref_segment = group[0]

            preference = self._query_preference(segment, ref_segment)

            if preference == 0:
                # segment preferred over ref_segment -> go higher (more preferred group)
                low = mid + 1
            elif preference == 1:
                # ref_segment preferred over segment -> go lower
                high = mid - 1
            else:
                # preference == 0.5, equal preference
                # insert segment into this group
                group.append(segment)
                return

        # After binary search, insert at position low
        # Check for equal preference with neighboring groups
        # First, insert as a new group at position low
        self.ranked_list.insert(low, [segment])

    def _query_preference(self, seg_a: 'Segment', seg_b: 'Segment') -> float:
        """
        Query preference between two segments using compare_fn.
        Args:
            seg_a (Segment): First segment.
            seg_b (Segment): Second segment.
        Returns:
            float: Preference label (0: seg_a preferred, 1: seg_b preferred, 0.5: tie)
        """
        # Use compare_fn which should return preference label accordingly
        # It's expected that compare_fn returns 0, 1, or 0.5
        preference = self.compare_fn(seg_a, seg_b)
        return preference
