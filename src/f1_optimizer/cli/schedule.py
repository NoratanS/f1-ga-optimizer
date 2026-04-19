"""Schedule formatting for displaying optimized race calendars."""

from dataclasses import dataclass
from f1_optimizer.data_access.schemas import TrackSchema


@dataclass
class ScheduleEntry:
    """A single event in the schedule."""
    week: int
    track_name: str
    location: str
    is_break: bool
    
    def __str__(self) -> str:
        """Format as string like 'Melbourne, Australia (Week 1)' or 'Break'."""
        if self.is_break:
            return f"Break (Week {self.week})"
        return f"{self.track_name}, {self.location} (Week {self.week})"


def format_schedule(
    chromosome: list[int],
    tracks: dict[int, TrackSchema],
) -> list[ScheduleEntry]:
    """Convert chromosome to human-readable schedule.
    
    Args:
        chromosome: List of gene values (track IDs 1..N and break codes 100..100+M)
        tracks: Dictionary of track ID -> TrackSchema
        
    Returns:
        List of ScheduleEntry objects in chronological order
    """
    schedule = []
    
    for week, gene in enumerate(chromosome, start=1):
        if gene >= 100:
            # This is a break week
            schedule.append(ScheduleEntry(
                week=week,
                track_name="",
                location="",
                is_break=True,
            ))
        else:
            # This is a race
            track = tracks[gene]
            schedule.append(ScheduleEntry(
                week=week,
                track_name=track.track_name,
                location=f"{track.city}, {track.country}",
                is_break=False,
            ))
    
    return schedule


def format_schedule_display(schedule: list[ScheduleEntry], compact: bool = False) -> str:
    """Format schedule as a readable string.
    
    Args:
        schedule: List of ScheduleEntry objects
        compact: If True, show only races (skip breaks)
        
    Returns:
        Formatted schedule string
    """
    if compact:
        races = [entry for entry in schedule if not entry.is_break]
        lines = [f"{i}. {entry}" for i, entry in enumerate(races, start=1)]
    else:
        lines = [f"Week {entry.week:2d}: {entry}" for entry in schedule]
    
    return "\n".join(lines)


def get_schedule_summary(schedule: list[ScheduleEntry]) -> dict:
    """Get summary statistics about the schedule.
    
    Args:
        schedule: List of ScheduleEntry objects
        
    Returns:
        Dictionary with summary stats
    """
    races = [entry for entry in schedule if not entry.is_break]
    breaks = [entry for entry in schedule if entry.is_break]
    
    return {
        "total_weeks": len(schedule),
        "num_races": len(races),
        "num_breaks": len(breaks),
        "first_race": races[0].track_name if races else None,
        "last_race": races[-1].track_name if races else None,
    }
