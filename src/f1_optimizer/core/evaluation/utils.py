import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two coordinates via Haversine formula."""
    R = 6371.0  # Earth's radius in km

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def circular_utc_distance(utc1: int, utc2: int) -> int:
    """Shortest circular distance in hours between two UTC offsets."""
    # Normalize to 0-23 range
    utc1_normalized = (utc1 % 24 + 24) % 24
    utc2_normalized = (utc2 % 24 + 24) % 24
    
    # Calculate both directions
    forward = (utc2_normalized - utc1_normalized) % 24
    backward = (utc1_normalized - utc2_normalized) % 24
    
    # Return minimum (shortest path on the circle)
    return min(forward, backward)


def extract_race_sequence(chromosome: list[int]) -> tuple[list[int], list[int]]:
    """Return (race_ids, week_positions) extracted from chromosome."""
    omega = []
    tau = []
    for week, race_id in enumerate(chromosome):
        if race_id < 100:  # Valid race ID
            omega.append(race_id)
            tau.append(week)
    return omega, tau
