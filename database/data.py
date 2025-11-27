import sqlite3
import os

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of this script
DB_PATH = os.path.join(BASE_DIR, "../../data/kilter.db")

RING_COLOR_MAP = {
    "12": "green",
    "13": "cyan",
    "14": "purple",
    "15": "orange"
}


# --- Utility functions (get_coordinates_for_hold, parse_frames) remain the same ---
def build_hold_lookup():
    """
    Build a dictionary mapping all hold IDs to their (x, y) coordinates.
    Returns:
        dict: {hold_id: (x, y)}
    """
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        return {}

    hold_lookup = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Join placements -> holes to get coordinates for all holds
        query = """
            SELECT p.id, h.x, h.y
            FROM placements p
            JOIN holes h ON p.hole_id = h.id
            WHERE h.product_id = 1
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        for hold_id, x, y in rows:
            if x is not None and y is not None:
                hold_lookup[str(hold_id)] = (x, y)

    except sqlite3.Error as e:
        print("Database error:", e)
    finally:
        conn.close()

    print(f"✅ Built hold lookup table with {len(hold_lookup)} holds.")
    return hold_lookup

def get_all_hold_ids():
    """Fetch all unique hold IDs from the database and return as a list."""
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        return []

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Fetch all unique hold IDs from the placements table
        cursor.execute("SELECT DISTINCT id FROM placements")
        rows = cursor.fetchall()

        # Convert to a simple list
        hold_ids = [row[0] for row in rows]

    except sqlite3.Error as e:
        print("Database error:", e)
        hold_ids = []
    finally:
        conn.close()

    return hold_ids


def get_coordinates_for_hold(hold_id):
    """Fetch (x, y) coordinates for a hold by its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT hole_id FROM placements WHERE id = ?", (hold_id,))
    placement = cursor.fetchone()
    if not placement:
        conn.close()
        return None
    hole_id = placement[0]
    cursor.execute("SELECT x, y FROM holes WHERE id = ? AND product_id = 1", (hole_id,))
    result = cursor.fetchone()
    conn.close()
    return result


def parse_frames(frames):
    """
    Parse the 'frames' string into a list of dictionaries with
    hold_id, x, y, color_name, and a placeholder for move.
    """
    coordinates = []
    subframes = [f for f in frames.split("p") if f]

    for sub in subframes:
        parts = sub.split("r")
        hold_id = parts[0]
        color_code = parts[1] if len(parts) > 1 else None
        color_name = RING_COLOR_MAP.get(color_code, "unknown")
        coords = get_coordinates_for_hold(hold_id)

        if coords and coords[0] is not None and coords[1] is not None:
            x, y = coords
            coordinates.append({
                "hold_id": hold_id,
                "x": x,
                "y": y,
                "color_name": color_name,
                "move": None
            })
        else:
            # 💥 CHANGE: Raise a specific ValueError instead of printing and returning None
            raise ValueError(f"Missing coordinates for hold ID: {hold_id}")

    return coordinates

# 🛠️ MODIFIED FUNCTION: Accepts angle and limit
def get_climbs(angle, limit=50, offset=0):
    """
    Fetch climbs for a specific angle from the database with support for batching.

    Args:
        angle (int): Angle of the climbs to fetch.
        limit (int): Maximum number of climbs to fetch in one batch.
        offset (int): Number of climbs to skip (for pagination).

    Returns:
        list of tuples: Each tuple is (climb_name, ascensionist_count, frames)
    """
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        return []

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = """
            SELECT c.name, cs.ascensionist_count, c.frames
            FROM climbs c
            JOIN climb_stats cs ON c.uuid = cs.climb_uuid
            WHERE cs.angle = ?
            ORDER BY cs.ascensionist_count DESC
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, (angle, limit, offset))
        climbs = cursor.fetchall()

    except sqlite3.Error as e:
        print("Database error:", e)
        climbs = []

    finally:
        conn.close()

    return climbs

# 🛠️ MODIFIED FUNCTION: Accepts angle and limit
def get_climbs_with_coordinates(angle=40, num_climbs=5, offset=0):
    """Fetch climbs and preprocess frames into coordinates."""

    climbs = get_climbs(angle, num_climbs, offset=offset)
    result = {}

    for climb in climbs:
        name, ascensionist_count, frames = climb

        # 💥 CHANGE: We now wrap the dangerous call in a try/except block.
        # This catches the ValueError raised in parse_frames.
        try:
            coordinates = parse_frames(frames)
        except ValueError as e:
            # Instead of silently skipping, we RERAISE the error.
            # Rerasing will bubble the error up to the generate_dataset function.
            print(f"Propagating error for climb '{name}'...")
            continue

            # If parse_frames succeeds, coordinates is never None
        result[name] = {
            "ascensionist_count": ascensionist_count,
            "coordinates": coordinates
        }
    print(f"Fetched {len(result)} climbs with coordinates for angle {angle}.")
    return result
# Example usage:
# climbs_data = get_climbs_with_coordinates(angle=30, num_climbs=10)
# print(f"Fetched data for {len(climbs_data)} climbs.")