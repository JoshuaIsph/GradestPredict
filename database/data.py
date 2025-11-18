import sqlite3
import os

DB_PATH = "../../data/kilter.db"
RING_COLOR_MAP = {
    "12": "green",
    "13": "cyan",
    "14": "purple",
    "15": "orange"
}


# --- Utility functions (get_coordinates_for_hold, parse_frames) remain the same ---

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
                "move": None  # Placeholder for move
            })
        else:
            print(f"Missing coordinates for hold ID: {hold_id}")

    return coordinates


# 🛠️ MODIFIED FUNCTION: Accepts angle and limit
def get_climbs(angle, limit):
    """Fetch top climbs for a given angle sorted by ascensionist count."""
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = f"""
            SELECT c.name, cs.ascensionist_count, c.frames
            FROM climbs c
            JOIN climb_stats cs ON c.uuid = cs.climb_uuid
            WHERE cs.angle = ?
            ORDER BY cs.ascensionist_count DESC
            LIMIT {limit} 
        """
        cursor.execute(query, (angle,))
        climbs = cursor.fetchall()
    except sqlite3.Error as e:
        print("Database error:", e)
        climbs = []
    finally:
        conn.close()
    return climbs


# 🛠️ MODIFIED FUNCTION: Accepts angle and limit
def get_climbs_with_coordinates(angle=40, num_climbs=5):
    """Fetch climbs and preprocess frames into coordinates."""

    # Pass both arguments to the fetching function
    climbs = get_climbs(angle, num_climbs)
    result = {}

    for climb in climbs:
        name, ascensionist_count, frames = climb
        coordinates = parse_frames(frames)

        # Use the climb name as the key, and store the rest as a dictionary
        result[name] = {
            "ascensionist_count": ascensionist_count,
            "coordinates": coordinates
        }
    return result

# Example usage:
# climbs_data = get_climbs_with_coordinates(angle=30, num_climbs=10)
# print(f"Fetched data for {len(climbs_data)} climbs.")