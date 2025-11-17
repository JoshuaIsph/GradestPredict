import tkinter as tk
from PIL import Image, ImageTk
import os

from database.data import get_climbs_with_coordinates
from preprocessing.moveOrder import process_all_climbs

IMAGE_FILENAME = "Kilterboard.png"
IMAGE_PATH = os.path.join("../utils", IMAGE_FILENAME)

def draw_circle(canvas, x, y, color, canvas_height=400):
    radius = 10
    flipped_y = canvas_height - y
    canvas.create_oval(x - radius, flipped_y - radius, x + radius, flipped_y + radius, outline=color, width=2)

def draw_line(canvas, x1, y1, x2, y2, color="gray", width=1, canvas_height=400):
    y1_flipped = canvas_height - y1
    y2_flipped = canvas_height - y2
    canvas.create_line(x1, y1_flipped, x2, y2_flipped, fill=color, width=width)

class ClimbWindow(tk.Tk):
    def __init__(self, climbs):
        super().__init__()
        self.title("Climb List")
        self.geometry("1000x650")
        self.eval('tk::PlaceWindow . center')

        self.climbs = climbs
        self.current_climb_name = None

        self.scale_x = tk.DoubleVar(value=2.8)
        self.scale_y = tk.DoubleVar(value=2.55)
        self.offset_x = tk.DoubleVar(value=-2)
        self.offset_y = tk.DoubleVar(value=2)

        self.columnconfigure(0, weight=1, minsize=250)
        self.columnconfigure(1, weight=3, minsize=400)
        self.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(self, font=("Arial", 12))
        for climb_name in climbs.keys():
            self.listbox.insert(tk.END, climb_name)
        self.listbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.listbox.bind("<<ListboxSelect>>", self.on_climb_select)

        right_frame = tk.Frame(self)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(right_frame, width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        controls = tk.Frame(right_frame)
        controls.grid(row=1, column=0, sticky="ew", pady=10)
        controls.columnconfigure((0, 1, 2, 3), weight=1)

        tk.Scale(
            controls,
            from_=0.5, to=4.0, resolution=0.1, orient="horizontal",
            label="X Scale", variable=self.scale_x,
            command=lambda e: self.redraw_climb()
        ).grid(row=0, column=0, sticky="ew", padx=5)

        tk.Scale(
            controls,
            from_=0.5, to=4.0, resolution=0.01, orient="horizontal",
            label="Y Scale", variable=self.scale_y,
            command=lambda e: self.redraw_climb()
        ).grid(row=0, column=1, sticky="ew", padx=5)

        tk.Scale(
            controls,
            from_=-200, to=200, resolution=1, orient="horizontal",
            label="X Offset", variable=self.offset_x,
            command=lambda e: self.redraw_climb()
        ).grid(row=0, column=2, sticky="ew", padx=5)

        tk.Scale(
            controls,
            from_=-200, to=200, resolution=1, orient="horizontal",
            label="Y Offset", variable=self.offset_y,
            command=lambda e: self.redraw_climb()
        ).grid(row=0, column=3, sticky="ew", padx=5)

        self.load_image()

    def load_image(self):
        self.canvas.delete("all")
        try:
            img = Image.open(IMAGE_PATH)
            img = img.resize((400, 400))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        except Exception as e:
            self.canvas.create_text(200, 200, text=f"Image not found:\n{e}", fill="black", font=("Arial", 14))

    def on_climb_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        index = selection[0]
        self.current_climb_name = self.listbox.get(index)
        self.redraw_climb()

    def redraw_climb(self):
        if not self.current_climb_name:
            return

        self.load_image()
        climb_info = self.climbs[self.current_climb_name]
        coordinates = climb_info["coordinates"]

        scale_x = self.scale_x.get()
        scale_y = self.scale_y.get()
        offset_x = self.offset_x.get()
        offset_y = self.offset_y.get()

        # Draw lines between all holds
        n = len(coordinates)
        for i in range(n):
            x1 = (coordinates[i]["x"] * scale_x) + offset_x
            y1 = (coordinates[i]["y"] * scale_y) + offset_y
            for j in range(i + 1, n):
                x2 = (coordinates[j]["x"] * scale_x) + offset_x
                y2 = (coordinates[j]["y"] * scale_y) + offset_y
                draw_line(self.canvas, x1, y1, x2, y2)

        # Draw holds and move labels
        for coord in coordinates:
            x = (coord["x"] * scale_x) + offset_x
            y = (coord["y"] * scale_y) + offset_y
            draw_circle(self.canvas, x, y, coord["color_name"])
            if coord.get("move") is not None:
                self.canvas.create_text(x + 15, 400 - y, text=str(coord["move"]), fill="red", font=("Arial", 18))

if __name__ == "__main__":
    climbs = get_climbs_with_coordinates()
    climbs = process_all_climbs(climbs)
    print(climbs)
    if climbs:
        app = ClimbWindow(climbs)
        app.mainloop()
    else:
        print("No climbs found.")