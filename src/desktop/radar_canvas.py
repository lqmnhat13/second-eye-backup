"""
2D Top-Down Spatial Radar Widget for Second Eye Desktop.
High-visibility, high-contrast radar display with concentric distance rings,
directional sectors (Left, Center, Right), radar sweep line, and glowing obstacle blips.
"""

import math
import tkinter as tk
from typing import List, Optional
from src.core.distance_estimator import DetectedObject

class RadarWidget(tk.Canvas):
    def __init__(
        self,
        master,
        width: int = 400,
        height: int = 230,
        max_range_meters: float = 4.0,
        bg: str = "#0b1120",
        **kwargs
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            bg=bg,
            highlightthickness=1,
            highlightbackground="#1e293b",
            **kwargs
        )
        self.w = width
        self.h = height
        self.max_range = max_range_meters
        self.radar_angle = 0.0
        self.detected_objects: List[DetectedObject] = []
        self._sweep_dir = 1

    def update_objects(self, objects: List[DetectedObject]):
        """Update the list of detected objects to plot on the radar."""
        self.detected_objects = objects
        self.draw_radar()

    def advance_sweep(self):
        """Advance the sweep animation line and redraw."""
        self.radar_angle += 0.05 * self._sweep_dir
        if self.radar_angle > 0.85:
            self._sweep_dir = -1
        elif self.radar_angle < -0.85:
            self._sweep_dir = 1
        self.draw_radar()

    def draw_radar(self):
        """Full redraw of the radar grid, sectors, sweep, and obstacle blips."""
        self.delete("all")

        cx = self.w // 2
        cy = self.h - 22
        max_rad = self.h - 38
        scale = max_rad / self.max_range

        # 1. Background grid concentric distance arcs (1m, 2m, 3m, 4m)
        distance_steps = [1.0, 2.0, 3.0, 4.0]
        for r in distance_steps:
            rad_px = r * scale
            if rad_px > max_rad:
                continue

            # Upper semicircle arc with vibrant blue/cyan
            outline_color = "#0284c7" if r == 2.0 or r == 4.0 else "#0369a1"
            line_w = 2 if r == 2.0 or r == 4.0 else 1

            self.create_arc(
                cx - rad_px, cy - rad_px,
                cx + rad_px, cy + rad_px,
                start=0, extent=180,
                style=tk.ARC,
                outline=outline_color,
                width=line_w
            )

            # Distance pill badge
            badge_text = f" {int(r)}m "
            self.create_rectangle(
                cx - 14, cy - rad_px - 7,
                cx + 14, cy - rad_px + 7,
                fill="#0f172a", outline="#1e293b", width=1
            )
            self.create_text(
                cx, cy - rad_px,
                text=f"{int(r)}m",
                fill="#93c5fd",
                font=("Arial", 8, "bold")
            )

        # 2. Sector dividing rays (Left, Center, Right)
        angles_deg = [30, 60, 90, 120, 150]
        for deg in angles_deg:
            rad = math.radians(deg)
            x_end = cx + max_rad * math.cos(rad)
            y_end = cy - max_rad * math.sin(rad)
            is_main = deg in [60, 120]
            color = "#0284c7" if is_main else "#1e293b"
            dash = () if is_main else (2, 4)
            self.create_line(cx, cy, x_end, y_end, fill=color, dash=dash, width=1)

        # Sector Text Tags at Top
        self.create_text(cx - int(max_rad * 0.72), cy - max_rad + 6, text="TRÁI", fill="#94a3b8", font=("Arial", 9, "bold"))
        self.create_text(cx, cy - max_rad - 8, text="PHÍA TRƯỚC (GIỮA)", fill="#38bdf8", font=("Arial", 9, "bold"))
        self.create_text(cx + int(max_rad * 0.72), cy - max_rad + 6, text="PHẢI", fill="#94a3b8", font=("Arial", 9, "bold"))

        # 3. Dynamic radar sweep line
        sweep_rad = math.pi / 2.0 + self.radar_angle
        sw_x = cx + max_rad * math.cos(sweep_rad)
        sw_y = cy - max_rad * math.sin(sweep_rad)
        self.create_line(cx, cy, sw_x, sw_y, fill="#38bdf8", width=2)

        # Sweep trail
        trail_rad = sweep_rad - 0.07 * self._sweep_dir
        tr_x = cx + max_rad * math.cos(trail_rad)
        tr_y = cy - max_rad * math.sin(trail_rad)
        self.create_line(cx, cy, tr_x, tr_y, fill="#0284c7", width=1)

        # 4. Origin (User/Camera Position)
        self.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill="#38bdf8", outline="#ffffff", width=2)
        self.create_text(cx, cy + 13, text="Camera (Vị trí bạn)", fill="#94a3b8", font=("Arial", 8, "bold"))

        # 5. Plot detected obstacles with high contrast glowing dots & labels
        for obj in self.detected_objects:
            if obj.distance > self.max_range * 1.25:
                continue

            x_lateral, z_depth = obj.coord_3d
            z_depth = max(0.2, z_depth)

            # Map to canvas coordinates
            px = cx + (x_lateral / self.max_range) * max_rad
            py = cy - (z_depth / self.max_range) * max_rad

            # Clamp inside visible radar bounds
            px = max(16, min(self.w - 16, px))
            py = max(18, min(cy - 6, py))

            # Color and size by risk
            if obj.risk_level == "DANGER":
                blip_color = "#ef4444"
                ring_color = "#f87171"
                r_dot = 8
            elif obj.risk_level == "WARNING":
                blip_color = "#f59e0b"
                ring_color = "#fbbf24"
                r_dot = 7
            else:
                blip_color = "#10b981"
                ring_color = "#34d399"
                r_dot = 6

            # Outer glowing ring
            self.create_oval(px - r_dot - 5, py - r_dot - 5, px + r_dot + 5, py + r_dot + 5, outline=ring_color, width=1.5)
            # Solid center dot
            self.create_oval(px - r_dot, py - r_dot, px + r_dot, py + r_dot, fill=blip_color, outline="#ffffff", width=2)

            # High-visibility Text Pill Badge
            label_text = f"{obj.name_vi.capitalize()} {obj.distance:.1f}m"
            anchor = "w" if px < cx else "e"
            offset_x = 14 if anchor == "w" else -14

            # Pill backdrop for label
            text_x = px + offset_x
            text_y = py - 4
            self.create_text(
                text_x, text_y,
                text=label_text,
                fill="#f8fafc",
                font=("Arial", 9, "bold"),
                anchor=anchor
            )
