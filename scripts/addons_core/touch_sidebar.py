# touch_sidebar.py — Touch-friendly sidebar toggle for Blender on Android
# (and desktop) v1.0.0
# NR1 (2026-11-13): the sidebar (N-panel) open control on the phone is a
# ~2px-wide action-zone tab at the screen edge (region_azone_tab_plus,
# area.cc:1275) — untappable by finger, so the user could only open the
# sidebar via KEYBOARD + N. The direct fix (widening that tab) is native
# C++ (round N1) — this add-on is the NO-REBUILD workaround:
#   1. A big "☰ Sidebar" button prepended to the 3D View header (first item,
#      wide row) that toggles the sidebar — no keyboard, no edge tap.
#   2. The sidebar opens by default: a load_post handler + a one-shot timer
#      after register() set show_region_ui = True in every 3D View, so the
#      toggle is rarely needed at all. (Only fires on file load / startup,
#      never while the user is working — it will not fight a deliberate close.)
#
# Blender's Python UI exposes no per-widget font size, so "big" here means
# a wide header button (the header itself is ~50 px tall on the phone —
# already close to the 48dp touch minimum in height; the old tab failed on
# WIDTH, ~2 px). The button text uses U+2630 (☰); Blender's UI font is
# DejaVu-based, which carries that glyph — the phone screenshot in the
# test card will confirm it renders.

bl_info = {
    "name": "Touch Sidebar",
    "author": "Muse (for anony)",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Header",
    "description": "Big touch-friendly Sidebar toggle in the 3D View header + sidebar open by default",
    "category": "System",
}

import bpy
from bpy.types import Operator

ADDON_NAME = __package__ or "touch_sidebar"


def _toggle_space(space):
    """Flip show_region_ui on a SpaceView3D. Returns the new state, or None
    if the space is not a 3D View. Pure function (no bpy.context) so the
    desktop test can exercise it with a fake space object."""
    if space is None or getattr(space, "type", None) != "VIEW_3D":
        return None
    space.show_region_ui = not bool(space.show_region_ui)
    return bool(space.show_region_ui)


def _open_sidebars():
    """Open the sidebar in every 3D View of every window. Returns None (a
    one-shot timer callback must not repeat). Never raises: headless runs
    have no windows at all, and startup must never fail."""
    try:
        wm = bpy.context.window_manager
        for window in wm.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    try:
                        area.spaces.active.show_region_ui = True
                    except Exception:
                        pass
    except Exception:
        pass
    return None


class TOUCH_OT_toggle_sidebar(Operator):
    bl_idname = "touch_sidebar.toggle"
    bl_label = "Toggle Sidebar"
    bl_description = "Show or hide the sidebar (N-panel) — big touch-friendly toggle, no keyboard needed"

    def execute(self, context):
        state = _toggle_space(context.space_data)
        if state is None:
            self.report({"WARNING"}, "No 3D View here")
            return {"CANCELLED"}
        return {"FINISHED"}


def draw_sidebar_button(self, context):
    """Header draw callback (prepended, so it is the first item in the
    3D View header). Wide row: width is what the old ~2px tab lacked."""
    row = self.layout.row(align=True)
    row.scale_x = 2.0
    row.operator("touch_sidebar.toggle", text="☰ Sidebar")


@bpy.app.handlers.persistent
def _load_post_open_sidebar(_dummy):
    _open_sidebars()


def register():
    bpy.utils.register_class(TOUCH_OT_toggle_sidebar)
    bpy.types.VIEW3D_HT_header.prepend(draw_sidebar_button)
    bpy.app.handlers.load_post.append(_load_post_open_sidebar)
    # One-shot timer for the already-loaded startup file (load_post may not
    # fire for it). Runs once, ~1s after enable, then stops (returns None).
    if not bpy.app.timers.is_registered(_open_sidebars):
        bpy.app.timers.register(_open_sidebars, first_interval=1.0)


def unregister():
    try:
        if bpy.app.timers.is_registered(_open_sidebars):
            bpy.app.timers.unregister(_open_sidebars)
    except Exception:
        pass
    try:
        bpy.app.handlers.load_post.remove(_load_post_open_sidebar)
    except ValueError:
        pass
    try:
        bpy.types.VIEW3D_HT_header.remove(draw_sidebar_button)
    except Exception:
        pass
    bpy.utils.unregister_class(TOUCH_OT_toggle_sidebar)


if __name__ == "__main__":
    register()
