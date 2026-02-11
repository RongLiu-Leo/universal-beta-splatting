import viser
from nerfview import Viewer, RenderTabState
from typing import Literal
from typing import Callable, Tuple
import threading
import time


class BetaRenderTabState(RenderTabState):
    # non-controlable parameters
    total_count_number: int = 0
    rendered_count_number: int = 0

    # controlable parameters
    timestamp: float = 0.0
    near_plane: float = 1e-3
    far_plane: float = 1e3
    radius_clip: float = 0.0
    b_xyz: Tuple[int, int] = (0, 100)
    b_view: Tuple[int, int] = (0, 100)
    b_time: Tuple[int, int] = (0, 100)
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal["RGB", "Alpha", "Depth", "Normal"] = "RGB"


class BetaViewer(Viewer):
    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        input_dim: int = 6,
        mode: Literal["rendering", "training"] = "rendering",
        share_url: bool = False,
    ):
        self.input_dim = input_dim
        super().__init__(server, render_fn, mode=mode)
        server.gui.set_panel_label("Beta Splatting Viewer")
        self._playing_time = False
        self._play_thread = None
        self._play_thread_lock = threading.Lock()
        if share_url:
            server.request_share_url()

    def _init_rendering_tab(self):
        self.render_tab_state = BetaRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        with self._rendering_folder:

            if self.input_dim == 7:
                self.gui_slider_time = self.server.gui.add_slider(
                    "Time",
                    min=0.0,
                    max=1.0,
                    step=0.0001,
                    initial_value=self.render_tab_state.timestamp,
                )

                @self.gui_slider_time.on_update
                def _(_) -> None:
                    self.render_tab_state.timestamp = self.gui_slider_time.value
                    self.rerender(_)

                self.gui_play_speed = self.server.gui.add_slider(
                    "Play speed (×)",
                    min=0.1,
                    max=4.0,
                    step=0.1,
                    initial_value=1.0,
                    hint="Playback speed multiplier for the Time slider.",
                )

                self.gui_btn_toggle = self.server.gui.add_button("▶ Play")

                @self.gui_btn_toggle.on_click
                def _toggle(_) -> None:
                    with self._play_thread_lock:
                        self._playing_time = not self._playing_time
                        self._set_play_button_ui(self._playing_time)
                        if self._playing_time:
                            # start a fresh loop thread
                            self._play_thread = threading.Thread(
                                target=self._autoplay_time_loop, daemon=True
                            )
                            self._play_thread.start()

            with self.server.gui.add_folder("Geometry Dependency Control"):
                self.gui_multi_slider_xyz = self.server.gui.add_multi_slider(
                    "Geo quantile",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=self.render_tab_state.b_xyz,
                )

                @self.gui_multi_slider_xyz.on_update
                def _(_) -> None:
                    self.render_tab_state.b_xyz = self.gui_multi_slider_xyz.value
                    self.rerender(_)

            with self.server.gui.add_folder("View Dependency Control"):
                self.gui_multi_slider_view = self.server.gui.add_multi_slider(
                    "View quantile",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=self.render_tab_state.b_view,
                )

                @self.gui_multi_slider_view.on_update
                def _(_) -> None:
                    self.render_tab_state.b_view = self.gui_multi_slider_view.value
                    self.rerender(_)

            if self.input_dim == 7:
                with self.server.gui.add_folder("Time Dependency Control"):
                    self.gui_multi_slider_time = self.server.gui.add_multi_slider(
                        "Time quantile",
                        min=0,
                        max=100,
                        step=1,
                        initial_value=self.render_tab_state.b_time,
                    )

                    @self.gui_multi_slider_time.on_update
                    def _(_) -> None:
                        self.render_tab_state.b_time = self.gui_multi_slider_time.value
                        self.rerender(_)

            with self.server.gui.add_folder("Render Mode"):
                self.render_mode_dropdown = self.server.gui.add_dropdown(
                    "Mode",
                    ["RGB", "Alpha", "Depth", "Normal"],
                    initial_value=self.render_tab_state.render_mode,
                )

                @self.render_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.render_mode = self.render_mode_dropdown.value
                    self.rerender(_)

                self.total_count_number = self.server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_count_number,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                self.rendered_count_number = self.server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_count_number,
                    disabled=True,
                    hint="Number of splats rendered.",
                )
                self.radius_clip_slider = self.server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @self.radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = self.radius_clip_slider.value
                    self.rerender(_)

                self.near_far_plane_vec2 = self.server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @self.near_far_plane_vec2.on_update
                def _(_) -> None:
                    (
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ) = self.near_far_plane_vec2.value
                    self.rerender(_)

                self.backgrounds_slider = self.server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @self.backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = self.backgrounds_slider.value
                    self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "timestamp": self.gui_slider_time if self.input_dim == 7 else 0.0,
                "b_xyz": self.gui_multi_slider_xyz,
                "b_view": self.gui_multi_slider_view,
                "b_time": self.gui_multi_slider_time
                if self.input_dim == 7
                else (0, 100),
                "total_count_number": self.total_count_number,
                "rendered_count_number": self.rendered_count_number,
                "near_far_plane_vec2": self.near_far_plane_vec2,
                "radius_clip_slider": self.radius_clip_slider,
                "rener_mode_dropdown": self.render_mode_dropdown,
                "backgrounds_slider": self.backgrounds_slider,
                "play_speed": getattr(self, "gui_play_speed", None),
                "btn_toggle": getattr(self, "gui_btn_toggle", None),
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles[
            "total_count_number"
        ].value = self.render_tab_state.total_count_number
        self._rendering_tab_handles[
            "rendered_count_number"
        ].value = self.render_tab_state.rendered_count_number

    def _set_play_button_ui(self, playing: bool):
        label = "⏸ Pause" if playing else "▶ Play"
        btn = getattr(self, "gui_btn_toggle", None)
        if not btn:
            return
        # try common attribute/method names; ignore if not supported
        for attr in ("label", "name", "title", "text"):
            if hasattr(btn, attr):
                try:
                    setattr(btn, attr, label)
                    return
                except Exception:
                    pass
        if hasattr(btn, "set_label"):
            try:
                btn.set_label(label)
                return
            except Exception:
                pass

    def _autoplay_time_loop(self):
        dt = 0.01
        while True:
            with self._play_thread_lock:
                if not self._playing_time:
                    # ensure UI shows Play when loop exits (just in case)
                    self._set_play_button_ui(False)
                    break

            speed = (
                float(getattr(self, "gui_play_speed", 1.0).value)
                if hasattr(self, "gui_play_speed")
                else 1.0
            )
            curr_t = float(self.gui_slider_time.value)
            new_t = (curr_t + speed * dt) % 1.0

            self.gui_slider_time.value = new_t
            self.render_tab_state.timestamp = new_t
            self.rerender(None)
            time.sleep(dt)
