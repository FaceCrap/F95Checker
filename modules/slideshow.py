from __future__ import annotations

import ctypes
import random
import threading
import time
from typing import TYPE_CHECKING
from pathlib import Path
from modules import globals

import glfw
import numpy as np
from PIL import Image
from OpenGL import GL

if TYPE_CHECKING:
    pass  # avoid circular imports at runtime

# ---------------------------------------------------------------------------
# Public helpers – call these from gui.py
# ---------------------------------------------------------------------------
def open(main_window, game=None, tab_only: bool | None = None) -> None:
    """Open (or bring to front) the slideshow window.

    Parameters
    ----------
    main_window : GLFWwindow
        The main application GLFW window (used to detect the current monitor).
    game : Game | None
        If given, the slideshow starts with image from this game first.
    tab_only : bool | None
        True  - use only games in the current active tab.
        False - use all games regardless of tab.
        None  - fall back to the slideshow_tab_only setting (default).
    """
    if globals.slideshow_window is None or not globals.slideshow_window.alive:
        globals.slideshow_window = SlideshowWindow(main_window, starting_game=game, tab_only=tab_only)
    else:
        globals.slideshow_window.bring_to_front()

def close() -> None:
    """Close the slideshow window if it is open."""
    if globals.slideshow_window is not None:
        globals.slideshow_window.destroy()
        globals.slideshow_window = None

def reset_idle_timer() -> None:
    """Reset the idle timer.  Call on every mouse-move / key event."""
    globals.slideshow_idle_timer = 0.0
    # Slideshow is only dismissed by ESC - activity does NOT close it.

def tick(main_window) -> None:
    """Must be called once per main-loop frame.

    Drives the idle timer and, if the slideshow is open, draws it.
    """
    # --- idle timer ---
    now = time.monotonic()
    dt = now - getattr(globals, "_slideshow_last_tick", now)
    globals._slideshow_last_tick = now
    globals.slideshow_idle_timer += dt

    idle_threshold = getattr(globals.settings, "slideshow_idle_seconds", 0)
    if (
        idle_threshold > 0
        and globals.slideshow_idle_timer >= idle_threshold
        and globals.slideshow_window is None
    ):
        sw = SlideshowWindow(main_window)
        sw.opened_by_idle = True
        globals.slideshow_window = sw

    # --- draw / lifecycle ---
    if globals.slideshow_window is not None:
        if globals.slideshow_window.alive:
            globals.slideshow_window.draw_frame()
        else:
            globals.slideshow_window = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRANSITION_DURATION = 0.6   # seconds for cross-fade, non-configurable
_DISPLAY_INTERVAL    = 10    # seconds per image (auto-play)
_PRELOAD_AHEAD       = 2     # number of images to pre-load ahead


# ---------------------------------------------------------------------------
# Utility: OpenGL texture helpers
# ---------------------------------------------------------------------------
def _upload_pil(img: Image.Image) -> int:
    """Upload a PIL image to an OpenGL texture and return the GLuint handle."""
    img = img.convert("RGBA")
    w, h = img.size
    data = img.tobytes()
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
        w, h, 0,
        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data,
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return int(tex)

def _delete_tex(tex_id: int) -> None:
    if tex_id:
        GL.glDeleteTextures(1, [tex_id])

def _load_image_from_path(path: Path) -> Image.Image | None:
    try:
        return Image.open(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ImageEntry – represents one displayable image (local file or URL)
# ---------------------------------------------------------------------------
class ImageEntry:
    """Lazy-loading wrapper around a single image source."""

    __slots__ = (
        "path",
        "_pil",
        "_tex_id",
        "_size",
        "_loading",
        "_lock"
    )

    def __init__(self, path: Path | None):
        self.path = path
        self._pil: Image.Image | None = None
        self._tex_id: int = 0
        self._size: tuple[int, int] = (0, 0)
        self._loading = False
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    def start_preload(self) -> None:
        """Fire a background thread to load pixel data (not GL upload)."""
        with self._lock:
            if self._pil is not None or self._loading:
                return
            self._loading = True
        t = threading.Thread(target=self._load_worker, daemon=True)
        t.start()

    def _load_worker(self) -> None:
        img = None
        if self.path and self.path.exists():
            img = _load_image_from_path(self.path)
        with self._lock:
            self._pil = img
            self._loading = False

    # -----------------------------------------------------------------------
    def ensure_texture(self) -> tuple[int, int, int]:
        """Return (tex_id, width, height).  Uploads to GL if ready.

        Must be called from the GL thread.
        """
        if self._tex_id:
            return self._tex_id, *self._size

        with self._lock:
            pil = self._pil

        if pil is None:
            # not loaded yet – start loading if not already
            self.start_preload()
            return 0, 0, 0

        img = pil.convert("RGBA")
        self._size = img.size
        self._tex_id = _upload_pil(img)
        self._pil = None  # free RAM after upload
        return self._tex_id, *self._size

    def release(self) -> None:
        """Delete GL texture (call from GL thread)."""
        _delete_tex(self._tex_id)
        self._tex_id = 0


# ---------------------------------------------------------------------------
# ImagePlaylist – ordered list + random-cycle logic
# ---------------------------------------------------------------------------
class ImagePlaylist:
    """Manages the ordered/random sequence of ImageEntry objects."""

    def __init__(self, entries: list[ImageEntry], random_order: bool = False):
        self._entries   = entries
        self._random    = random_order
        self._index     = 0
        self._cycle: list[int] = []
        self._cycle_pos = 0
        self._build_cycle()


    def _build_cycle(self) -> None:
        indices = list(range(len(self._entries)))
        if self._random:
            random.shuffle(indices)
        self._cycle = indices
        self._cycle_pos = 0

    # -----------------------------------------------------------------------
    @property
    def current(self) -> ImageEntry | None:
        if not self._cycle:
            return None
        return self._entries[self._cycle[self._cycle_pos]]


    def advance(self) -> ImageEntry | None:
        if not self._cycle:
            return None
        self._cycle_pos += 1
        if self._cycle_pos >= len(self._cycle):
            self._build_cycle()  # start a fresh cycle
        return self.current

    def go_back(self) -> ImageEntry | None:
        if not self._cycle:
            return None
        self._cycle_pos = max(0, self._cycle_pos - 1)
        return self.current

    def peek_next(self) -> ImageEntry | None:
        """Returns what advance() would return, without changing state."""
        if not self._cycle:
            return None
        next_pos = self._cycle_pos + 1
        if next_pos >= len(self._cycle):
            # Wrapping - first entry of the next cycle
            # For sequential this is always index 0; good enough to preload
            return self._entries[0] if self._entries else None
        return self._entries[self._cycle[next_pos]]

    def peek_ahead(self, n: int = 1) -> list[ImageEntry]:
        result = []
        for i in range(1, n + 1):
            pos = self._cycle_pos + i
            if pos < len(self._cycle):
                result.append(self._entries[self._cycle[pos]])
        return result

    def set_random(self, value: bool) -> None:
        if value != self._random:
            self._random = value
            self._build_cycle()

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# SlideshowWindow
# ---------------------------------------------------------------------------
class SlideshowWindow:
    """Full-screen slideshow rendered in a dedicated GLFW window."""

    def __init__(self, main_window, starting_game=None, tab_only: bool | None = None):
        self.alive           = False
        self.opened_by_idle  = False

        # Settings (read from globals.settings if available, else defaults)
        self._auto_play      = True
        self._random_order   = False
        self._tab_only       = False  # default: use all games
        self._trans_duration = _TRANSITION_DURATION
        self._interval       = _DISPLAY_INTERVAL

        self._try_read_settings()

        # Explicit argument overrides the persisted setting
        if tab_only is not None:
            self._tab_only = tab_only

        # Build image list
        self._entries = self._collect_images(starting_game)
        if not self._entries:
            return # nothing to show

        self._playlist = ImagePlaylist(self._entries, self._random_order)

        # Transition state
        self._current_tex    : int = 0
        self._current_size   : tuple[int, int] = (0, 0)
        self._next_tex       : int = 0
        self._next_size      : tuple[int, int] = (0, 0)
        self._trans_alpha    : float = 0.0   # 0 = fully on current, 1 = fully on next

        self._in_transition  = False
        self._trans_start    = 0.0

        self._last_advance   = time.monotonic()
        self._paused         = not self._auto_play

        # Pending GL uploads (queued from preload thread -> uploaded on GL thread)
        self._pending_lock   = threading.Lock()
        self._pending_upload : list[ImageEntry] = []
        self._current_entry  : ImageEntry | None = None
        self._next_entry     : ImageEntry | None = None

        # Create the GLFW window
        monitor = self._detect_monitor(main_window)
        self._window = self._create_window(monitor)
        if self._window is None:
            return

        self.alive = True
        self._install_callbacks()

        # Kick off preloading
        for entry in self._playlist.peek_ahead(_PRELOAD_AHEAD):
            entry.start_preload()
        if self._playlist.current:
            self._playlist.current.start_preload()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def bring_to_front(self) -> None:
        if self._window:
            glfw.focus_window(self._window)

    def destroy(self) -> None:
        self.alive = False
        self._release_textures()
        if self._window:
            glfw.destroy_window(self._window)
            self._window = None

    # -----------------------------------------------------------------------
    # Settings helper
    # -----------------------------------------------------------------------
    def _try_read_settings(self) -> None:
        try:
            s = globals.settings
            self._auto_play      = getattr(s, "slideshow_auto_play",      self._auto_play)
            self._random_order   = getattr(s, "slideshow_random_order",   self._random_order)
            self._tab_only       = getattr(s, "slideshow_tab_only",       self._tab_only)
            self._trans_duration = getattr(s, "slideshow_transition_dur", self._trans_duration)
            self._interval       = getattr(s, "slideshow_interval",       self._interval)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Image collection
    # -----------------------------------------------------------------------
    def _collect_images(self, starting_game=None) -> list[ImageEntry]:
        """Gather ImageEntry objects for the chosen game scope (starting game first).

        When self._tab_only is True, only games whose .tab matches
        globals.current_tab are included.  Otherwise all games are used.
        """
        # Determine the active tab filter (None means the Default(New) tab)
        active_tab = None
        filter_by_tab = False
        if self._tab_only:
            try:
                active_tab = globals.gui.current_tab
                tab_games  = globals.gui.show_games_ids[active_tab]
                games_list = list()
                for id in tab_games:
                    games_list.append(globals.games[id])
                filter_by_tab = True
            except Exception:
                pass  # if current_tab is unavailable, fall back to all games

        if not filter_by_tab:
            try:
                games_list = list(globals.games.values())
            except Exception:
                return [], entry_to_game

        entries: list[ImageEntry] = []

        def _entry_for(game) -> ImageEntry | None:
            # Local image: stored as  <images_dir>/<thread_id>.<ext>
            img_path: Path | None = None
            try:
                img_dir = globals.data_path / "images"
                candidates = list(img_dir.glob(f"{game.id}.*"))
                if candidates:
                    img_path = candidates[0]
            except Exception:
                pass

            if img_path is None:
                return None
            return ImageEntry(img_path)

        def _in_scope(game) -> bool:
            """Return True if this game should be included given the current scope."""
            if not filter_by_tab:
                return True
            game_tab = getattr(game, "tab", None)
            # Both None means "Default(New) tab" - treat as a match
            if game_tab is None and active_tab is None:
                return True
            return game_tab is active_tab

        if starting_game is not None:
            e = _entry_for(starting_game)
            if e:
                entries.append(e)

        for game in games_list:
            if starting_game is not None and game is starting_game:
                continue
            if not _in_scope(game):
                continue
            e = _entry_for(game)
            if e:
                entries.append(e)

        return entries

    # -----------------------------------------------------------------------
    # Monitor detection
    # -----------------------------------------------------------------------
    def _detect_monitor(self, main_window) -> object:
        """Return the glfw.Monitor that contains the main window."""
        try:
            wx, wy = glfw.get_window_pos(main_window)
            ww, wh = glfw.get_window_size(main_window)
            cx = wx + ww // 2
            cy = wy + wh // 2

            best       = None
            best_area  = -1

            for mon in glfw.get_monitors():
                mx, my = glfw.get_monitor_pos(mon)
                vm     = glfw.get_video_mode(mon)
                mw, mh = vm.size.width, vm.size.height

                # Clamp main-window centre to this monitor rect
                ox = max(mx, min(cx, mx + mw)) - cx
                oy = max(my, min(cy, my + mh)) - cy
                if ox == 0 and oy == 0:
                    return mon  # centre is inside -> perfect match

                # Fallback: choose monitor with most overlap area
                overlap_w = max(0, min(wx + ww, mx + mw) - max(wx, mx))
                overlap_h = max(0, min(wy + wh, my + mh) - max(wy, my))
                area = overlap_w * overlap_h
                if area > best_area:
                    best_area = area
                    best = mon

            return best or glfw.get_primary_monitor()
        except Exception:
            return glfw.get_primary_monitor()

    # -----------------------------------------------------------------------
    # Window creation
    # -----------------------------------------------------------------------
    def _create_window(self, monitor) -> object | None:
        try:
            vm = glfw.get_video_mode(monitor)
            w, h = vm.size.width, vm.size.height
            mx, my = glfw.get_monitor_pos(monitor)

            glfw.window_hint(glfw.DECORATED,             glfw.FALSE)
            glfw.window_hint(glfw.FOCUSED,               glfw.TRUE)
            # AUTO_ICONIFY only meaningful for true fullscreen; keep FALSE so
            # the window doesn't minimize when focus moves to another monitor.
            glfw.window_hint(glfw.AUTO_ICONIFY,          glfw.FALSE)
            glfw.window_hint(glfw.FLOATING,              glfw.TRUE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE,        glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

            # Share GL context with main window so textures are usable there too
            try:
                share_ctx = globals.gui.window
            except Exception:
                share_ctx = None

            # A true fullscreen window blanks every other monitor because the GPU
            # performs a mode-switch; a borderless window sized/positioned to fill
            # the monitor looks identical but should leave other outputs untouched.
            win = glfw.create_window(w, h, "F95Checker Slideshow", monitor, share_ctx)
            if win is None:
                return None

            # Move to the target monitor's top-left corner before making current
            # so the compositor never briefly shows it on the wrong display.
            glfw.set_window_pos(win, mx, my)

            glfw.make_context_current(win)
            glfw.swap_interval(1)

            # Pre-clear both framebuffers so the OS compositor never sees garbage pixels
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            glfw.swap_buffers(win)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            glfw.swap_buffers(win)

            return win
        except Exception as exc:
            print(f"[slideshow] Failed to create window: {exc}")
            return None
        finally:
            # Restore main-window context
            try:
                glfw.make_context_current(globals.gui.window)
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # GLFW callbacks
    # -----------------------------------------------------------------------
    def _install_callbacks(self) -> None:
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor_move)
        glfw.set_window_close_callback(self._window, self._on_close)
        glfw.set_scroll_callback(self._window, self._on_scroll)

    def _on_close(self, win) -> None:
        # Suppress OS-level close (alt-F4, etc.).
        # Only ESC terminates the slideshow.
        glfw.set_window_should_close(win, glfw.FALSE)

# For now we prevent any kind of manual control, maybe later if desired
    def _on_key(self, win, key, scancode, action, mods) -> None:
        reset_idle_timer()
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            self.destroy()
        """
        elif key == glfw.KEY_RIGHT:
            self._manual_next()
        elif key == glfw.KEY_LEFT:
            self._manual_prev()
        elif key == glfw.KEY_SPACE:
            self._paused = not self._paused
        elif key == glfw.KEY_R:
            self._random_order = not self._random_order
            self._playlist.set_random(self._random_order)
        """

    def _on_mouse_button(self, win, button, action, mods) -> None:
        reset_idle_timer()
        if action != glfw.PRESS:
            return
        """
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._manual_next()
        """
        # Right-click is intentionally ignored - only ESC closes the slideshow.

    def _on_cursor_move(self, win, x, y) -> None:
        reset_idle_timer()

    def _on_scroll(self, win, xoff, yoff) -> None:
        reset_idle_timer()
        """
        if yoff > 0:
            self._manual_next()
        elif yoff < 0:
            self._manual_prev()
        """

    # -----------------------------------------------------------------------
    # Playback control
    # -----------------------------------------------------------------------
    """
    def _manual_next(self) -> None:
        self._start_transition(forward=True)
        self._last_advance = time.monotonic()

    def _manual_prev(self) -> None:
        self._start_transition(forward=False)
        self._last_advance = time.monotonic()
    """
    def _start_transition(self, forward: bool = True) -> None:
        if self._in_transition:
            # Commit current transition immediately, then start new one
            self._commit_transition()

        if forward:
            candidate = self._playlist.peek_next()
            if candidate is not None:
                tex_id, _, _ = candidate.ensure_texture()
                if tex_id == 0:
                    candidate.start_preload()
                    return False # not ready - don't advance, don't reset timer
            next_entry = self._playlist.advance()
        else:
            next_entry = self._playlist.go_back()

        if next_entry is None:
            return False

        # Upload next texture if ready, otherwise wait
        tex_id, w, h = next_entry.ensure_texture()
        if tex_id == 0:
            # Not yet loaded – preload and skip transition for now
            return False

        self._next_tex      = tex_id
        self._next_size     = (w, h)
        self._next_entry    = next_entry
        self._in_transition = True
        self._trans_start   = time.monotonic()
        self._trans_alpha   = 0.0

        # Pre-load further ahead
        for e in self._playlist.peek_ahead(_PRELOAD_AHEAD):
            e.start_preload()

        return True

    def _commit_transition(self) -> None:
        """Snap to the next image immediately."""
        if self._current_entry is not None and self._current_entry is not self._next_entry:
            # takes care of setting tex_id to 0 and deleting the GL texture
            # so that whenever the last image is drawn, the cycle starts again
            self._current_entry.release()
        self._current_tex    = self._next_tex
        self._current_size   = self._next_size
        self._current_entry  = self._next_entry
        self._next_tex      = 0
        self._next_size     = (0, 0)
        self._in_transition = False
        self._trans_alpha   = 0.0

    # -----------------------------------------------------------------------
    # Per-frame draw
    # -----------------------------------------------------------------------
    def draw_frame(self) -> None:

        if not self.alive or self._window is None:
            return

        # Save + switch GL context
        try:
            prev_ctx = globals.gui.window
        except Exception:
            prev_ctx = None

        glfw.make_context_current(self._window)
        glfw.poll_events()

        now   = time.monotonic()
        sw, sh = glfw.get_framebuffer_size(self._window)

        GL.glViewport(0, 0, sw, sh)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # Ensure current image is loaded
        if self._current_tex == 0 and self._playlist.current:
            tex_id, w, h = self._playlist.current.ensure_texture()
            if tex_id:
                self._current_tex  = tex_id
                self._current_size = (w, h)
                self._current_entry = self._playlist.current

        # Auto-advance
        if (
            not self._paused
            and not self._in_transition
            and self._current_tex
            and (now - self._last_advance) >= self._interval
        ):
            if self._start_transition(forward=True):
                # only reset on success
                self._last_advance = now

        # Update transition alpha
        if self._in_transition:
            elapsed = now - self._trans_start
            self._trans_alpha = min(elapsed / max(self._trans_duration, 0.001), 1.0)
            if self._trans_alpha >= 1.0:
                self._commit_transition()

        # Draw image(s) using immediate-mode GL quads via ImGui draw list approach.
        # We use raw OpenGL so we don't need a separate ImGui context for this window.
        self._gl_draw_image(self._current_tex, self._current_size, sw, sh, alpha=1.0)

        if self._in_transition and self._next_tex:
            self._gl_draw_image(
                self._next_tex, self._next_size, sw, sh,
                alpha=self._trans_alpha,
            )

        # Minimal HUD via ImGui if we share context
        # - skip for now

        glfw.swap_buffers(self._window)

        # Restore main context
        if prev_ctx:
            glfw.make_context_current(prev_ctx)

    # -----------------------------------------------------------------------
    # OpenGL image drawing (without ImGui – raw quad)
    # -----------------------------------------------------------------------
    _shader_prog: int = 0   # class-level cached shader

    def _gl_draw_image(
        self,
        tex_id:   int,
        tex_size: tuple[int, int],
        vp_w:     int,
        vp_h:     int,
        alpha:    float = 1.0,
    ) -> None:
        """Render a full-viewport, aspect-correct, alpha-blended quad."""
        if tex_id == 0 or vp_w == 0 or vp_h == 0:
            return

        if not SlideshowWindow._shader_prog:
            SlideshowWindow._shader_prog = self._build_shader()

        prog = SlideshowWindow._shader_prog
        if not prog:
            return

        img_w, img_h = tex_size
        if img_w == 0 or img_h == 0:
            return

        # Compute aspect-correct rect (contain / letterbox)
        vp_aspect  = vp_w  / vp_h
        img_aspect = img_w / img_h

        if img_aspect >= vp_aspect:
            # Image wider than viewport -> fit width
            draw_w = vp_w
            draw_h = int(vp_w / img_aspect)
        else:
            # Image taller than viewport -> fit height
            draw_h = vp_h
            draw_w = int(vp_h * img_aspect)

        # Normalised device coords for the rect
        x0 = (vp_w - draw_w) / 2 / vp_w * 2 - 1
        y0 = (vp_h - draw_h) / 2 / vp_h * 2 - 1
        x1 = x0 + draw_w / vp_w * 2
        y1 = y0 + draw_h / vp_h * 2

        # Vertices: x, y, u, v
        verts = np.array([
            x0, y0, 0.0, 1.0,
            x1, y0, 1.0, 1.0,
            x1, y1, 1.0, 0.0,
            x0, y1, 0.0, 0.0,
        ], dtype=np.float32)

        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STREAM_DRAW)

        stride = 4 * verts.itemsize
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(2 * verts.itemsize))
        GL.glEnableVertexAttribArray(1)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glUseProgram(prog)
        loc = GL.glGetUniformLocation(prog, "u_alpha")
        GL.glUniform1f(loc, alpha)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        loc_tex = GL.glGetUniformLocation(prog, "u_texture")
        GL.glUniform1i(loc_tex, 0)

        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

        GL.glBindVertexArray(0)
        GL.glDeleteVertexArrays(1, [vao])
        GL.glDeleteBuffers(1, [vbo])
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_BLEND)

    @staticmethod
    def _build_shader() -> int:
        vert_src = """
#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
"""
        frag_src = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
uniform float u_alpha;
void main() {
    vec4 col = texture(u_texture, v_uv);
    frag_color = vec4(col.rgb, col.a * u_alpha);
}
"""
        def _compile(src, kind):
            sh = GL.glCreateShader(kind)
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
                err = GL.glGetShaderInfoLog(sh).decode()
                print(f"[slideshow] Shader compile error: {err}")
                GL.glDeleteShader(sh)
                return 0
            return sh

        vs = _compile(vert_src, GL.GL_VERTEX_SHADER)
        fs = _compile(frag_src, GL.GL_FRAGMENT_SHADER)
        if not vs or not fs:
            return 0

        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            err = GL.glGetProgramInfoLog(prog).decode()
            print(f"[slideshow] Shader link error: {err}")
            GL.glDeleteProgram(prog)
            return 0

        return prog

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    def _release_textures(self) -> None:
        for e in self._entries:
            e.release()
        _delete_tex(self._current_tex)
        self._current_tex = 0
        _delete_tex(self._next_tex)
        self._next_tex = 0


# ---------------------------------------------------------------------------
# Variables needing to be added to settings/db schema
# ---------------------------------------------------------------------------
# slideshow_auto_play       bool    default True
# slideshow_random_order    bool    default False
# slideshow_tab_only        bool    default False   (True = active tab only, False = all games)
# slideshow_transition_dur  float   default 0.6     (seconds, currently hardcoded)
# slideshow_interval        int     default 10      (seconds)
# slideshow_idle_seconds    int     default 0       (0 = disabled)