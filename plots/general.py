""" General plotting functions. """
import inspect
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from typing import Sequence

import distinctipy
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import key_press_handler
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from numpy import linalg
from numpy import ndarray

colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (0, 0, 0), (0, 0.75, 0.75), (0.75, 0, 0.75), (0.75, 0.75, 0)]
colors += distinctipy.get_colors(10, colors + [(1, 1, 1)])
markers = "o*Xvs"
marker_sizes = {"o": 7, "*": 8, "X": 8, "v": 10, "s": 7, "none": 0}
styles = ["-", "--", "-."]
annotation_facecolor = (1.0, 0.7, 0.7)
annotation_textcolor = "black"
annotation_font = ("TkDefaultFont", 10)
annotation_padding_x = 10
annotation_padding_y = 8
annotation_connector_width = 2
annotation_default_offset = (20, -20)
shift_mask = 0x0001


@dataclass
class Line:
    """Class that represents a line in a 2D plot.
    :var xs: X-axis coordinates of the plotted points.
    :var ys: Y-axis coordinates of the plotted points.
    :var error_margins: Optional y-axis error bars for each point.
    :var color: Matplotlib color or index into predefined color list.
    :var marker: Matplotlib marker or index into predefined marker list.
    :var style: Matplotlib line style or index into predefined style list.
    :var label: Legend label, or ``'_nolabel_'`` to omit from legend.
    """
    xs: Sequence
    ys: Sequence
    error_margins: Sequence = None
    color: str | tuple | int = colors[0]
    marker: str | int = "o"
    style: str | int = "-"
    label: str = "_nolabel_"

    def set_color(self, color: tuple | int):
        self.color = color
        if isinstance(color, int):
            self.color = colors[color]

    def __post_init__(self):
        if isinstance(self.color, int):
            self.color = colors[self.color]
        if isinstance(self.marker, int):
            self.marker = markers[self.marker]
        if isinstance(self.style, int):
            self.style = styles[self.style]


@dataclass
class AnnotationBubble:
    """Stores one native Tk-canvas annotation bubble.
    :var point: Data-space point annotated by this bubble.
    :var axes: Matplotlib axes used to transform ``point`` during redraws.
    :var text: Current bubble text.
    :var offset_canvas: Bubble offset from the anchor point in Tk-canvas coordinates.
    :var connector_id: Tk canvas line item that connects the anchor point to the bubble.
    :var bubble_id: Tk canvas rectangle item that draws the bubble background.
    :var text_id: Tk canvas text item that displays bubble text when not editing.
    :var tag: Shared Tk canvas tag used to bind events for all bubble items.
    """
    point: tuple[float, float]
    axes: Axes
    text: str
    offset_canvas: tuple[float, float]
    connector_id: int
    bubble_id: int
    text_id: int
    tag: str


@dataclass
class AnnotationManager:
    """Manager of interactive native Tk-canvas annotation bubbles.
    :var annotations: Mapping from point coordinates to active annotation bubbles.
    :var canvas: Matplotlib canvas associated with the managed figure.
    :var canvas_widget: Tk canvas widget that hosts the rendered figure and native bubble items.
    :var axes_lines: Data line artists registered for click hit-testing by axes.
    :var default_key_press_handler_id: Matplotlib callback id for the figure's default shortcut handler, or ``None`` when disabled.
    :var editing_point: Point whose annotation is currently being edited, or ``None`` when idle.
    :var original_text: Original annotation text restored when editing is canceled.
    :var editor_widget: Active Tk text widget used for in-bubble text editing.
    :var editor_window_id: Canvas item id that anchors the active text widget inside the Tk canvas.
    :var editor_size_buffer: Extra width and height reserved while editing to prevent transient text scrolling.
    :var dragging_point: Point whose bubble is currently being dragged, or ``None`` when idle.
    :var drag_start_canvas: Mouse position at the start of the current drag in Tk-canvas coordinates.
    :var drag_start_offset: Bubble offset at the start of the current drag.
    :var next_tag_id: Monotonic counter used to generate unique Tk canvas tags for new bubbles.
    """
    annotations: dict[tuple[float, float], AnnotationBubble] = field(default_factory=dict)
    canvas: object | None = None
    canvas_widget: tk.Canvas | None = None
    axes_lines: dict[Axes, list[Line2D]] = field(default_factory=dict)
    default_key_press_handler_id: int | None = None
    editing_point: tuple[float, float] | None = None
    original_text: str = ""
    editor_widget: tk.Text | None = None
    editor_window_id: int | None = None
    editor_size_buffer: tuple[int, int] = (0, 0)
    dragging_point: tuple[float, float] | None = None
    drag_start_canvas: tuple[float, float] | None = None
    drag_start_offset: tuple[float, float] | None = None
    next_tag_id: int = 0

    def attach_canvas(self, canvas: object):
        """Attaches the Matplotlib and Tk canvases used for native bubble rendering.
        :param canvas: Matplotlib canvas associated with the current figure.
        """
        self.canvas = canvas
        self.canvas_widget = canvas.get_tk_widget()

    def register_line(self, axes: Axes, artist: Line2D):
        """Registers one plotted data line for point hit-testing.
        :param axes: Axes that own the plotted line.
        :param artist: Main data line artist produced by Matplotlib plotting.
        """
        self.axes_lines.setdefault(axes, []).append(artist)

    def button_press_event_handler(self, event):
        """Handles clicks on plotted data points.
        :param event: Matplotlib button-press event produced after clicking in the plot.
        """
        if self.editing_point is not None:
            return
        if event.dblclick or event.inaxes is None:
            return
        point = self._get_closest_clicked_point(event)
        if point is None:
            return
        self._toggle_annotation(point, event.inaxes)
        event.canvas.draw_idle()

    def draw_event_handler(self, event):
        """Repositions all native bubbles after the figure is redrawn.
        :param event: Matplotlib draw event produced after the canvas is rendered.
        """
        if self.canvas is None:
            self.attach_canvas(event.canvas)
        for point in self.annotations:
            self._update_annotation_geometry(point)

    def _toggle_annotation(self, point: tuple[float, float], axes: Axes):
        """Creates or removes a native bubble for one data point.
        :param point: Coordinates of the data point.
        :param axes: Matplotlib axes where the annotated point lives.
        """
        if point in self.annotations:
            self._remove_annotation(point)
            return
        self._create_annotation(point, axes)

    def _create_annotation(self, point: tuple[float, float], axes: Axes):
        """Creates native Tk-canvas items for one annotation bubble.
        :param point: Coordinates of the annotated data point.
        :param axes: Matplotlib axes where the annotated point lives.
        """
        assert self.canvas_widget is not None, "Canvas widget must be attached before creating annotations."
        tag = f"annotation_{self.next_tag_id}"
        self.next_tag_id += 1
        connector_id = self.canvas_widget.create_line(0, 0, 0, 0, fill=self._rgb_to_hex(annotation_facecolor), width=annotation_connector_width, tags=(tag,))
        bubble_id = self.canvas_widget.create_rectangle(0, 0, 0, 0, fill=self._rgb_to_hex(annotation_facecolor), outline="", tags=(tag,))
        text = self._format_default_text(point)
        text_id = self.canvas_widget.create_text(0, 0, text=text, anchor="nw", fill=annotation_textcolor, font=annotation_font, tags=(tag,))
        self.annotations[point] = AnnotationBubble(point, axes, text, annotation_default_offset, connector_id, bubble_id, text_id, tag)
        self._bind_annotation_events(point)
        self._update_annotation_geometry(point)

    def _bind_annotation_events(self, point: tuple[float, float]):
        """Binds native Tk mouse events for one annotation bubble.
        :param point: Coordinates of the annotated data point.
        """
        assert self.canvas_widget is not None, "Canvas widget must be attached before binding annotation events."
        tag = self.annotations[point].tag
        self.canvas_widget.tag_bind(tag, "<ButtonPress-1>", lambda event, point=point: self._bubble_press_event(point, event))
        self.canvas_widget.tag_bind(tag, "<B1-Motion>", lambda event, point=point: self._bubble_motion_event(point, event))
        self.canvas_widget.tag_bind(tag, "<ButtonRelease-1>", lambda event, point=point: self._bubble_release_event(point, event))
        self.canvas_widget.tag_bind(tag, "<Double-Button-1>", lambda event, point=point: self._bubble_double_click_event(point, event))

    def _bubble_press_event(self, point: tuple[float, float], event: object) -> str:
        """Starts dragging one bubble after a mouse press on its native canvas items.
        :param point: Coordinates of the annotated data point.
        :param event: Tk mouse event raised on the annotation items.
        :return: ``"break"`` so Tk does not continue propagating the click.
        """
        if self.editing_point is not None:
            return "break"
        self.dragging_point = point
        self.drag_start_canvas = (event.x, event.y)
        self.drag_start_offset = self.annotations[point].offset_canvas
        self._raise_annotation(point)
        return "break"

    def _bubble_motion_event(self, point: tuple[float, float], event: object) -> str:
        """Updates bubble position while dragging native canvas items.
        :param point: Coordinates of the annotated data point.
        :param event: Tk mouse event raised while the bubble is being dragged.
        :return: ``"break"`` so Tk does not continue propagating the drag event.
        """
        if self.dragging_point != point or self.drag_start_canvas is None or self.drag_start_offset is None:
            return "break"
        dx = event.x - self.drag_start_canvas[0]
        dy = event.y - self.drag_start_canvas[1]
        self.annotations[point].offset_canvas = (self.drag_start_offset[0] + dx, self.drag_start_offset[1] + dy)
        self._update_annotation_geometry(point)
        return "break"

    def _bubble_release_event(self, point: tuple[float, float], event: object) -> str:
        """Finishes dragging after mouse release on one bubble.
        :param point: Coordinates of the annotated data point.
        :param event: Tk mouse event raised on button release.
        :return: ``"break"`` so Tk does not continue propagating the release event.
        """
        if self.dragging_point == point:
            self.dragging_point = None
            self.drag_start_canvas = None
            self.drag_start_offset = None
        return "break"

    def _bubble_double_click_event(self, point: tuple[float, float], event: object) -> str:
        """Starts in-place editing after double-clicking one native bubble.
        :param point: Coordinates of the annotated data point.
        :param event: Tk mouse event raised on double click.
        :return: ``"break"`` so Tk does not continue propagating the double click.
        """
        if self.editing_point is not None:
            return "break"
        self.dragging_point = None
        self.drag_start_canvas = None
        self.drag_start_offset = None
        self._start_editing(point)
        return "break"

    def _start_editing(self, point: tuple[float, float]):
        """Starts native Tk text editing inside one bubble.
        :param point: Coordinates of the annotated data point.
        """
        assert self.canvas is not None, "Canvas must be attached before starting editing."
        self._disable_default_shortcuts(self.canvas)
        self.editing_point = point
        self.original_text = self.annotations[point].text
        self._create_editor()

    def _create_editor(self):
        """Creates and configures the native Tk text widget used inside the active bubble."""
        assert self.editing_point is not None, "Editing state must exist before creating the editor."
        assert self.canvas_widget is not None, "Canvas widget must be attached before creating the editor."
        bubble = self.annotations[self.editing_point]
        self.editor_widget = tk.Text(self.canvas_widget, relief="flat", borderwidth=0, highlightthickness=0, bg=self._rgb_to_hex(annotation_facecolor),
                                     fg=annotation_textcolor, insertbackground=annotation_textcolor, exportselection=False, wrap="none",
                                     undo=True, padx=0, pady=0, font=annotation_font)
        editor_font = tkfont.Font(root=self.canvas_widget, font=annotation_font)
        self.editor_size_buffer = (editor_font.measure("0") + 4, max(2, editor_font.metrics("linespace") // 4))
        self.editor_widget.insert("1.0", bubble.text)
        self.editor_widget.edit_modified(False)
        self.editor_widget.bind("<<Modified>>", self._editor_text_updated)
        self.editor_widget.bind("<Return>", self._enter_editing_event)
        self.editor_widget.bind("<KP_Enter>", self._enter_editing_event)
        self.editor_widget.bind("<Escape>", self._cancel_editing_event)
        self.editor_widget.bind("<FocusOut>", self._save_editing_event)
        self.editor_widget.bind("<Control-a>", self._select_all_editor_text)
        self.editor_widget.bind("<Control-A>", self._select_all_editor_text)
        self.editor_window_id = self.canvas_widget.create_window(0, 0, anchor="nw", window=self.editor_widget)
        self.canvas_widget.itemconfigure(bubble.text_id, fill=self._rgb_to_hex(annotation_facecolor))
        self._update_annotation_geometry(self.editing_point)
        self.editor_widget.focus_set()
        self.editor_widget.mark_set(tk.INSERT, "end-1c")

    def _editor_text_updated(self, event: object):
        """Synchronizes native Tk text edits back into the bubble geometry.
        :param event: Tk virtual event raised after text widget content changes.
        """
        if self.editing_point is None or self.editor_widget is None or not self.editor_widget.edit_modified():
            return
        self.editor_widget.edit_modified(False)
        self.annotations[self.editing_point].text = self.editor_widget.get("1.0", "end-1c")
        self._update_annotation_geometry(self.editing_point)

    def _enter_editing_event(self, event: object) -> str:
        """Handles Enter while editing, saving on plain Enter and inserting a newline on Shift+Enter.
        :param event: Tk key event raised by the active text widget.
        :return: ``"break"`` so Tk does not continue handling the same event.
        """
        assert self.editor_widget is not None, "Editor widget must exist before handling Enter."
        if event.state & shift_mask:
            if self.editor_widget.tag_ranges(tk.SEL):
                self.editor_widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.editor_widget.insert(tk.INSERT, "\n")
            return "break"
        return self._save_editing_event(event)

    def _save_editing_event(self, event: object) -> str:
        """Commits native Tk text editing for the active bubble.
        :param event: Tk event raised by the active text widget.
        :return: ``"break"`` so Tk does not continue handling the same event.
        """
        if self.editing_point is None or self.editor_widget is None:
            return "break"
        self.annotations[self.editing_point].text = self.editor_widget.get("1.0", "end-1c")
        self._finish_editing(save=True)
        return "break"

    def _cancel_editing_event(self, event: object) -> str:
        """Cancels native Tk text editing for the active bubble.
        :param event: Tk event raised by the active text widget.
        :return: ``"break"`` so Tk does not continue handling the same event.
        """
        if self.editing_point is None or self.editor_widget is None:
            return "break"
        self._finish_editing(save=False)
        return "break"

    def _select_all_editor_text(self, event: object) -> str:
        """Selects all text inside the active native Tk text widget.
        :param event: Tk event raised by the active text widget.
        :return: ``"break"`` so Tk does not continue handling the same event.
        """
        assert self.editor_widget is not None, "Editor widget must exist before selecting text."
        self.editor_widget.tag_add(tk.SEL, "1.0", "end-1c")
        self.editor_widget.mark_set(tk.INSERT, "end-1c")
        self.editor_widget.see(tk.INSERT)
        return "break"

    def _finish_editing(self, save: bool):
        """Finishes native bubble editing and either commits or restores the original text.
        :param save: Whether to keep the edited text instead of restoring the original text.
        """
        assert self.canvas is not None, "Canvas must be attached before finishing editing."
        assert self.canvas_widget is not None, "Canvas widget must be attached before finishing editing."
        point = self.editing_point
        if point is None:
            return
        bubble = self.annotations[point]
        bubble.text = bubble.text if save else self.original_text
        self._destroy_editor()
        self.canvas_widget.itemconfigure(bubble.text_id, fill=annotation_textcolor)
        self._enable_default_shortcuts(self.canvas)
        self.editing_point = None
        self.original_text = ""
        self._update_annotation_geometry(point)
        self.canvas.draw_idle()

    def _destroy_editor(self):
        """Destroys the active native Tk text widget and its canvas window item."""
        assert self.canvas_widget is not None, "Canvas widget must be attached before destroying the editor."
        editor_window_id = self.editor_window_id
        editor_widget = self.editor_widget
        self.editor_window_id = None
        self.editor_widget = None
        self.editor_size_buffer = (0, 0)
        if editor_window_id is not None:
            self.canvas_widget.delete(editor_window_id)
        if editor_widget is not None:
            editor_widget.destroy()

    def _update_annotation_geometry(self, point: tuple[float, float]):
        """Updates one bubble's native canvas geometry from its anchor point and current text.
        :param point: Coordinates of the annotated data point.
        """
        assert self.canvas_widget is not None, "Canvas widget must be attached before updating annotations."
        bubble = self.annotations[point]
        anchor_x, anchor_y = self._data_to_canvas_coords(bubble.axes, bubble.point)
        text_x = round(anchor_x + bubble.offset_canvas[0]) + annotation_padding_x
        text_y = round(anchor_y + bubble.offset_canvas[1]) + annotation_padding_y
        self.canvas_widget.itemconfigure(bubble.text_id, text=bubble.text if len(bubble.text) > 0 else " ")
        self.canvas_widget.coords(bubble.text_id, text_x, text_y)
        text_bbox = self.canvas_widget.bbox(bubble.text_id)
        assert text_bbox is not None, "Text item must have a bounding box after being positioned."
        editor_text_bbox = (text_bbox[0], text_bbox[1], text_bbox[2] + self.editor_size_buffer[0], text_bbox[3] + self.editor_size_buffer[1]) \
            if point == self.editing_point else text_bbox
        bubble_bbox = (editor_text_bbox[0] - annotation_padding_x, editor_text_bbox[1] - annotation_padding_y,
                       editor_text_bbox[2] + annotation_padding_x, editor_text_bbox[3] + annotation_padding_y)
        self.canvas_widget.coords(bubble.bubble_id, *bubble_bbox)
        connector_x, connector_y = self._get_connector_target((anchor_x, anchor_y), bubble_bbox)
        self.canvas_widget.coords(bubble.connector_id, round(anchor_x), round(anchor_y), connector_x, connector_y)
        self.canvas_widget.tag_raise(bubble.connector_id)
        self.canvas_widget.tag_raise(bubble.bubble_id)
        self.canvas_widget.tag_raise(bubble.text_id)
        if point == self.editing_point:
            self._place_editor(editor_text_bbox)

    def _place_editor(self, text_bbox: tuple[int, int, int, int]):
        """Places the active native Tk text widget inside the current bubble text bounds.
        :param text_bbox: Canvas bounding box of the hidden text item used for bubble sizing.
        """
        assert self.canvas_widget is not None, "Canvas widget must be attached before positioning the editor."
        assert self.editor_widget is not None, "Editor widget must exist before positioning."
        assert self.editor_window_id is not None, "Editor window item must exist before positioning."
        self.canvas_widget.coords(self.editor_window_id, text_bbox[0], text_bbox[1])
        self.canvas_widget.itemconfigure(self.editor_window_id, width=max(1, text_bbox[2] - text_bbox[0]), height=max(1, text_bbox[3] - text_bbox[1]))
        self.canvas_widget.tag_raise(self.editor_window_id)

    def _raise_annotation(self, point: tuple[float, float]):
        """Raises one bubble's native canvas items above the rendered plot.
        :param point: Coordinates of the annotated data point.
        """
        assert self.canvas_widget is not None, "Canvas widget must be attached before raising annotations."
        bubble = self.annotations[point]
        self.canvas_widget.tag_raise(bubble.connector_id)
        self.canvas_widget.tag_raise(bubble.bubble_id)
        self.canvas_widget.tag_raise(bubble.text_id)
        if point == self.editing_point and self.editor_window_id is not None:
            self.canvas_widget.tag_raise(self.editor_window_id)

    def _disable_default_shortcuts(self, canvas: object):
        """Disables Matplotlib's built-in key shortcuts for the current figure.
        :param canvas: Figure canvas whose default key handler should be disabled.
        """
        manager = canvas.manager
        self.default_key_press_handler_id = getattr(manager, "key_press_handler_id", None)
        if self.default_key_press_handler_id is None:
            return
        canvas.mpl_disconnect(self.default_key_press_handler_id)
        manager.key_press_handler_id = None

    def _enable_default_shortcuts(self, canvas: object):
        """Re-enables Matplotlib's built-in key shortcuts for the current figure.
        :param canvas: Figure canvas whose default key handler should be re-enabled.
        """
        if self.default_key_press_handler_id is None:
            return
        manager = canvas.manager
        manager.key_press_handler_id = canvas.mpl_connect("key_press_event", key_press_handler)
        self.default_key_press_handler_id = None

    def _remove_annotation(self, point: tuple[float, float]):
        """Removes one native bubble from the Tk canvas.
        :param point: Coordinates of the annotated data point.
        """
        assert self.canvas_widget is not None, "Canvas widget must be attached before removing annotations."
        if point == self.editing_point:
            self._finish_editing(save=False)
        bubble = self.annotations.pop(point)
        self.canvas_widget.delete(bubble.connector_id, bubble.bubble_id, bubble.text_id)

    def _get_connector_target(self, anchor: tuple[float, float], bubble_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        """Returns the point on the bubble border nearest to the anchor point.
        :param anchor: Anchor point in Tk-canvas coordinates.
        :param bubble_bbox: Bubble rectangle bounds in Tk-canvas coordinates.
        :return: Bubble-border point where the connector line should terminate.
        """
        return round(np.clip(anchor[0], bubble_bbox[0], bubble_bbox[2])), round(np.clip(anchor[1], bubble_bbox[1], bubble_bbox[3]))

    def _data_to_canvas_coords(self, axes: Axes, point: tuple[float, float]) -> tuple[float, float]:
        """Transforms one data-space point into Tk-canvas coordinates.
        :param axes: Matplotlib axes that define the data transform.
        :param point: Data-space point to transform.
        :return: Tk-canvas coordinates of the point.
        """
        assert self.canvas is not None, "Canvas must be attached before transforming coordinates."
        assert self.canvas_widget is not None, "Canvas widget must be attached before transforming coordinates."
        display_x, display_y = axes.transData.transform(point)
        image_center_x, image_center_y = self.canvas_widget.coords(self.canvas._tkcanvas_image_region)
        return image_center_x - self.canvas._tkphoto.width() / 2 + display_x, image_center_y + self.canvas._tkphoto.height() / 2 - display_y

    def _get_closest_clicked_point(self, event) -> tuple[float, float] | None:
        """Returns nearest clicked data point to the mouse position.
        :param event: Matplotlib button-press event produced after clicking in the plot.
        :return: Coordinates of the nearest clicked point, or ``None`` when the click is too far from any marker.
        """
        closest_point = None
        closest_distance = np.inf
        click_coords = np.array([event.x, event.y])
        for artist in self.axes_lines.get(event.inaxes, []):
            contains, details = artist.contains(event)
            if not contains:
                continue
            if artist.get_markersize() <= 0:
                continue
            x_data = np.asarray(artist.get_xdata())
            y_data = np.asarray(artist.get_ydata())
            if len(x_data) == 0:
                continue
            candidate_coords = np.column_stack((x_data, y_data))
            candidate_disp_coords = artist.get_transform().transform(candidate_coords)
            candidate_indices = details.get("ind", range(len(candidate_coords)))
            distances = linalg.norm(candidate_disp_coords[candidate_indices] - click_coords, axis=1)
            closest_local_index = int(np.argmin(distances))
            distance = distances[closest_local_index]
            if distance > max(artist.get_markersize(), artist.get_pickradius()):
                continue
            if distance >= closest_distance:
                continue
            closest_distance = distance
            closest_index = int(candidate_indices[closest_local_index])
            closest_point = float(candidate_coords[closest_index][0]), float(candidate_coords[closest_index][1])
        return closest_point

    @staticmethod
    def _format_default_text(point: tuple[float, float]) -> str:
        """Formats the default bubble text for one annotated point.
        :param point: Coordinates of the annotated data point.
        :return: Default bubble label text.
        """
        return f"({point[0]:.3g}, {point[1]:.3g})"

    @staticmethod
    def _rgb_to_hex(color: tuple[float, float, float]) -> str:
        """Converts a Matplotlib RGB tuple to a Tk color string.
        :param color: RGB tuple with components in the range ``[0, 1]``.
        :return: Tk-compatible hexadecimal color string.
        """
        return "#{:02x}{:02x}{:02x}".format(*(round(255 * component) for component in color))


def data_matrix_to_lines(data: ndarray, line_labels: list[str] = None, colors: list[int] = None, **kwargs) -> list[Line]:
    """Converts a given data matrix to a set of lines (each line is a row).
    :param data: 3D data matrix of size 2 x num_lines x num_points. 1st dim - (x, y); 2nd - lines; 3rd - data points.
    :param line_labels: Line labels.
    :param colors: Line colors.
    :param kwargs: Unused compatibility arguments accepted for call-site convenience.
    :return: List of lines.
    """
    lines = []
    for i in range(data.shape[1]):
        xs = np.trim_zeros(data[0, i, :], trim="b")
        ys = np.trim_zeros(data[1, i, :], trim="b")
        lines.append(Line(xs, ys))
        if colors is not None:
            lines[-1].set_color(colors[i])
        if line_labels is not None:
            lines[-1].label = line_labels[i]
    return lines


def plot_general(lines: list[Line], axis_labels: tuple[str | None, str | None] = None, tick_multiples: tuple[float | None, float | None] = None,
                 boundaries: tuple[float | None, float | None, float | None, float | None] = None, font_size: int = 20, legend_loc: str = "best",
                 figure_id: int = None, **kwargs):
    """Plots specified list of lines.
    :param lines: List of lines.
    :param axis_labels: Labels for x and y axes.
    :param tick_multiples: Base multiples for ticks along x and y axes.
    :param boundaries: x min, x max, y min, y max floats defining plot boundaries.
    :param font_size: Font size.
    :param legend_loc: Location of legend.
    :param figure_id: ID of the figure where the results should be plotted or None to create new figure.
    :param kwargs: Extra plotting arguments reserved for future extensions.
    """
    if figure_id is None:
        new_figure = True
        fig = plt.figure()
    else:
        new_figure = plt.fignum_exists(figure_id)
        fig = plt.figure(figure_id)
    manager = AnnotationManager()
    manager.attach_canvas(fig.canvas)
    fig.canvas.mpl_connect("button_press_event", lambda event: manager.button_press_event_handler(event))
    fig.canvas.mpl_connect("draw_event", lambda event: manager.draw_event_handler(event))
    plt.rcParams.update({"font.size": font_size})

    for line in lines:
        errorbar = plt.errorbar(line.xs, line.ys, yerr=line.error_margins, color=line.color, marker=line.marker, linestyle=line.style,
                                markersize=marker_sizes[line.marker], label=line.label, capsize=5, picker=5)
        manager.register_line(plt.gca(), errorbar.lines[0])
        if line.label != "_nolabel_":
            handles, labels = plt.gca().get_legend_handles_labels()
            handles = [h[0] for h in handles]  # strips the error bars from the legend.
            plt.legend(handles, labels, loc=legend_loc, draggable=True)

    if axis_labels is not None:
        if axis_labels[0] is not None:
            plt.xlabel(axis_labels[0])
        if axis_labels[1] is not None:
            plt.ylabel(axis_labels[1])

    if tick_multiples is not None:
        if tick_multiples[0] is not None:
            plt.gca().xaxis.set_major_locator(MultipleLocator(tick_multiples[0]))
        if tick_multiples[1] is not None:
            plt.gca().yaxis.set_major_locator(MultipleLocator(tick_multiples[1]))

    if boundaries is not None:
        if boundaries[0] is not None:
            plt.xlim(left=boundaries[0])
        if boundaries[1] is not None:
            plt.xlim(right=boundaries[1])
        if boundaries[2] is not None:
            plt.ylim(bottom=boundaries[2])
        if boundaries[3] is not None:
            plt.ylim(top=boundaries[3])

    if new_figure:
        win = fig.canvas.manager.window
        win.after(1000, lambda: win.state("zoomed"))
        plt.gca().set_box_aspect(1)
        plt.gcf().set_size_inches(10, 10)


def save_figure(file_name: str = None):
    """Saves figure to a file.
    :param file_name: Name of the file or None to use caller's name (without plot_).
    """
    file_name = inspect.currentframe().f_back.f_code.co_name[5:] if file_name is None else file_name
    plt.savefig(f"out/{file_name}.jpg", dpi=300, bbox_inches="tight")
