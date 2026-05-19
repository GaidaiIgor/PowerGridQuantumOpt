"""Provides general QtAgg plotting helpers."""
from collections.abc import Sequence
import inspect
from dataclasses import dataclass, field

import distinctipy
import numpy as np
from matplotlib import use as use_matplotlib_backend

use_matplotlib_backend("QtAgg", force=True)

from matplotlib.axes import Axes
from matplotlib.backend_bases import DrawEvent, MouseButton, MouseEvent, key_press_handler
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import MultipleLocator
from numpy import linalg
from numpy import ndarray
from PyQt6.QtWidgets import QInputDialog

colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (0, 0, 0), (0, 0.75, 0.75), (0.75, 0, 0.75), (0.75, 0.75, 0)]
colors += distinctipy.get_colors(10, colors + [(1, 1, 1)])
markers = "o*Xvs"
marker_sizes = {"o": 7, "*": 8, "X": 8, "v": 10, "s": 7, "none": 0}
styles = ["-", "--", "-."]
annotation_facecolor = (1, 0.7, 0.7)
annotation_textcolor = "black"
annotation_box_padding = 0.4
annotation_connector_width = 2
annotation_default_offset = (20, -20)


@dataclass
class Line:
    """Represents a line in a 2D plot.
    :var xs: X-axis coordinates of the plotted points.
    :var ys: Y-axis coordinates of the plotted points.
    :var error_margins: Optional y-axis error bars for each point.
    :var color: Matplotlib color or index into predefined color list.
    :var marker: Matplotlib marker or index into predefined marker list.
    :var style: Matplotlib line style or index into predefined style list.
    :var label: Legend label, or ``"_nolabel_"`` to omit from legend."""
    xs: Sequence
    ys: Sequence
    error_margins: Sequence | None = None
    color: str | tuple[float, float, float] | int = colors[0]
    marker: str | int = "o"
    style: str | int = "-"
    label: str = "_nolabel_"

    def set_color(self, color: tuple[float, float, float] | int):
        """Sets the line color from a concrete RGB tuple or predefined color index.
        :param color: RGB tuple or index into the predefined color list."""
        self.color = color
        if isinstance(color, int):
            self.color = colors[color]

    def __post_init__(self):
        """Normalizes indexed color, marker, and style values after initialization."""
        if isinstance(self.color, int):
            self.color = colors[self.color]
        if isinstance(self.marker, int):
            self.marker = markers[self.marker]
        if isinstance(self.style, int):
            self.style = styles[self.style]


@dataclass
class AnnotationBubble:
    """Stores one Matplotlib annotation bubble.
    :var point: Data-space point annotated by this bubble.
    :var axes: Matplotlib axes that own the annotation artist.
    :var annotation: Matplotlib annotation artist that draws the bubble.
    :var text: Current bubble text.
    :var drag_start_canvas: Mouse position at the start of the current drag in display coordinates.
    :var drag_start_offset: Bubble offset at the start of the current drag in display coordinates."""
    point: tuple[float, float]
    axes: Axes
    annotation: Annotation
    text: str
    drag_start_canvas: tuple[float, float] | None = None
    drag_start_offset: tuple[float, float] | None = None


@dataclass
class AnnotationManager:
    """Manages interactive QtAgg-compatible annotation bubbles.
    :var annotations: Mapping from point coordinates to active annotation bubbles.
    :var canvas: Matplotlib canvas associated with the managed figure.
    :var axes_lines: Data line artists registered for click hit-testing by axes.
    :var default_key_press_handler_id: Matplotlib callback id for the default figure shortcut handler, or ``None`` when disabled.
    :var editing_point: Point whose annotation is currently being edited, or ``None`` when idle.
    :var dragging_point: Point whose bubble is currently being dragged, or ``None`` when idle."""
    annotations: dict[tuple[float, float], AnnotationBubble] = field(default_factory=dict)
    canvas: object | None = None
    axes_lines: dict[Axes, list[Line2D]] = field(default_factory=dict)
    default_key_press_handler_id: int | None = None
    editing_point: tuple[float, float] | None = None
    dragging_point: tuple[float, float] | None = None

    def attach_canvas(self, canvas: object):
        """Attaches the Matplotlib canvas used for annotation event handling.
        :param canvas: Matplotlib canvas associated with the current figure."""
        self.canvas = canvas

    def register_line(self, axes: Axes, artist: Line2D):
        """Registers one plotted data line for point hit-testing.
        :param axes: Axes that own the plotted line.
        :param artist: Main data line artist produced by Matplotlib plotting."""
        self.axes_lines.setdefault(axes, []).append(artist)

    def button_press_event_handler(self, event: MouseEvent):
        """Handles clicks on plotted data points and annotation bubbles.
        :param event: Matplotlib button-press event produced after clicking in the plot."""
        if self.editing_point is not None or event.button != MouseButton.LEFT:
            return
        point = self._get_clicked_annotation(event)
        if point is not None:
            if event.dblclick:
                self._start_editing(point)
            else:
                self._start_dragging(point, event)
            return
        if event.dblclick or event.inaxes is None:
            return
        clicked_point = self._get_closest_clicked_point(event)
        if clicked_point is None:
            return
        self._toggle_annotation(clicked_point, event.inaxes)
        event.canvas.draw_idle()

    def motion_notify_event_handler(self, event: MouseEvent):
        """Moves the active annotation bubble while it is being dragged.
        :param event: Matplotlib mouse-motion event produced by the figure canvas."""
        if self.dragging_point is None:
            return
        bubble = self.annotations[self.dragging_point]
        assert bubble.drag_start_canvas is not None, "Drag start coordinates must be recorded before dragging."
        assert bubble.drag_start_offset is not None, "Drag start offset must be recorded before dragging."
        dx = event.x - bubble.drag_start_canvas[0]
        dy = event.y - bubble.drag_start_canvas[1]
        bubble.annotation.set_position((bubble.drag_start_offset[0] + dx, bubble.drag_start_offset[1] + dy))
        event.canvas.draw_idle()

    def button_release_event_handler(self, event: MouseEvent):
        """Finishes dragging the active annotation bubble.
        :param event: Matplotlib button-release event produced by the figure canvas."""
        if self.dragging_point is None:
            return
        bubble = self.annotations[self.dragging_point]
        bubble.drag_start_canvas = None
        bubble.drag_start_offset = None
        self.dragging_point = None

    def draw_event_handler(self, event: DrawEvent):
        """Caches the canvas when Matplotlib draws the figure.
        :param event: Matplotlib draw event produced after the canvas is rendered."""
        if self.canvas is None:
            self.attach_canvas(event.canvas)

    def _get_clicked_annotation(self, event: MouseEvent) -> tuple[float, float] | None:
        """Returns the annotation bubble clicked by an event, if any.
        :param event: Matplotlib button-press event to test against existing annotations.
        :return: Coordinates of the clicked annotation point, or ``None`` when no annotation was clicked."""
        for point, bubble in reversed(list(self.annotations.items())):
            contains, _ = bubble.annotation.contains(event)
            if contains:
                return point
        return None

    def _start_dragging(self, point: tuple[float, float], event: MouseEvent):
        """Starts dragging one annotation bubble.
        :param point: Coordinates of the annotated data point.
        :param event: Matplotlib button-press event that started the drag."""
        bubble = self.annotations[point]
        self.dragging_point = point
        bubble.drag_start_canvas = (event.x, event.y)
        bubble.drag_start_offset = bubble.annotation.get_position()
        self._raise_annotation(point)

    def _toggle_annotation(self, point: tuple[float, float], axes: Axes):
        """Creates or removes an annotation bubble for one data point.
        :param point: Coordinates of the data point.
        :param axes: Matplotlib axes where the annotated point lives."""
        if point in self.annotations:
            self._remove_annotation(point)
            return
        self._create_annotation(point, axes)

    def _create_annotation(self, point: tuple[float, float], axes: Axes):
        """Creates a Matplotlib annotation bubble for one data point.
        :param point: Coordinates of the annotated data point.
        :param axes: Matplotlib axes where the annotated point lives."""
        text = self._format_default_text(point)
        bbox = {"boxstyle": f"round,pad={annotation_box_padding:g}", "facecolor": annotation_facecolor, "edgecolor": "none"}
        arrowprops = {"arrowstyle": "-", "color": annotation_facecolor, "linewidth": annotation_connector_width}
        annotation = axes.annotate(text, xy=point, xytext=annotation_default_offset, textcoords="offset pixels", color=annotation_textcolor, bbox=bbox,
                                   arrowprops=arrowprops, zorder=10)
        annotation.set_picker(True)
        self.annotations[point] = AnnotationBubble(point, axes, annotation, text)

    def _start_editing(self, point: tuple[float, float]):
        """Opens a Qt text dialog for editing one annotation bubble.
        :param point: Coordinates of the annotated data point."""
        assert self.canvas is not None, "Canvas must be attached before starting editing."
        self.editing_point = point
        self._disable_default_shortcuts(self.canvas)
        try:
            bubble = self.annotations[point]
            text, accepted = QInputDialog.getMultiLineText(self.canvas.manager.window, "Edit Annotation", "Text:", bubble.text)
            if accepted:
                bubble.text = text
                bubble.annotation.set_text(text if len(text) > 0 else " ")
                self.canvas.draw_idle()
        finally:
            self.editing_point = None
            self._enable_default_shortcuts(self.canvas)

    def _raise_annotation(self, point: tuple[float, float]):
        """Raises one annotation bubble above the rest.
        :param point: Coordinates of the annotated data point."""
        zorder = max((bubble.annotation.get_zorder() for bubble in self.annotations.values()), default=10) + 1
        self.annotations[point].annotation.set_zorder(zorder)

    def _disable_default_shortcuts(self, canvas: object):
        """Disables Matplotlib built-in key shortcuts for the current figure.
        :param canvas: Figure canvas whose default key handler should be disabled."""
        manager = canvas.manager
        self.default_key_press_handler_id = getattr(manager, "key_press_handler_id", None)
        if self.default_key_press_handler_id is None:
            return
        canvas.mpl_disconnect(self.default_key_press_handler_id)
        manager.key_press_handler_id = None

    def _enable_default_shortcuts(self, canvas: object):
        """Re-enables Matplotlib built-in key shortcuts for the current figure.
        :param canvas: Figure canvas whose default key handler should be re-enabled."""
        if self.default_key_press_handler_id is None:
            return
        manager = canvas.manager
        manager.key_press_handler_id = canvas.mpl_connect("key_press_event", key_press_handler)
        self.default_key_press_handler_id = None

    def _remove_annotation(self, point: tuple[float, float]):
        """Removes one annotation bubble.
        :param point: Coordinates of the annotated data point."""
        bubble = self.annotations.pop(point)
        bubble.annotation.remove()

    def _get_closest_clicked_point(self, event: MouseEvent) -> tuple[float, float] | None:
        """Returns nearest clicked data point to the mouse position.
        :param event: Matplotlib button-press event produced after clicking in the plot.
        :return: Coordinates of the nearest clicked point, or ``None`` when the click is too far from any marker."""
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
        :return: Default bubble label text."""
        return f"({point[0]:.3g}, {point[1]:.3g})"


def data_matrix_to_lines(data: ndarray, line_labels: list[str] | None = None, colors: list[int] | None = None, **kwargs: object) -> list[Line]:
    """Converts a given data matrix to a set of lines (each line is a row).
    :param data: 3D data matrix of size 2 x num_lines x num_points. 1st dim - (x, y); 2nd - lines; 3rd - data points.
    :param line_labels: Line labels.
    :param colors: Line colors.
    :param kwargs: Unused compatibility arguments accepted for call-site convenience.
    :return: List of lines."""
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


def apply_plot_settings(figure: Figure, font_size: int = 20, enable_annotations: bool = False, maximize: bool = True) -> AnnotationManager | None:
    """Applies common interactive plot display settings.
    :param figure: Figure whose display settings should be updated.
    :param font_size: Font size.
    :param enable_annotations: Whether plotted data lines should support click annotations.
    :param maximize: Whether to maximize the figure window and force a square axes box.
    :return: Annotation manager when annotations are enabled, otherwise ``None``."""
    manager = None
    if enable_annotations:
        manager = AnnotationManager()
        manager.attach_canvas(figure.canvas)
        figure.canvas.mpl_connect("button_press_event", lambda event: manager.button_press_event_handler(event))
        figure.canvas.mpl_connect("motion_notify_event", lambda event: manager.motion_notify_event_handler(event))
        figure.canvas.mpl_connect("button_release_event", lambda event: manager.button_release_event_handler(event))
        figure.canvas.mpl_connect("draw_event", lambda event: manager.draw_event_handler(event))
    plt.rcParams.update({"font.size": font_size})
    if maximize:
        figure.gca().set_box_aspect(1)
        figure.set_size_inches(10, 10)
        figure.canvas.manager.window.showMaximized()
    return manager


def plot_general(lines: list[Line], axis_labels: tuple[str | None, str | None] = None, tick_multiples: tuple[float | None, float | None] = None,
                 boundaries: tuple[float | None, float | None, float | None, float | None] = None, font_size: int = 20, legend_loc: str = "best",
                 figure_id: int | None = None, **kwargs: object):
    """Plots specified list of lines.
    :param lines: List of lines.
    :param axis_labels: Labels for x and y axes.
    :param tick_multiples: Base multiples for ticks along x and y axes.
    :param boundaries: x min, x max, y min, y max floats defining plot boundaries.
    :param font_size: Font size.
    :param legend_loc: Location of legend.
    :param figure_id: ID of the figure where the results should be plotted or None to create new figure.
    :param kwargs: Extra plotting arguments reserved for future extensions."""
    new_figure = figure_id is None or not plt.fignum_exists(figure_id)
    fig = plt.figure(figure_id)
    manager = apply_plot_settings(fig, font_size, True, new_figure)
    assert manager is not None, "Annotation manager must be created before registering plotted lines."

    for line in lines:
        errorbar = plt.errorbar(line.xs, line.ys, yerr=line.error_margins, color=line.color, marker=line.marker, linestyle=line.style,
                                markersize=marker_sizes[line.marker], label=line.label, capsize=5, picker=5)
        manager.register_line(plt.gca(), errorbar.lines[0])
        if line.label != "_nolabel_":
            handles, labels = plt.gca().get_legend_handles_labels()
            handles = [h[0] for h in handles]  # Strips the error bars from the legend.
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


def save_figure(file_path: str | None = None):
    """Saves figure to a file.
    :param file_path: Path to the output file or None to use default = out/caller name (without plot_)."""
    if file_path is None:
        file_name = inspect.currentframe().f_back.f_code.co_name[5:]
        file_path = f"out/{file_name}.jpg"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
