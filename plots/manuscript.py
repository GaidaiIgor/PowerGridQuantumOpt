"""Draws manuscript-specific quantum circuit figures."""
from itertools import combinations, pairwise
from math import cos, hypot, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

background_color = "#f8fbff"
blue = "#1550c8"
green = "#2b7a16"
orange = "#c97800"
rzz_edge = "#d28a22"
purple = "#5a3db6"
text_color = "#111111"
workflow_width = 1514


def draw_quantum_circuit(file_path: str | Path | None = None) -> Figure:
    """Draws a three-qubit all-to-all variational quantum circuit.
    :param file_path: Optional path where the rendered circuit image should be saved.
    :return: Matplotlib figure containing the rendered quantum circuit."""
    num_qubits = 3
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.h(range(num_qubits))
    circuit.barrier()

    zz_angles = ParameterVector("gamma", num_qubits * (num_qubits - 1) // 2)
    for angle, (i, j) in zip(zz_angles, combinations(range(num_qubits), 2), strict=True):
        circuit.rzz(angle, i, j)
    circuit.barrier()

    z_angles = ParameterVector("beta_z", num_qubits)
    x_angles = ParameterVector("beta_x", num_qubits)
    for i in range(num_qubits):
        circuit.rz(z_angles[i], i)
    circuit.barrier()
    for i in range(num_qubits):
        circuit.rx(x_angles[i], i)
    circuit.barrier()

    circuit.measure(range(num_qubits), range(num_qubits))
    figure = circuit.draw(output="mpl", fold=-1)
    figure.tight_layout()
    if file_path is not None:
        figure.savefig(file_path, dpi=300, bbox_inches="tight")
    return figure


def draw_workflow_outline(file_path: str | Path | None = None) -> Figure:
    """Draws a controlled-code version of the hybrid optimization workflow figure.
    :param file_path: Optional path where the rendered workflow image should be saved.
    :return: Matplotlib figure containing the workflow diagram."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 18})
    figure = plt.figure(figsize=(15.14, 11.48), facecolor=background_color)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_xlim(0, workflow_width)
    axes.set_ylim(0, 1148)
    axes.axis("off")
    axes.set_facecolor(background_color)
    gradient = [[i / 1148] for i in range(1149)]
    axes.imshow(gradient, extent=(0, workflow_width, 0, 1148), origin="lower", aspect="auto",
                cmap=LinearSegmentedColormap.from_list("workflow_background", ((0, "#feebd0"), (530 / 1148, "#ffffff"), (1, "#d9ebfe"))), zorder=-20)

    axes.text(workflow_width / 2, 1106, "Quantum", ha="center", va="center", fontsize=42, weight="bold", color=blue)
    _draw_quantum_panel(axes)
    _draw_exchange_line(axes)
    _draw_cost_panel(axes)
    _draw_optimizer_panel(axes)
    _draw_workflow_arrows(axes)

    axes.text(workflow_width / 2, 32, "Classical", ha="center", va="center", fontsize=36, weight="bold", color=orange)
    if file_path is not None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(file_path, dpi=300, facecolor=background_color)
    return figure


def draw_right_curly_brace(file_path: str | Path | None = None) -> Figure:
    """Draws only one right curly brace on a blank canvas.
    :param file_path: Optional path where the rendered brace image should be saved.
    :return: Matplotlib figure containing only the right curly brace."""
    figure = plt.figure(figsize=(3, 6), facecolor=background_color)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_xlim(0, 300)
    axes.set_ylim(0, 600)
    axes.axis("off")
    axes.set_facecolor(background_color)
    _draw_right_curly_brace(axes, 48, 60, 540, 160, text_color, 3)
    if file_path is not None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(file_path, dpi=300, facecolor=background_color)
    return figure


def _draw_quantum_panel(axes: Axes):
    """Draws the top quantum circuit panel.
    :param axes: Axes receiving the circuit primitives."""
    panel = FancyBboxPatch((127, 585), workflow_width - 254, 480, boxstyle="round,pad=0,rounding_size=22", edgecolor=blue, facecolor="#fbfdff", linewidth=2)
    axes.add_patch(panel)

    composition_y_shift = -11
    qubit_ys = tuple(y + composition_y_shift for y in (936, 836, 736))
    h_width = 52
    rzz_width = 112
    rotation_width = 106
    meter_width = 52
    gate_gap = 27
    composition_shift = -23.25
    rz_x = 855 + composition_shift
    rx_x = rz_x + rotation_width + gate_gap
    rzz_xs = tuple(rz_x - rotation_width // 2 - rzz_width // 2 - gate_gap - (2 - i) * (rzz_width + gate_gap) for i in range(3))
    h_x = rzz_xs[0] - rzz_width // 2 - h_width // 2 - gate_gap
    meter_x = rx_x + rotation_width // 2 + meter_width // 2 + gate_gap
    wire_start = h_x - h_width // 2 - gate_gap
    initial_state_x = wire_start - 37
    for i, y in enumerate(qubit_ys, start=1):
        axes.text(initial_state_x, y, r"$|0\rangle$", ha="center", va="center", fontsize=25, color=text_color)
        axes.plot((wire_start, meter_x + 8), (y, y), color=text_color, linewidth=1.6, zorder=1)
        _draw_gate(axes, h_x, y, h_width, h_width, r"$H$", blue, "#eef5ff", 21)

    _draw_rzz_gate(axes, rzz_xs[0] - rzz_width // 2, qubit_ys[0], qubit_ys[1], r"$R_{ZZ}(\theta_{12})$")
    _draw_rzz_gate(axes, rzz_xs[1] - rzz_width // 2, qubit_ys[1], qubit_ys[2], r"$R_{ZZ}(\theta_{23})$")
    _draw_rzz_gate(axes, rzz_xs[2] - rzz_width // 2, qubit_ys[0], qubit_ys[2], r"$R_{ZZ}(\theta_{13})$")

    for i, y in enumerate(qubit_ys, start=1):
        _draw_gate(axes, rz_x, y, rotation_width, 50, rf"$R_Z(\theta^z_{{{i}}})$", green, "#eef8ea", 17)
        _draw_gate(axes, rx_x, y, rotation_width, 50, rf"$R_X(\theta^x_{{{i}}})$", purple, "#f0edff", 17)
        _draw_meter(axes, meter_x, y)

    _draw_right_curly_brace(axes, meter_x + 33, qubit_ys[-1] - 34, qubit_ys[0] + 34, 32, text_color, 1.7)
    axes.text(meter_x + 73, qubit_ys[1], "Measurements", ha="left", va="center", fontsize=17, color=text_color)


def _draw_exchange_line(axes: Axes):
    """Draws the dashed separator and exchanged variable labels.
    :param axes: Axes receiving the separator primitives."""
    axes.plot((8, 116), (530, 530), color=blue, linewidth=2, linestyle=(0, (4, 4)))
    axes.plot((291, 1197), (530, 530), color=blue, linewidth=2, linestyle=(0, (4, 4)))
    axes.plot((1398, workflow_width - 8), (530, 530), color=blue, linewidth=2, linestyle=(0, (4, 4)))
    axes.text(134, 530, "angles", ha="left", va="center", fontsize=27, weight="bold", color=green)
    axes.text(1215, 530, r"$\bar{u}_1,\;\ldots,\;\bar{u}_M$", ha="left", va="center", fontsize=25, weight="bold", color=blue)


def _draw_cost_panel(axes: Axes):
    """Draws the lower-right cost computation panel.
    :param axes: Axes receiving the cost-panel primitives."""
    panel = FancyBboxPatch((876, 160), 344, 310, boxstyle="round,pad=0,rounding_size=16", edgecolor=orange, facecolor="#fffdf8", linewidth=2)
    axes.add_patch(panel)
    axes.text(1048, 435, "2. Compute Cost", ha="center", va="center", fontsize=25, weight="bold", color=orange)

    _draw_cost_chart(axes, 936, 285, 235, 112)
    axes.text(1049, 213, r"$C(\bar{u}_1)\;\ldots\;C(\bar{u}_M)$", ha="center", va="center", fontsize=26, color=orange)
    axes.text(1048, 76, "Inner Optimization", ha="left", va="center", fontsize=24, weight="bold", color=orange)
    _draw_small_arrow(axes, (1126, 94), (973, 195), orange)
    _draw_small_arrow(axes, (1126, 94), (1121, 195), orange)


def _draw_optimizer_panel(axes: Axes):
    """Draws the lower-left outer optimization panel.
    :param axes: Axes receiving the optimization-panel primitives."""
    panel = FancyBboxPatch((283.4375, 162), 393.125, 310, boxstyle="round,pad=0,rounding_size=16", edgecolor=green, facecolor="#fbfff8", linewidth=2)
    axes.add_patch(panel)
    axes.text(480, 434, "3. Outer Optimization", ha="center", va="center", fontsize=22, weight="bold", color=green)

    _draw_monitor(axes, 369, 225, 225, 172)
    axes.add_patch(Polygon(((462, 225), (496, 225), (502, 190), (456, 190)), closed=True, facecolor="#84b85b", edgecolor=green, linewidth=1.6))
    axes.add_patch(FancyBboxPatch((435, 185), 95, 12, boxstyle="round,pad=0,rounding_size=3", edgecolor=green, facecolor="#a6ca86", linewidth=1.6))


def _draw_workflow_arrows(axes: Axes):
    """Draws the large directional arrows connecting the workflow panels.
    :param axes: Axes receiving the workflow arrows."""
    axes.add_patch(FancyArrowPatch((1398, 760), (1248, 290), connectionstyle="arc3,rad=-0.32", arrowstyle="Simple,head_length=28,head_width=28,tail_width=7",
                                   color=blue, linewidth=0, mutation_scale=1))
    axes.add_patch(FancyArrowPatch((1048, 132), (480, 134), connectionstyle="arc3,rad=-0.22", arrowstyle="Simple,head_length=24,head_width=24,tail_width=5",
                                   color=orange, linewidth=0, mutation_scale=1))
    axes.add_patch(FancyArrowPatch((266, 290), (116, 760), connectionstyle="arc3,rad=-0.32", arrowstyle="Simple,head_length=24,head_width=24,tail_width=6",
                                   color=green, linewidth=0, mutation_scale=1))
    axes.text(753, 108, r"$\bar{F}(C)$", ha="center", va="center", fontsize=23, color=orange)


def _draw_gate(axes: Axes, x: float, y: float, width: float, height: float, label: str, edge_color: str, fill_color: str, font_size: int):
    """Draws one rounded quantum gate.
    :param axes: Axes receiving the gate.
    :param x: Center x-coordinate of the gate.
    :param y: Center y-coordinate of the gate.
    :param width: Gate width.
    :param height: Gate height.
    :param label: Text drawn inside the gate.
    :param edge_color: Gate stroke color.
    :param fill_color: Gate fill color.
    :param font_size: Gate label font size."""
    gate = FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0,rounding_size=6", edgecolor=edge_color,
                          facecolor=fill_color, linewidth=1.8, zorder=3)
    axes.add_patch(gate)
    axes.text(x, y, label, ha="center", va="center", fontsize=font_size, color=text_color, zorder=4)


def _draw_rzz_gate(axes: Axes, x: float, y1: float, y2: float, label: str):
    """Draws one two-qubit ZZ rotation gate.
    :param axes: Axes receiving the gate.
    :param x: Center x-coordinate of the coupling.
    :param y1: Y-coordinate of the first qubit wire.
    :param y2: Y-coordinate of the second qubit wire.
    :param label: Text drawn inside the coupling box."""
    y_top = max(y1, y2)
    y_bottom = min(y1, y2)
    axes.add_patch(Circle((x, y1), 5.5, facecolor=text_color, edgecolor=text_color, zorder=5))
    axes.add_patch(Circle((x, y2), 5.5, facecolor=text_color, edgecolor=text_color, zorder=5))
    gate_padding = 16
    gate = FancyBboxPatch((x, y_bottom - gate_padding), 112, y_top - y_bottom + 2 * gate_padding, boxstyle="round,pad=0,rounding_size=6", edgecolor=rzz_edge,
                          facecolor="#fff1d8", linewidth=1.7, zorder=3)
    axes.add_patch(gate)
    axes.text(x + 56, (y_top + y_bottom) / 2, label, ha="center", va="center", fontsize=16, color=text_color, rotation=0, zorder=4)


def _draw_meter(axes: Axes, x: float, y: float):
    """Draws one measurement meter.
    :param axes: Axes receiving the meter.
    :param x: Center x-coordinate of the meter.
    :param y: Center y-coordinate of the meter."""
    box = FancyBboxPatch((x - 26, y - 25), 52, 50, boxstyle="round,pad=0,rounding_size=6", edgecolor=text_color, facecolor="#ffffff", linewidth=1.8,
                         zorder=3)
    axes.add_patch(box)
    axes.add_patch(Arc((x, y - 9), 34, 34, theta1=0, theta2=180, edgecolor=text_color, linewidth=1.6, zorder=4))
    axes.add_patch(FancyArrowPatch((x - 3, y - 10), (x + 21, y + 17), arrowstyle="-|>", color=text_color, linewidth=1.6, mutation_scale=10, zorder=4))


def _draw_right_curly_brace(axes: Axes, x: float, y_bottom: float, y_top: float, width: float, color: str, linewidth: float):
    """Draws a stroked right curly brace.
    :param axes: Axes receiving the brace.
    :param x: Left x-coordinate of the brace tips.
    :param y_bottom: Lower y-coordinate.
    :param y_top: Upper y-coordinate.
    :param width: Horizontal distance from tips to the middle cusp.
    :param color: Brace stroke color.
    :param linewidth: Brace stroke width."""
    y_mid = (y_top + y_bottom) / 2
    height = y_top - y_bottom
    cap_height = 0.12 * height
    cusp_height = 0.075 * height
    stem_x = x + 0.52 * width
    cusp_x = x + width
    vertices = [(x, y_top), (x + 0.25 * width, y_top), (stem_x, y_top - 0.02 * height), (stem_x, y_top - cap_height),
                (stem_x, y_mid + cusp_height), (stem_x, y_mid + 0.55 * cusp_height), (cusp_x - 0.18 * width, y_mid + 0.12 * cusp_height),
                (cusp_x, y_mid), (cusp_x - 0.18 * width, y_mid - 0.12 * cusp_height), (stem_x, y_mid - 0.55 * cusp_height),
                (stem_x, y_mid - cusp_height), (stem_x, y_bottom + cap_height), (stem_x, y_bottom + 0.02 * height), (x + 0.25 * width, y_bottom),
                (x, y_bottom)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    axes.add_patch(PathPatch(MplPath(vertices, codes), fill=False, edgecolor=color, linewidth=linewidth, capstyle="round", joinstyle="round"))


def _draw_cost_chart(axes: Axes, x: float, y: float, width: float, height: float):
    """Draws the small bitstring cost chart.
    :param axes: Axes receiving the chart.
    :param x: Lower-left chart x-coordinate.
    :param y: Lower-left chart y-coordinate.
    :param width: Chart width.
    :param height: Chart height."""
    values = (0.03, 0.55, 0.25, 0.72, 0.32, 0.28, 0.48, 0.75, 0.08, 0.31, 0.51)
    xs = [x + 18 + i * 18 for i in range(len(values))]
    for xi, value in zip(xs, values, strict=True):
        top = y + 5 + value * (height - 21)
        axes.plot((xi, xi), (y, top), color="#1f78ff", linewidth=3, solid_capstyle="butt", zorder=2)
        axes.add_patch(Circle((xi, top), 5, facecolor="#1f78ff", edgecolor="#1f78ff", zorder=2))
    axes.add_patch(FancyArrowPatch((x, y), (x + width, y), arrowstyle="-|>", color=blue, linewidth=1.5, mutation_scale=14, shrinkA=0, shrinkB=0, zorder=4))
    axes.add_patch(FancyArrowPatch((x, y), (x, y + height), arrowstyle="-|>", color=blue, linewidth=1.5, mutation_scale=14, shrinkA=0, shrinkB=0, zorder=4))
    axes.text(x - 18, y + height * 0.72, r"$C$", ha="right", va="center", fontsize=22, color=text_color)
    axes.text(x + width * 0.56, y - 23, "Bitstrings", ha="center", va="center", fontsize=20, color=text_color)


def _draw_small_arrow(axes: Axes, start: tuple[float, float], end: tuple[float, float], color: str):
    """Draws a small straight annotation arrow.
    :param axes: Axes receiving the arrow.
    :param start: Arrow start coordinates.
    :param end: Arrow end coordinates.
    :param color: Arrow color."""
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", color=color, linewidth=1.5, mutation_scale=15, shrinkA=0))


def _draw_monitor(axes: Axes, x: float, y: float, width: float, height: float):
    """Draws a stylized monitor with a trajectory over contour lines.
    :param axes: Axes receiving the monitor.
    :param x: Lower-left monitor x-coordinate.
    :param y: Lower-left monitor y-coordinate.
    :param width: Monitor width.
    :param height: Monitor height."""
    axes.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=6", edgecolor=green, facecolor="#73a952", linewidth=2))
    screen_margin = 13
    axes.add_patch(Rectangle((x + screen_margin, y + screen_margin + 10), width - 2 * screen_margin, height - 2 * screen_margin - 12,
                             facecolor="#f2f7ec", edgecolor="#d1e5c8", linewidth=1))
    _draw_contours(axes, x + screen_margin, y + screen_margin + 10, width - 2 * screen_margin, height - 2 * screen_margin - 12)
    axes.add_patch(Circle((x + width / 2, y + 11), 3.2, facecolor="#f8fbff", edgecolor="#f8fbff"))


def _draw_contours(axes: Axes, x: float, y: float, width: float, height: float):
    """Draws deterministic contour-like lines and an optimization path.
    :param axes: Axes receiving the contours.
    :param x: Lower-left screen x-coordinate.
    :param y: Lower-left screen y-coordinate.
    :param width: Screen width.
    :param height: Screen height."""
    contour_margin = 14
    contour_angle = -0.58
    raw_contours = []
    for scale, color in zip((1, 0.82, 0.64, 0.47, 0.31), ("#bcc5cd", "#aebbc9", "#9db2cd", "#8aa8d5", "#76a0df"), strict=True):
        vertices = []
        for i in range(128):
            theta = 2 * pi * i / 128
            local_x = scale * 0.37 * width * cos(theta) + 0.06 * scale * width * sin(theta) ** 2
            local_y = scale * 0.27 * height * sin(theta) - 0.035 * scale * height * cos(theta)
            vertices.append((local_x * cos(contour_angle) - local_y * sin(contour_angle),
                             local_x * sin(contour_angle) + local_y * cos(contour_angle)))
        raw_contours.append((vertices, color))

    outer_xs = [point[0] for point in raw_contours[0][0]]
    outer_ys = [point[1] for point in raw_contours[0][0]]
    raw_left = min(outer_xs)
    raw_bottom = min(outer_ys)
    raw_width = max(outer_xs) - raw_left
    raw_height = max(outer_ys) - raw_bottom
    target_left = x + contour_margin
    target_bottom = y + contour_margin
    target_width = width - 2 * contour_margin
    target_height = height - 2 * contour_margin
    contours = []
    for vertices, color in raw_contours:
        contour_vertices = [(target_left + (point[0] - raw_left) * target_width / raw_width,
                             target_bottom + (point[1] - raw_bottom) * target_height / raw_height) for point in vertices]
        contours.append(contour_vertices)
        contour_path = MplPath([*contour_vertices, contour_vertices[0]], [MplPath.MOVETO, *([MplPath.LINETO] * len(contour_vertices))])
        axes.add_patch(PathPatch(contour_path, fill=False, edgecolor=color, linewidth=1.05, zorder=5))

    inner_xs = [point[0] for point in contours[-1]]
    inner_ys = [point[1] for point in contours[-1]]
    star_center = ((min(inner_xs) + max(inner_xs)) / 2, (min(inner_ys) + max(inner_ys)) / 2)
    marker_radius = 5
    star_radius = 10
    star_dip = (star_center[0] + 0.42 * star_radius * cos(11 * pi / 10), star_center[1] + 0.42 * star_radius * sin(11 * pi / 10))
    path_shift = (-5, 4)
    third_point = (star_dip[0] - 12 + path_shift[0], star_dip[1] - 12 + path_shift[1])
    points = ((target_left + 34 + path_shift[0], target_bottom + target_height - 24 + path_shift[1]),
              (target_left + 50 + path_shift[0], target_bottom + target_height - 48 + path_shift[1]), third_point, star_dip)
    for index, (start, end) in enumerate(pairwise(points)):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = hypot(dx, dy)
        end_radius = 1.4 if index == len(points) - 2 else marker_radius
        segment_start = (start[0] + marker_radius * dx / length, start[1] + marker_radius * dy / length)
        segment_end = (end[0] - end_radius * dx / length, end[1] - end_radius * dy / length)
        axes.add_patch(FancyArrowPatch(segment_start, segment_end, arrowstyle="-|>", color=green, linewidth=1.6, mutation_scale=8, shrinkA=0,
                                       shrinkB=0, zorder=7))
    for point in points[:-1]:
        axes.add_patch(Circle(point, marker_radius, facecolor="#f2f7ec", edgecolor=green, linewidth=1.8, zorder=8))
    _draw_star(axes, star_center, star_radius, green)


def _draw_star(axes: Axes, center: tuple[float, float], radius: float, color: str):
    """Draws a five-point optimization target star.
    :param axes: Axes receiving the star.
    :param center: Star center coordinates.
    :param radius: Outer star radius.
    :param color: Star stroke and fill color."""
    vertices = []
    for i in range(10):
        angle = pi / 2 + i * pi / 5
        local_radius = radius if i % 2 == 0 else radius * 0.42
        vertices.append((center[0] + local_radius * cos(angle), center[1] + local_radius * sin(angle)))
    axes.add_patch(Polygon(vertices, closed=True, facecolor="#d8f5cb", edgecolor=color, linewidth=2, zorder=9))


if __name__ == "__main__":
    figure = draw_workflow_outline(Path(__file__).resolve().parent / "graphics" / "workflow_outline.png")
    plt.close(figure)
