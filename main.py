# main_gui.py (Giao diện Pygame - Phiên bản đầy đủ)
import pygame
import sys
import time
import random
from tkinter import ttk
from tkinter import messagebox
# Import tất cả các solvers
from bfs_solver import BFSSolver
from dfs_solver import DFSSolver
from ucs_solver import UCSSolver
from iddfs_solver import IDDFSSolver
from greedy_solver import GreedySolver
from a_star_solver import AStarSolver
from ida_star_solver import IDAStarSolver
from hill_climbing_solver import HillClimbingSolver
from simulated_annealing_solver import SimulatedAnnealingSolver
from beam_search_solver import BeamSearchSolver
from backtracking_solver import BacktrackingSolver
from forward_checking_solver import ForwardCheckingSolver
from ac3_solver import AC3Solver
from genetic_algorithm_solver import GeneticAlgorithmSolver
from q_learning_solver import QLearningSolver

# >>> THÊM IMPORT CHO MATPLOTLIB VÀ TKINTER EMBEDDING <<<
import matplotlib

matplotlib.use('TkAgg')  # Quan trọng: Đặt backend cho matplotlib để dùng với Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk  # Import tkinter
import numpy as np

pygame.init()

# --- Các định nghĩa màu sắc, font, kích thước màn hình (giữ nguyên) ---
WHITE = (255, 255, 255);
BLACK = (0, 0, 0);
LIGHT_GREY = (200, 200, 200)
MEDIUM_GREY = (150, 150, 150);
DARK_GREY = (100, 100, 100);
BLUE = (50, 100, 200)
LIGHT_BLUE = (135, 206, 250);
ORANGE = (255, 165, 0);
GREEN = (0, 128, 0);
RED = (200, 0, 0)

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700  # Initial height, might be adjusted
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("8-Puzzle Solver - AI Algorithms Demo")

TITLE_FONT = pygame.font.Font(None, 36);
LABEL_FONT = pygame.font.Font(None, 28)
BUTTON_FONT = pygame.font.Font(None, 24);
SMALL_BUTTON_FONT = pygame.font.Font(None, 20)
GRID_FONT = pygame.font.Font(None, 50);
OUTPUT_FONT = pygame.font.Font(None, 22)


# --- Các hàm vẽ (draw_text, draw_grid_pygame - giữ nguyên) ---
def draw_text(text, font, color, surface, x, y, center=False):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    if center:
        textrect.center = (x, y)
    else:
        textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


def draw_grid_pygame(surface, x_start, y_start, size, state, label):
    label_x = x_start + (size / 2)
    draw_text(label, LABEL_FONT, BLACK, surface, label_x, y_start - 30, center=True)
    tile_size = size // 3;
    padding = 2
    for i in range(3):
        for j in range(3):
            outer_rect = pygame.Rect(x_start + j * tile_size, y_start + i * tile_size, tile_size, tile_size)
            pygame.draw.rect(surface, DARK_GREY, outer_rect)
            inner_rect = pygame.Rect(x_start + j * tile_size + padding, y_start + i * tile_size + padding,
                                     tile_size - 2 * padding, tile_size - 2 * padding)
            if 0 <= i < len(state) and 0 <= j < len(state[i]):
                tile_value = state[i][j]
                if tile_value != 0:
                    pygame.draw.rect(surface, BLUE, inner_rect, border_radius=3)
                    draw_text(str(tile_value), GRID_FONT, WHITE, surface, inner_rect.centerx, inner_rect.centery,
                              center=True)
                else:
                    pygame.draw.rect(surface, MEDIUM_GREY, inner_rect, border_radius=3)
            else:
                pygame.draw.rect(surface, MEDIUM_GREY, inner_rect, border_radius=3)


# --- Dữ liệu và cấu hình giao diện ---
initial_state_default = [[0, 7, 6], [5, 4, 3], [2, 1, 8]]  # Example of a solvable state
# A slightly harder, but generally solvable state for testing
# initial_state_default = [[8, 6, 7], [2, 5, 4], [3, 0, 1]]
# initial_state_default = [[1, 2, 3], [4, 0, 5], [7, 8, 6]] # Easy one
target_state_default = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
initial_state_current = [row[:] for row in initial_state_default]
target_state_current = [row[:] for row in target_state_default]

algorithm_buttons_config_texts = [
    "BFS", "UCS", "DFS", "IDDFS",
    "Greedy", "A_Star", "IDA*",
    "Simple HC", "Steepest HC", "Stochastic HC",
    "SA", "Genetic Algo", "Beam Search",
    "Backtracking", "Forward Checking",
    "AC-3 (Info)",
    "Q-Learning",
    "Compare Group 1", "Compare Group 2"  # Added "Compare Group 2"
]
selected_algorithm_name = "Compare Group 1"  # Default selection
solution_path_default = ["Solution path or Log will appear here."]
solution_path_current = list(solution_path_default)


def get_output_labels_for_algo(algo_name):
    steps_l = "Steps (g)"
    nodes_l = "Nodes Expanded"
    h_l = "Final Heuristic (h)"
    f_l = "Final Total Cost (f=g+h)"

    if "HC" in algo_name or algo_name == "Beam Search": nodes_l = "Nodes Evaluated"
    if algo_name == "Beam Search":
        h_l = "Best State (h)";
        f_l = "Best State (f=g+h)"
    elif algo_name == "SA":
        nodes_l = "Iterations";
        h_l = "Best State (h)";
        f_l = "Best State (f=g+h)"
    elif algo_name == "Genetic Algo":
        nodes_l = "Generations";
        h_l = "Best Chrom. (h)";
        f_l = "Best Chrom. (f=g+h)"
    elif algo_name == "AC-3 (Info)":
        steps_l = "Revisions";
        nodes_l = "Arcs Processed";
        h_l = "Consistent";
        f_l = "Status"
    elif algo_name == "Q-Learning":
        nodes_l = "Training Episodes"
    elif algo_name == "Compare Group 1" or algo_name == "Compare Group 2":  # Combined for comparison groups
        steps_l = "Path Lengths"
        nodes_l = "Nodes Comparison"
        h_l = "Times Comparison"
        f_l = "(See Plot & Log)"
    return [f"Algorithm : {algo_name}", "Time : -", f"{steps_l} : -", f"{nodes_l} : -", f"{h_l} : -", f"{f_l} : -"]


output_data_current = get_output_labels_for_algo(selected_algorithm_name)

# --- Định nghĩa Rects cho các nút ---
algo_button_width = 120;
algo_button_height = 30
algo_x_start = 500;
algo_y_start = 70
algo_padding_x = 10;
algo_padding_y = 8;
algo_columns = 2  # Keep 2 columns for now, or adjust SCREEN_WIDTH if more are needed side-by-side
num_rows_needed = (len(algorithm_buttons_config_texts) + algo_columns - 1) // algo_columns
required_algo_list_height = num_rows_needed * (algo_button_height + algo_padding_y)
# other_elements_min_height = 70 + 180 + 30 + 40 + 150 + 40 + 40 + 30 # Recalculate based on actual usage
# Recalculate min_total_height_for_ui based on dynamic content later
min_total_height_for_ui = algo_y_start + required_algo_list_height + 250  # Initial estimate, can be refined

if min_total_height_for_ui > SCREEN_HEIGHT:
    SCREEN_HEIGHT = min_total_height_for_ui
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

algorithm_button_definitions = []
current_x_algo = algo_x_start;
current_y_algo = algo_y_start;
col_count_algo = 0
for text in algorithm_buttons_config_texts:
    rect = pygame.Rect(current_x_algo, current_y_algo, algo_button_width, algo_button_height)
    algorithm_button_definitions.append(
        {'text': text, 'rect': rect, 'default_bg_color': LIGHT_BLUE, 'default_text_color': BLACK,
         'selected_bg_color': ORANGE, 'selected_text_color': WHITE, 'font': SMALL_BUTTON_FONT})
    col_count_algo += 1
    if col_count_algo >= algo_columns:
        col_count_algo = 0;
        current_x_algo = algo_x_start
        current_y_algo += algo_button_height + algo_padding_y
    else:
        current_x_algo += algo_button_width + algo_padding_x

main_button_y_offset = 25
last_row_y_algo = algo_y_start
if algorithm_button_definitions:
    if col_count_algo == 0 and len(algorithm_buttons_config_texts) > 0:  # Last button completed a row
        last_row_y_algo = current_y_algo - (algo_button_height + algo_padding_y)
    else:  # Last button is in an incomplete row
        last_row_y_algo = current_y_algo if len(algorithm_buttons_config_texts) > 0 else algo_y_start

run_reset_button_y = last_row_y_algo + \
                     (algo_button_height if col_count_algo != 0 or not algorithm_buttons_config_texts else 0) + \
                     main_button_y_offset
if not algorithm_buttons_config_texts: run_reset_button_y = algo_y_start + main_button_y_offset

run_button_rect_def = pygame.Rect(algo_x_start, run_reset_button_y, algo_button_width, 40)
reset_button_rect_def = pygame.Rect(algo_x_start + algo_button_width + algo_padding_x, run_reset_button_y,
                                    algo_button_width, 40)

q_learning_agent = None
GRID_SIZE_PYGAME = 180;
GRID_Y_START_PYGAME = 80
FIXED_SOLUTION_PATH_X_PYGAME = algo_x_start
SOLUTION_PATH_Y_START_OFFSET_PYGAME = 20
SOLUTION_PATH_Y_START_PYGAME = run_reset_button_y + 40 + SOLUTION_PATH_Y_START_OFFSET_PYGAME
SOLUTION_PATH_RECT_WIDTH_PYGAME = (algo_columns * algo_button_width + (
        algo_columns - 1) * algo_padding_x) if algo_columns > 0 else algo_button_width
MAX_SOLUTION_PATH_HEIGHT_PYGAME = SCREEN_HEIGHT - SOLUTION_PATH_Y_START_PYGAME - 20
if MAX_SOLUTION_PATH_HEIGHT_PYGAME < 100: MAX_SOLUTION_PATH_HEIGHT_PYGAME = 100


def create_comparison_chart_window(comparison_data, group_title="Algorithm Comparison"):
    """Tạo và hiển thị cửa sổ Tkinter với biểu đồ so sánh."""
    try:
        # Tạo cửa sổ Toplevel mới
        # chart_window = tk.Tk() # Creates a new root window. If Pygame is main, Toplevel is better.
        # Check if a root window already exists for Tkinter (e.g. if another Tk window is open)
        try:
            # Try to get the default root; if it doesn't exist, create one.
            root = tk.Toplevel()  # Use Toplevel if Pygame is the main app loop
            # If this is the first Tk window, tk.Tk() might be needed initially.
            # However, for subsequent windows, Toplevel is preferred.
        except tk.TclError:  # No default root
            root = tk.Tk()
            root.withdraw()  # Hide the blank root window if we only want the Toplevel
            chart_window = tk.Toplevel(root)
        else:
            chart_window = tk.Toplevel(root)  # Create Toplevel if root exists

        chart_window.title(group_title)  # Use dynamic title
        # chart_window.geometry("800x900") # Kích thước cửa sổ Tkinter

        # Chuẩn bị dữ liệu
        algo_names = list(comparison_data.keys())
        plot_times = [comparison_data[name].get("time_taken", 0) for name in algo_names]
        plot_nodes = [comparison_data[name].get("nodes_expanded", 0) for name in algo_names]

        raw_path_lengths = [comparison_data[name].get("steps", comparison_data[name].get("g_cost", float('inf'))) for
                            name in algo_names]

        max_reasonable_len = 0
        for length in raw_path_lengths:
            if length != float('inf') and length > max_reasonable_len:
                max_reasonable_len = length
        if max_reasonable_len == 0 and any(
                pl != float('inf') for pl in raw_path_lengths):  # Handle case where all paths are 0
            max_reasonable_len = max(pl for pl in raw_path_lengths if pl != float('inf'))

        capped_path_lengths = []
        path_length_labels = []
        for length in raw_path_lengths:
            if length == float('inf'):
                capped_path_lengths.append(
                    max_reasonable_len + 50 if max_reasonable_len > 0 else 200)  # Giá trị lớn để vẽ
                path_length_labels.append("Fail/Inf")
            else:
                capped_path_lengths.append(length)
                path_length_labels.append(str(int(length)))

        metrics_to_plot = ["Time Taken (s)", "Nodes Expanded", "Path Length"]
        data_for_plot = [plot_times, plot_nodes, capped_path_lengths]
        y_labels_for_plot = [None, None, path_length_labels]  # Nhãn đặc biệt cho Path Length

        # Tạo Figure và các Axes cho matplotlib
        fig_height_per_plot = 2.5 if len(metrics_to_plot) > 1 else 4  # Adjust height
        fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(7, fig_height_per_plot * len(metrics_to_plot) + 1))
        if len(metrics_to_plot) == 1: axs = [axs]

        for i, metric_name in enumerate(metrics_to_plot):
            values = data_for_plot[i]
            # Ensure enough colors if more algos are added
            bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                          '#bcbd22', '#17becf']
            bars = axs[i].bar(algo_names, values, color=bar_colors[:len(algo_names)])
            axs[i].set_ylabel(metric_name)
            axs[i].set_title(f"Comparison: {metric_name}")
            axs[i].tick_params(axis='x', rotation=15, labelsize=8)  # Rotate x-axis labels slightly if names are long

            custom_labels_for_metric = y_labels_for_plot[i]
            for bar_idx, bar in enumerate(bars):
                yval = bar.get_height()
                text_to_display = ""
                if custom_labels_for_metric:
                    text_to_display = custom_labels_for_metric[bar_idx]
                elif isinstance(yval, float) and 0 < yval < 0.0001:
                    text_to_display = f"{yval:.1e}"
                elif isinstance(yval, float):
                    text_to_display = f"{yval:.3f}"
                else:
                    text_to_display = f"{int(yval)}"

                axs[i].text(bar.get_x() + bar.get_width() / 2.0, yval, text_to_display,
                            ha='center', va='bottom' if yval >= 0 else 'top', fontsize=9)

        plt.tight_layout(pad=3.0)

        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        close_button = ttk.Button(chart_window, text="Close Charts", command=chart_window.destroy)
        close_button.pack(side=tk.BOTTOM, pady=10)

        chart_window.mainloop()

    except Exception as e:
        print(f"Error creating Tkinter chart window: {e}")
        messagebox.showerror("Chart Error", f"Could not display comparison chart: {e}")


def compare_group1_algorithms_and_plot(initial_state, goal_state, ui_log_area):
    algo_names_group1 = ["BFS", "UCS", "DFS", "IDDFS"]
    comparison_results = {name: {} for name in algo_names_group1}
    solvers_map = {"BFS": BFSSolver(), "UCS": UCSSolver(), "DFS": DFSSolver(), "IDDFS": IDDFSSolver()}

    ui_log_area[:] = ["--- Starting Group 1 Comparison ---"]
    redraw_static_ui_for_training(ui_log_area)

    max_time_limit = 30  # seconds per algorithm (DFS might need more or be excluded from strict time limit)

    for name in algo_names_group1:
        ui_log_area.append(f"Running {name}...")
        redraw_static_ui_for_training(ui_log_area)
        print(f"Comparing Group 1: Running {name}...")

        solver_instance = solvers_map[name]
        start_run_time = time.time()
        res = {}
        try:
            current_max_time = max_time_limit
            if name == "DFS":  # Potentially very long, give it more time or different handling
                current_max_time = 60  # Allow DFS more time, or set a very high node/depth limit in its solver

            if name == "IDDFS":  # IDDFS solve might take max_iterations
                # Assuming IDDFSSolver has a solve method like others for this comparison.
                # If it requires a specific max_depth, it should be set here.
                # For now, let's assume a generic solve, or a reasonable default within the solver.
                res = solver_instance.solve(initial_state, goal_state,
                                            max_iterations=30)  # Adjust max_iterations as needed
            else:
                res = solver_instance.solve(initial_state, goal_state)  # Add timeout if solver supports it

            run_time = time.time() - start_run_time
            res["time_taken"] = run_time

            if run_time > current_max_time:
                print(f"{name} exceeded time limit of {current_max_time}s. Actual: {run_time:.2f}s")
                res.update({"success": False, "reason": f"Timeout (> {current_max_time}s)",
                            "time_taken": run_time, "nodes_expanded": res.get("nodes_expanded", 0),
                            "steps": float('inf'), "g_cost": float('inf')})

            comparison_results[name] = res
            success_str = "Success" if res.get("success", False) else f"Failed ({res.get('reason', 'Unknown')})"
            path_len_val = res.get("steps", res.get("g_cost", "N/A"))
            path_len_str = str(int(path_len_val)) if isinstance(path_len_val, (int, float)) and path_len_val != float(
                'inf') else ("0" if path_len_val == 0 else "N/A")

            ui_log_area.append(
                f"  {name}: {success_str}, Path: {path_len_str}, Nodes: {res.get('nodes_expanded', 'N/A')}, Time: {res.get('time_taken', 0):.4f}s")
        except Exception as e:
            run_time = time.time() - start_run_time
            print(f"Error running {name} for Group 1 comparison: {e}")
            comparison_results[name] = {"success": False, "reason": str(e), "time_taken": run_time, "nodes_expanded": 0,
                                        "steps": float('inf'), "g_cost": float('inf')}
            ui_log_area.append(f"  {name}: Error - {e}")
        redraw_static_ui_for_training(ui_log_area)

    ui_log_area.append("--- Group 1 Data Collected. Generating Plot... ---")
    redraw_static_ui_for_training(ui_log_area)

    create_comparison_chart_window(comparison_results, "Group 1 Algorithm Comparison")

    ui_log_area.append("--- Group 1 Plot Window Closed ---")
    redraw_static_ui_for_training(ui_log_area)
    return comparison_results


def compare_group2_algorithms_and_plot(initial_state, goal_state, ui_log_area):
    algo_names_group2 = ["Greedy", "A_Star", "IDA*"]
    comparison_results = {name: {} for name in algo_names_group2}
    solvers_map = {"Greedy": GreedySolver(), "A_Star": AStarSolver(), "IDA*": IDAStarSolver()}

    ui_log_area[:] = ["--- Starting Group 2 Comparison ---"]
    redraw_static_ui_for_training(ui_log_area)

    max_time_limit = 45  # seconds per algorithm for informed search (can be demanding)

    for name in algo_names_group2:
        ui_log_area.append(f"Running {name}...")
        redraw_static_ui_for_training(ui_log_area)
        print(f"Comparing Group 2: Running {name}...")

        solver_instance = solvers_map[name]
        start_run_time = time.time()
        res = {}
        try:
            # For IDA*, if it supports max_iterations or a depth limit, it can be passed here.
            # Assuming generic solve for now.
            # Example: if IDAStarSolver().solve supports max_iterations:
            # if name == "IDA*":
            #     res = solver_instance.solve(initial_state, goal_state, max_iterations=100000) # Adjust as needed
            # else:
            #     res = solver_instance.solve(initial_state, goal_state)
            res = solver_instance.solve(initial_state, goal_state)  # Add timeout if solver supports it
            run_time = time.time() - start_run_time
            res["time_taken"] = run_time

            if run_time > max_time_limit:
                print(f"{name} exceeded time limit of {max_time_limit}s. Actual: {run_time:.2f}s")
                res.update({"success": False, "reason": f"Timeout (> {max_time_limit}s)",
                            "time_taken": run_time, "nodes_expanded": res.get("nodes_expanded", 0),
                            "steps": float('inf'), "g_cost": float('inf')})

            comparison_results[name] = res
            success_str = "Success" if res.get("success", False) else f"Failed ({res.get('reason', 'Unknown')})"
            path_len_val = res.get("steps", res.get("g_cost", "N/A"))
            path_len_str = str(int(path_len_val)) if isinstance(path_len_val, (int, float)) and path_len_val != float(
                'inf') else ("0" if path_len_val == 0 else "N/A")

            ui_log_area.append(
                f"  {name}: {success_str}, Path: {path_len_str}, Nodes: {res.get('nodes_expanded', 'N/A')}, Time: {res.get('time_taken', 0):.4f}s")
        except Exception as e:
            run_time = time.time() - start_run_time
            print(f"Error running {name} for Group 2 comparison: {e}")
            comparison_results[name] = {"success": False, "reason": str(e), "time_taken": run_time, "nodes_expanded": 0,
                                        "steps": float('inf'), "g_cost": float('inf')}
            ui_log_area.append(f"  {name}: Error - {e}")
        redraw_static_ui_for_training(ui_log_area)

    ui_log_area.append("--- Group 2 Data Collected. Generating Plot... ---")
    redraw_static_ui_for_training(ui_log_area)

    create_comparison_chart_window(comparison_results, "Group 2 Algorithm Comparison (Informed)")

    ui_log_area.append("--- Group 2 Plot Window Closed ---")
    redraw_static_ui_for_training(ui_log_area)
    return comparison_results


def redraw_static_ui_for_training(training_message_lines):
    screen.fill(LIGHT_GREY)
    draw_text("8-Puzzle Solver - AI Algorithms Demo", TITLE_FONT, BLACK, screen, SCREEN_WIDTH // 2, 25, center=True)

    # Determine if grids should be drawn
    show_grids = selected_algorithm_name not in ["AC-3 (Info)", "Compare Group 1", "Compare Group 2", "Q-Learning"]
    status_message_y_base = GRID_Y_START_PYGAME + GRID_SIZE_PYGAME // 2 - 10

    if selected_algorithm_name == "Q-Learning" and "Training Q-Table" in "".join(training_message_lines):
        show_grids = False  # Don't draw grids during Q-Learning training phase message

    if show_grids:
        draw_grid_pygame(screen, 50, GRID_Y_START_PYGAME, GRID_SIZE_PYGAME, initial_state_current, "Initial State")
        draw_grid_pygame(screen, 300, GRID_Y_START_PYGAME, GRID_SIZE_PYGAME, target_state_current, "Target State")
    elif selected_algorithm_name == "AC-3 (Info)":
        draw_text("AC-3 Demo (CSP not 8-Puzzle Pathfinding)", LABEL_FONT, BLACK, screen, 270, status_message_y_base,
                  center=True)
        draw_text("See Log for CSP details.", LABEL_FONT, BLACK, screen, 270, status_message_y_base + 30, center=True)
    elif selected_algorithm_name == "Compare Group 1":
        draw_text("Group 1 Comparison in Progress...", LABEL_FONT, BLACK, screen, 270, status_message_y_base,
                  center=True)
        draw_text("Check console & new window for plot.", LABEL_FONT, BLACK, screen, 270, status_message_y_base + 30,
                  center=True)
    elif selected_algorithm_name == "Compare Group 2":
        draw_text("Group 2 Comparison in Progress...", LABEL_FONT, BLACK, screen, 270, status_message_y_base,
                  center=True)
        draw_text("Check console & new window for plot.", LABEL_FONT, BLACK, screen, 270, status_message_y_base + 30,
                  center=True)
    elif selected_algorithm_name == "Q-Learning" and not show_grids:  # Specifically for Q-learning training message phase
        draw_text("Q-Learning Training in Progress...", LABEL_FONT, BLACK, screen, 270, status_message_y_base,
                  center=True)
        draw_text("UI may be unresponsive. Check console.", LABEL_FONT, BLACK, screen, 270, status_message_y_base + 30,
                  center=True)

    algorithms_label_x_draw = algo_x_start + (
            algo_columns * algo_button_width + (algo_columns - 1) * algo_padding_x) / 2
    draw_text("Algorithms", LABEL_FONT, BLACK, screen, algorithms_label_x_draw, algo_y_start - 30, center=True)
    for button_def_draw in algorithm_button_definitions:
        bg_c = button_def_draw['default_bg_color'] if button_def_draw['text'] != selected_algorithm_name else \
            button_def_draw['selected_bg_color']
        txt_c = button_def_draw['default_text_color'] if button_def_draw['text'] != selected_algorithm_name else \
            button_def_draw['selected_text_color']
        pygame.draw.rect(screen, bg_c, button_def_draw['rect'], border_radius=5)
        pygame.draw.rect(screen, DARK_GREY, button_def_draw['rect'], width=1, border_radius=5)
        draw_text(button_def_draw['text'], button_def_draw['font'], txt_c, screen, button_def_draw['rect'].centerx,
                  button_def_draw['rect'].centery, center=True)

    temp_output_labels = get_output_labels_for_algo(selected_algorithm_name)

    # Dynamic Y positioning for Output Area
    output_y_base_draw = GRID_Y_START_PYGAME + GRID_SIZE_PYGAME if show_grids else GRID_Y_START_PYGAME + 70
    output_y_start_label_draw = output_y_base_draw + 40
    # Ensure output area doesn't overlap with algorithm buttons if grids are not shown
    min_y_below_algo_buttons = last_row_y_algo + algo_button_height + algo_padding_y + 20  # Min Y for elements below algo list

    # If grids are not shown, and calculated output_y is too high (overlaps algo list), push it down
    if not show_grids and output_y_start_label_draw < min_y_below_algo_buttons:
        output_y_start_label_draw = min_y_below_algo_buttons + 30  # Add some padding

    # Further check to ensure it's below "Run/Reset" buttons if those are lower
    min_y_for_output_area = run_reset_button_y + 40 + 60  # Below run/reset buttons + padding for "Output" title + items
    if output_y_start_label_draw < min_y_for_output_area and not show_grids:
        output_y_start_label_draw = min_y_for_output_area

    # If initial state area is shown, the output area should be clearly below it.
    if show_grids:
        output_y_start_label_draw = GRID_Y_START_PYGAME + GRID_SIZE_PYGAME + 40

    # Final adjustment if still too high relative to algo list (can happen if algo list is very long)
    if output_y_start_label_draw < (last_row_y_algo + algo_button_height + 70) and not show_grids:
        output_y_start_label_draw = last_row_y_algo + algo_button_height + 70

    draw_text("Output", LABEL_FONT, BLACK, screen, 50 + GRID_SIZE_PYGAME / 2 if show_grids else 150,
              output_y_start_label_draw, center=True if show_grids else False)
    output_text_y_start_draw = output_y_start_label_draw + 30
    output_x_pos = 50

    for i_line, line_text_label in enumerate(temp_output_labels):
        current_line_text = output_data_current[i_line] if i_line < len(output_data_current) else line_text_label
        draw_text(current_line_text, OUTPUT_FONT, BLACK, screen, output_x_pos,
                  output_text_y_start_draw + i_line * 25)

    solution_path_label_text_draw = "Log" if selected_algorithm_name in ["AC-3 (Info)", "Compare Group 1",
                                                                         "Compare Group 2",
                                                                         "Q-Learning"] else "Solution Path"
    if selected_algorithm_name == "Q-Learning" and "Training Q-Table" not in "".join(
            training_message_lines) and "Extracting path" not in "".join(training_message_lines):
        if any("Path found" in line for line in training_message_lines) or any(
                "No solution found" in line for line in training_message_lines) or any(
                "Initial state is the Goal" in line for line in training_message_lines):
            solution_path_label_text_draw = "Solution Path"

    draw_text(solution_path_label_text_draw, LABEL_FONT, BLACK, screen,
              FIXED_SOLUTION_PATH_X_PYGAME + SOLUTION_PATH_RECT_WIDTH_PYGAME / 2, SOLUTION_PATH_Y_START_PYGAME - 30,
              center=True)
    temp_solution_rect = pygame.Rect(FIXED_SOLUTION_PATH_X_PYGAME, SOLUTION_PATH_Y_START_PYGAME,
                                     SOLUTION_PATH_RECT_WIDTH_PYGAME, MAX_SOLUTION_PATH_HEIGHT_PYGAME)
    pygame.draw.rect(screen, WHITE, temp_solution_rect);
    pygame.draw.rect(screen, DARK_GREY, temp_solution_rect, width=1)
    max_lines_temp = (MAX_SOLUTION_PATH_HEIGHT_PYGAME - 10) // 20
    for i_msg, msg_line in enumerate(training_message_lines):
        if i_msg < max_lines_temp:
            draw_text(msg_line, OUTPUT_FONT, BLACK, screen, temp_solution_rect.x + 5,
                      temp_solution_rect.y + 5 + i_msg * 20)
        elif i_msg == max_lines_temp:
            draw_text("... (Log truncated)", OUTPUT_FONT, DARK_GREY, screen, temp_solution_rect.x + 5,
                      temp_solution_rect.y + 5 + i_msg * 20);
            break

    run_btn_text_draw = "Run Action"
    if selected_algorithm_name == "AC-3 (Info)":
        run_btn_text_draw = "Run Demo"
    elif selected_algorithm_name in ["Compare Group 1", "Compare Group 2"]:
        run_btn_text_draw = "Run Comparison"
    elif selected_algorithm_name == "Q-Learning":
        run_btn_text_draw = "Train & Run"

    pygame.draw.rect(screen, GREEN, run_button_rect_def, border_radius=5);
    pygame.draw.rect(screen, DARK_GREY, run_button_rect_def, width=1, border_radius=5)
    draw_text(run_btn_text_draw, BUTTON_FONT, WHITE, screen, run_button_rect_def.centerx, run_button_rect_def.centery,
              center=True)
    pygame.draw.rect(screen, RED, reset_button_rect_def, border_radius=5);
    pygame.draw.rect(screen, DARK_GREY, reset_button_rect_def, width=1, border_radius=5)
    draw_text("Reset All", BUTTON_FONT, WHITE, screen, reset_button_rect_def.centerx, reset_button_rect_def.centery,
              center=True)
    pygame.display.flip()


# --- Vòng lặp chính của game ---
running = True
# Attempt to create a hidden Tk root window once, if not already done by matplotlib
# This helps in managing Toplevel windows later.
try:
    tk_root_instance = tk.Tk()
    tk_root_instance.withdraw()  # Hide it, we only need it for Toplevel context
except Exception as e:
    print(f"Could not initialize main Tk root (might be fine if matplotlib did): {e}")
    tk_root_instance = None  # In case of issues, proceed without it. create_comparison_chart_window has fallbacks.

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_pos = event.pos
                # Algorithm selection buttons
                for button_def in algorithm_button_definitions:
                    if button_def['rect'].collidepoint(mouse_pos):
                        selected_algorithm_name = button_def['text']
                        output_data_current = get_output_labels_for_algo(selected_algorithm_name)
                        solution_path_current = ["Select an algorithm and click 'Run'."]
                        if selected_algorithm_name in ["AC-3 (Info)", "Compare Group 1", "Compare Group 2",
                                                       "Q-Learning"]:
                            action = "Run Demo" if selected_algorithm_name == "AC-3 (Info)" else \
                                "Run Comparison" if "Compare" in selected_algorithm_name else "Train & Run"
                            solution_path_current = [f"Click '{action}' to start."]
                        print(f"Algorithm selected: {selected_algorithm_name}")
                        # No break here, redraw will happen after event processing
                        # break # Found button, no need to check others

                # Run button
                if run_button_rect_def.collidepoint(mouse_pos):
                    run_button_text_content = "Run Action"
                    if selected_algorithm_name == "AC-3 (Info)":
                        run_button_text_content = "Run Demo"
                    elif selected_algorithm_name in ["Compare Group 1", "Compare Group 2"]:
                        run_button_text_content = "Run Comparison"
                    elif selected_algorithm_name == "Q-Learning":
                        run_button_text_content = "Train & Run"

                    print(f"'{run_button_text_content}' clicked for {selected_algorithm_name}!")

                    original_solution_path_message = [f"Processing {selected_algorithm_name}..."]
                    solution_path_current = list(original_solution_path_message)

                    if selected_algorithm_name == "Q-Learning":
                        solution_path_current.append("Preparing for Q-Table training...")
                    elif selected_algorithm_name == "Compare Group 1":
                        solution_path_current = ["Starting Group 1 Algorithm Comparison..."]
                    elif selected_algorithm_name == "Compare Group 2":
                        solution_path_current = ["Starting Group 2 Algorithm Comparison..."]

                    redraw_static_ui_for_training(solution_path_current)  # Initial "Processing" message

                    results = None
                    initial_for_solver = [row[:] for row in initial_state_current]
                    target_for_solver = [row[:] for row in target_state_default]  # Always use default target

                    current_labels_dict = {
                        "steps": output_data_current[2].split(':')[0].strip(),
                        "nodes": output_data_current[3].split(':')[0].strip(),
                        "h": output_data_current[4].split(':')[0].strip(),
                        "f": output_data_current[5].split(':')[0].strip()
                    }
                    solver = None  # Reset solver

                    # --- LOGIC THỰC THI THUẬT TOÁN ---
                    if selected_algorithm_name == "Compare Group 1":
                        # solution_path_current is passed and modified by the function
                        comparison_data = compare_group1_algorithms_and_plot(initial_for_solver, target_for_solver,
                                                                             solution_path_current)
                        output_data_current = get_output_labels_for_algo(selected_algorithm_name)
                        output_data_current[0] = f"Algorithm : Group 1 Comparison"
                        output_data_current[1] = "Time : (See Plot/Log)"
                        results = {"success": True, "reason": "Group 1 Comparison complete."}

                    elif selected_algorithm_name == "Compare Group 2":
                        # solution_path_current is passed and modified by the function
                        comparison_data = compare_group2_algorithms_and_plot(initial_for_solver, target_for_solver,
                                                                             solution_path_current)
                        output_data_current = get_output_labels_for_algo(selected_algorithm_name)
                        output_data_current[0] = f"Algorithm : Group 2 Comparison"
                        output_data_current[1] = "Time : (See Plot/Log)"
                        results = {"success": True, "reason": "Group 2 Comparison complete."}

                    elif selected_algorithm_name == "BFS":
                        solver = BFSSolver()
                    elif selected_algorithm_name == "DFS":
                        solver = DFSSolver()
                    elif selected_algorithm_name == "UCS":
                        solver = UCSSolver()
                    elif selected_algorithm_name == "IDDFS":
                        solver = IDDFSSolver()  # Expects solve(i,g, max_iter)
                    elif selected_algorithm_name == "Greedy":
                        solver = GreedySolver()
                    elif selected_algorithm_name == "A_Star":
                        solver = AStarSolver()
                    elif selected_algorithm_name == "IDA*":
                        solver = IDAStarSolver()
                    elif selected_algorithm_name == "Simple HC":
                        solver = HillClimbingSolver(); results = solver.solve_simple_hc(initial_for_solver,
                                                                                        target_for_solver)
                    elif selected_algorithm_name == "Steepest HC":
                        solver = HillClimbingSolver(); results = solver.solve_steepest_ascent_hc(initial_for_solver,
                                                                                                 target_for_solver)
                    elif selected_algorithm_name == "Stochastic HC":
                        solver = HillClimbingSolver(); results = solver.solve_stochastic_hc(initial_for_solver,
                                                                                            target_for_solver)
                    elif selected_algorithm_name == "SA":
                        solver = SimulatedAnnealingSolver()  # Expects solve(i,g, temp, rate)
                    elif selected_algorithm_name == "Genetic Algo":
                        solver = GeneticAlgorithmSolver()  # Expects solve(i,g, pop_size, gens, mut_rate)
                    elif selected_algorithm_name == "Beam Search":
                        solver = BeamSearchSolver()  # Expects solve(i,g, beam_width)
                    elif selected_algorithm_name == "Backtracking":
                        solver = BacktrackingSolver()
                    elif selected_algorithm_name == "Forward Checking":
                        solver = ForwardCheckingSolver()
                    elif selected_algorithm_name == "AC-3 (Info)":
                        ac3_variables = ["X", "Y", "Z"];
                        ac3_domains = {"X": [1, 2, 3], "Y": [1, 2, 3], "Z": [1, 2]}


                        def x_neq_y(val_x, val_y):
                            return val_x != val_y


                        def y_neq_z(val_y, val_z):
                            return val_y != val_z


                        def x_lt_z(val_x, val_z):
                            return val_x < val_z


                        ac3_constraints = [("X", "Y", x_neq_y), ("Y", "X", x_neq_y), ("Y", "Z", y_neq_z),
                                           ("Z", "Y", y_neq_z), ("X", "Z", x_lt_z), ("Z", "X", lambda vz, vx: vz > vx)]
                        solver = AC3Solver();
                        results = solver.solve(ac3_variables, ac3_domains, ac3_constraints)
                    elif selected_algorithm_name == "Q-Learning":
                        if q_learning_agent is None or True:  # Re-init for fresh training or params change
                            q_learning_agent = QLearningSolver(epsilon_decay_rate=0.9998, learning_rate=0.1,
                                                               discount_factor=0.9)
                        num_train_episodes = 15000;
                        max_s_ep = 200  # Reduced for faster demo
                        print(
                            f"Q-Learning: Training for {num_train_episodes} episodes, max_steps={max_s_ep}. UI may freeze.")
                        temp_training_messages = list(original_solution_path_message)
                        temp_training_messages.append(f"Training Q-Table ({num_train_episodes} episodes)...")
                        temp_training_messages.append("This may take significant time. UI unresponsive.")
                        redraw_static_ui_for_training(temp_training_messages)  # Show training message

                        q_learning_agent.train(initial_for_solver, target_for_solver, num_episodes=num_train_episodes,
                                               max_steps_per_episode=max_s_ep)

                        temp_training_messages = list(original_solution_path_message)  # Reset messages
                        temp_training_messages.append("Training complete. Extracting path...")
                        solution_path_current = temp_training_messages  # Update main log area
                        redraw_static_ui_for_training(solution_path_current)  # Show extraction message
                        results = q_learning_agent.get_optimal_path(initial_for_solver, target_for_solver)
                        results["nodes_expanded"] = num_train_episodes  # For display
                    else:  # Fallback for unimplemented or error
                        if not results and selected_algorithm_name not in ["Compare Group 1", "Compare Group 2"]:
                            solution_path_current = [f"{selected_algorithm_name} not implemented or error."]
                            output_data_current = get_output_labels_for_algo(
                                selected_algorithm_name)  # Reset output labels
                            results = {"success": False, "reason": "Not implemented"}

                    # Generic solver execution for those initialized into 'solver' variable
                    if solver and selected_algorithm_name not in ["Simple HC", "Steepest HC", "Stochastic HC",
                                                                  "AC-3 (Info)", "Q-Learning", "Compare Group 1",
                                                                  "Compare Group 2"]:
                        start_time_solve = time.time()
                        try:
                            if selected_algorithm_name == "IDDFS":
                                results = solver.solve(initial_for_solver, target_for_solver,
                                                       max_iterations=30)  # Adjust iterations
                            elif selected_algorithm_name == "SA":  # SimulatedAnnealingSolver
                                results = solver.solve(initial_for_solver, target_for_solver, initial_temp=1000,
                                                       cooling_rate=0.99)
                            elif selected_algorithm_name == "Genetic Algo":
                                results = solver.solve(initial_for_solver, target_for_solver, population_size=50,
                                                       generations=100, mutation_rate=0.1)
                            elif selected_algorithm_name == "Beam Search":
                                results = solver.solve(initial_for_solver, target_for_solver, beam_width=3)
                            else:  # For BFS, DFS, UCS, Greedy, A*, IDA*, Backtracking, Forward Checking
                                results = solver.solve(initial_for_solver, target_for_solver)
                            results['time_taken'] = time.time() - start_time_solve
                        except Exception as e:
                            results = {"success": False, "reason": str(e), "time_taken": time.time() - start_time_solve,
                                       "nodes_expanded": 0, "steps": float('inf')}
                            print(f"Error during {selected_algorithm_name}.solve(): {e}")

                    # --- CẬP NHẬT GIAO DIỆN SAU KHI CHẠY THUẬT TOÁN (TRỪ CÁC GROUP SO SÁNH) ---
                    if results and selected_algorithm_name not in ["Compare Group 1", "Compare Group 2"]:
                        output_data_current[0] = f"Algorithm : {selected_algorithm_name}"
                        output_data_current[1] = f"Time : {results.get('time_taken', 0.0):.4f} s"
                        output_data_current[
                            3] = f"{current_labels_dict['nodes']} : {results.get('nodes_expanded', 'N/A')}"

                        # Clear previous solution path unless it's a Q-Learning continuation message
                        if not (
                                selected_algorithm_name == "Q-Learning" and solution_path_current and "Training complete" in
                                solution_path_current[-1]):
                            solution_path_current = []  # Start fresh for displaying results

                        if results.get("success", False):
                            g_cost_val = results.get('g_cost', results.get('steps', "N/A"))
                            h_cost_val = results.get('h_cost', "N/A")  # Might be N/A for non-heuristic algos or if goal

                            path_msg_prefix = "Path found"
                            if selected_algorithm_name == "Q-Learning":
                                path_msg_prefix = f"Optimal path found (after {results.get('nodes_expanded', 'N/A')} episodes)"

                            steps_val = results.get('steps', "N/A")
                            path_message = f"{path_msg_prefix} in {steps_val} steps"

                            if not results.get("path_actions") and results.get('steps',
                                                                               0) == 0 and initial_for_solver == target_for_solver:
                                path_message = "Initial state is the Goal state."
                            elif "reason" in results and results.get('reason') and results[
                                'reason'] != "Max iterations reached without finding goal for IDA*":  # IDA* reason is handled by path length
                                path_message += f" ({results['reason']})"
                            else:
                                path_message += "."
                            solution_path_current.append(path_message)

                            for i, action in enumerate(results.get("path_actions", [])):
                                if i < 15:
                                    solution_path_current.append(f"Step {i + 1}: {action}")
                                elif i == 15:
                                    solution_path_current.append("... (path truncated)"); break

                            output_data_current[2] = f"{current_labels_dict['steps']} : {g_cost_val}"
                            output_data_current[4] = f"{current_labels_dict['h']} : {h_cost_val}"
                            if isinstance(g_cost_val, (int, float)) and isinstance(h_cost_val, (int, float)):
                                output_data_current[5] = f"{current_labels_dict['f']} : {g_cost_val + h_cost_val}"
                            elif isinstance(g_cost_val, (int, float)):  # Only g_cost is valid (e.g. BFS, DFS)
                                output_data_current[5] = f"{current_labels_dict['f']} : {g_cost_val}"
                            else:
                                output_data_current[5] = f"{current_labels_dict['f']} : N/A"
                        else:  # No success
                            solution_path_current.append(
                                f"No solution found or process ended for {selected_algorithm_name}.")
                            if "reason" in results: solution_path_current.append(f"Reason: {results['reason']}")

                            output_data_current[
                                2] = f"{current_labels_dict['steps']} : {results.get('g_cost', results.get('steps', 'N/A'))}"
                            output_data_current[
                                4] = f"{current_labels_dict['h']} : {results.get('h_cost', 'N/A')}"  # Best h if available

                            g_val_disp = results.get('g_cost', results.get('steps'))
                            h_val_disp = results.get('h_cost')

                            if isinstance(g_val_disp, (int, float)) and isinstance(h_val_disp, (int, float)) and \
                                    (
                                            "HC" in selected_algorithm_name or selected_algorithm_name == "SA" or selected_algorithm_name == "Beam Search"):
                                output_data_current[
                                    5] = f"{current_labels_dict['f']} : {g_val_disp + h_val_disp} (Best State)"
                            elif "Genetic Algo" == selected_algorithm_name and isinstance(results.get('f_cost'),
                                                                                          (int, float)):
                                output_data_current[
                                    5] = f"{current_labels_dict['f']} : {results.get('f_cost')} (Best Chrom.)"
                            else:
                                output_data_current[5] = f"{current_labels_dict['f']} : N/A"

                # Reset button
                elif reset_button_rect_def.collidepoint(mouse_pos):
                    print("Reset All clicked!")
                    initial_state_current = [row[:] for row in initial_state_default]
                    # target_state_current = [row[:] for row in target_state_default] # Target is fixed
                    solution_path_current = list(solution_path_default)
                    output_data_current = get_output_labels_for_algo(
                        selected_algorithm_name)  # Reset based on current algo
                    q_learning_agent = None  # Reset Q-learning agent too
                    print("Game reset.")

    # --- Vẽ lên màn hình (Main Drawing Loop) ---
    screen.fill(LIGHT_GREY)
    draw_text("8-Puzzle Solver - AI Algorithms Demo", TITLE_FONT, BLACK, screen, SCREEN_WIDTH // 2, 25, center=True)

    # Conditional drawing of grids
    should_draw_grids_main = selected_algorithm_name not in ["AC-3 (Info)", "Compare Group 1", "Compare Group 2"]
    if selected_algorithm_name == "Q-Learning":  # For QL, only hide grids if actively showing training messages in log
        if any("Training Q-Table" in line for line in solution_path_current):
            should_draw_grids_main = False

    current_status_message_y_main = GRID_Y_START_PYGAME + GRID_SIZE_PYGAME // 2 - 10

    if should_draw_grids_main:
        draw_grid_pygame(screen, 50, GRID_Y_START_PYGAME, GRID_SIZE_PYGAME, initial_state_current, "Initial State")
        draw_grid_pygame(screen, 300, GRID_Y_START_PYGAME, GRID_SIZE_PYGAME, target_state_current, "Target State")
    elif selected_algorithm_name == "AC-3 (Info)":
        draw_text("AC-3 Demo (CSP)", LABEL_FONT, BLACK, screen, 270, current_status_message_y_main, center=True)
        draw_text("See Log for details.", LABEL_FONT, BLACK, screen, 270, current_status_message_y_main + 30,
                  center=True)
    elif selected_algorithm_name == "Compare Group 1":
        draw_text("Group 1 Comparison", LABEL_FONT, BLACK, screen, 270, current_status_message_y_main, center=True)
        draw_text("Check Log. Plot window may appear.", LABEL_FONT, BLACK, screen, 270,
                  current_status_message_y_main + 30, center=True)
    elif selected_algorithm_name == "Compare Group 2":
        draw_text("Group 2 Comparison", LABEL_FONT, BLACK, screen, 270, current_status_message_y_main, center=True)
        draw_text("Check Log. Plot window may appear.", LABEL_FONT, BLACK, screen, 270,
                  current_status_message_y_main + 30, center=True)
    elif selected_algorithm_name == "Q-Learning" and not should_draw_grids_main:  # QL Training Message
        draw_text("Q-Learning Training...", LABEL_FONT, BLACK, screen, 270, current_status_message_y_main, center=True)
        draw_text("Check console. UI may be unresponsive.", LABEL_FONT, BLACK, screen, 270,
                  current_status_message_y_main + 30, center=True)

    algorithms_label_x_main_draw = algo_x_start + (
            algo_columns * algo_button_width + (algo_columns - 1) * algo_padding_x) / 2
    draw_text("Algorithms", LABEL_FONT, BLACK, screen, algorithms_label_x_main_draw, algo_y_start - 30, center=True)
    for button_def in algorithm_button_definitions:
        bg_color = button_def['default_bg_color'];
        text_color = button_def['default_text_color']
        font_to_use = button_def.get('font', BUTTON_FONT)  # Fallback to BUTTON_FONT
        if button_def['text'] == selected_algorithm_name:
            bg_color = button_def['selected_bg_color'];
            text_color = button_def['selected_text_color']
        pygame.draw.rect(screen, bg_color, button_def['rect'], border_radius=5)
        pygame.draw.rect(screen, DARK_GREY, button_def['rect'], width=1, border_radius=5)
        draw_text(button_def['text'], font_to_use, text_color, screen, button_def['rect'].centerx,
                  button_def['rect'].centery, center=True)

    # Solution Path / Log Area
    log_label_text_main = "Log"
    if selected_algorithm_name not in ["AC-3 (Info)", "Compare Group 1", "Compare Group 2"]:
        if selected_algorithm_name == "Q-Learning":
            # Check if QL is done training and showing path/no solution
            if not any("Training Q-Table" in line for line in solution_path_current) and \
                    not any("Extracting path..." in line for line in solution_path_current) and \
                    (any("Path found" in line for line in solution_path_current) or \
                     any("No solution found" in line for line in solution_path_current) or \
                     any("Initial state is the Goal" in line for line in solution_path_current)):
                log_label_text_main = "Solution Path"
        else:
            log_label_text_main = "Solution Path"

    draw_text(log_label_text_main, LABEL_FONT, BLACK, screen,
              FIXED_SOLUTION_PATH_X_PYGAME + SOLUTION_PATH_RECT_WIDTH_PYGAME / 2, SOLUTION_PATH_Y_START_PYGAME - 30,
              center=True)
    solution_rect_main_draw = pygame.Rect(FIXED_SOLUTION_PATH_X_PYGAME, SOLUTION_PATH_Y_START_PYGAME,
                                          SOLUTION_PATH_RECT_WIDTH_PYGAME, MAX_SOLUTION_PATH_HEIGHT_PYGAME)
    pygame.draw.rect(screen, WHITE, solution_rect_main_draw);
    pygame.draw.rect(screen, DARK_GREY, solution_rect_main_draw, width=1)
    max_lines_main_draw = (MAX_SOLUTION_PATH_HEIGHT_PYGAME - 10) // 20
    for i, line in enumerate(solution_path_current):
        if i < max_lines_main_draw:
            draw_text(line, OUTPUT_FONT, BLACK, screen, solution_rect_main_draw.x + 5,
                      solution_rect_main_draw.y + 5 + i * 20)
        elif i == max_lines_main_draw:
            draw_text("... (Log truncated)", OUTPUT_FONT, DARK_GREY, screen, solution_rect_main_draw.x + 5,
                      solution_rect_main_draw.y + 5 + i * 20);
            break

    # Output Data Area
    # Adjust Y based on whether grids are shown, similar to redraw_static_ui_for_training
    output_y_base_main_draw = GRID_Y_START_PYGAME + GRID_SIZE_PYGAME if should_draw_grids_main else GRID_Y_START_PYGAME + 70
    output_y_start_label_main_draw = output_y_base_main_draw + 40

    min_y_below_algo_main = last_row_y_algo + algo_button_height + algo_padding_y + 20
    if not should_draw_grids_main and output_y_start_label_main_draw < min_y_below_algo_main:
        output_y_start_label_main_draw = min_y_below_algo_main + 30

    min_y_for_output_main = run_reset_button_y + 40 + 60  # Below run/reset buttons + padding
    if output_y_start_label_main_draw < min_y_for_output_main and not should_draw_grids_main:
        output_y_start_label_main_draw = min_y_for_output_main

    if should_draw_grids_main:
        output_y_start_label_main_draw = GRID_Y_START_PYGAME + GRID_SIZE_PYGAME + 40  # Standard position when grids are shown

    if output_y_start_label_main_draw < (last_row_y_algo + algo_button_height + 70) and not should_draw_grids_main:
        output_y_start_label_main_draw = last_row_y_algo + algo_button_height + 70

    output_label_x_main = 50 + GRID_SIZE_PYGAME / 2 if should_draw_grids_main else 150
    draw_text("Output", LABEL_FONT, BLACK, screen, output_label_x_main, output_y_start_label_main_draw,
              center=True if should_draw_grids_main else False)
    output_text_y_start_main_draw = output_y_start_label_main_draw + 30;
    output_text_x_start_main_draw = 50
    for i, line in enumerate(output_data_current):
        draw_text(line, OUTPUT_FONT, BLACK, screen, output_text_x_start_main_draw,
                  output_text_y_start_main_draw + i * 25)

    # Run/Reset Buttons
    run_button_text_main = "Run Action"
    if selected_algorithm_name == "AC-3 (Info)":
        run_button_text_main = "Run Demo"
    elif selected_algorithm_name in ["Compare Group 1", "Compare Group 2"]:
        run_button_text_main = "Run Comparison"
    elif selected_algorithm_name == "Q-Learning":
        run_button_text_main = "Train & Run"

    pygame.draw.rect(screen, GREEN, run_button_rect_def, border_radius=5)
    pygame.draw.rect(screen, DARK_GREY, run_button_rect_def, width=1, border_radius=5)
    draw_text(run_button_text_main, BUTTON_FONT, WHITE, screen, run_button_rect_def.centerx,
              run_button_rect_def.centery, center=True)
    pygame.draw.rect(screen, RED, reset_button_rect_def, border_radius=5)
    pygame.draw.rect(screen, DARK_GREY, reset_button_rect_def, width=1, border_radius=5)
    draw_text("Reset All", BUTTON_FONT, WHITE, screen, reset_button_rect_def.centerx, reset_button_rect_def.centery,
              center=True)

    pygame.display.flip()

if tk_root_instance:  # Clean up the hidden Tk root window if it was created
    try:
        tk_root_instance.destroy()
    except tk.TclError:
        pass  # Window might have already been destroyed (e.g. by matplotlib or chart window)
pygame.quit()
sys.exit()