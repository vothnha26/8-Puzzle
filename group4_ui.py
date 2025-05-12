# group4_ui.py
import random
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from and_or_search_solver import AndOrPuzzleSolver
from belief_search_demo_solver import BeliefStatePuzzleDemo
from partially_observable_demo_solver import PartiallyObservableDemo  # Đã import
import random


class Group4UI:
    def __init__(self, master):
        self.master = master
        master.title("Group 4: Search in Complex Environments")
        master.geometry("900x780")

        self.fixed_cells_highlight_ids = {}
        self.initial_state_default = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]
        self.target_state_default = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.current_initial_state_to_draw = [row[:] for row in self.initial_state_default]
        self.current_final_state_to_draw = [row[:] for row in self.target_state_default]
        self.active_fixed_config = {}

        self.selected_algorithm_var = tk.StringVar()
        self.algorithms = [
            "AND-OR Tree Search (8-Puzzle)",
            "Partially Observable Search (Belief Space Demo)",
            "Unknown/Dynamic Environment Search (Path Demo)"
        ]
        if self.algorithms:
            self.selected_algorithm_var.set(self.algorithms[1])  # Mặc định chọn Partially Observable

        self.top_frame = ttk.Frame(master, padding=10)
        self.top_frame.pack(fill=tk.X, padx=10, pady=5)
        self.control_frame = ttk.LabelFrame(self.top_frame, text="Algorithm Selection & Controls", padding=10)
        self.control_frame.pack(fill=tk.X)
        self.puzzle_display_frame = ttk.LabelFrame(master, text="8-Puzzle Visualization", padding=10)
        self.puzzle_display_frame.pack(padx=10, pady=(5, 0), fill=tk.X)
        self.display_frame = ttk.LabelFrame(master, text="Algorithm Output & Results", padding=10)
        self.display_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        algo_label = ttk.Label(self.control_frame, text="Select Algorithm:")
        algo_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.algo_combobox = ttk.Combobox(self.control_frame, textvariable=self.selected_algorithm_var,
                                          values=self.algorithms, state="readonly", width=45)
        if self.algorithms:
            self.algo_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.control_frame.grid_columnconfigure(1, weight=1)

        self.run_button_text_var = tk.StringVar()
        self.run_button = ttk.Button(self.control_frame, textvariable=self.run_button_text_var,
                                     command=self.run_selected_algorithm)
        self.run_button.grid(row=1, column=0, padx=5, pady=10, columnspan=1)
        self.algo_combobox.bind("<<ComboboxSelected>>", self.update_run_button_text)

        self.clear_button = ttk.Button(self.control_frame, text="Clear Output & Reset Grids", command=self.clear_output)
        self.clear_button.grid(row=1, column=1, padx=5, pady=10, sticky=tk.W)
        self.update_run_button_text()

        self.canvas_frame = ttk.Frame(self.puzzle_display_frame)
        self.canvas_frame.pack()

        self.label_canvas1 = ttk.Label(self.canvas_frame, text="Initial / Sample 1")
        self.label_canvas1.grid(row=0, column=0, pady=(0, 2))
        self.canvas1 = tk.Canvas(self.canvas_frame, width=150, height=150, bg="#f0f0f0")
        self.canvas1.grid(row=1, column=0, padx=20, pady=5)

        self.label_canvas2 = ttk.Label(self.canvas_frame, text="Target / Final Sample")
        self.label_canvas2.grid(row=0, column=1, pady=(0, 2))
        self.canvas2 = tk.Canvas(self.canvas_frame, width=150, height=150, bg="#f0f0f0")
        self.canvas2.grid(row=1, column=1, padx=20, pady=5)

        self.puzzle_display_frame.grid_columnconfigure(0, weight=1)
        self.puzzle_display_frame.grid_columnconfigure(1, weight=1)

        self.draw_tkinter_grid(self.canvas1, self.current_initial_state_to_draw, {})
        self.draw_tkinter_grid(self.canvas2, self.current_final_state_to_draw, {}, is_target=True)

        self.output_text = scrolledtext.ScrolledText(self.display_frame, wrap=tk.WORD, width=90, height=15,
                                                     font=("Arial", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self._configure_output_text(tk.DISABLED)

        menubar = tk.Menu(master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=master.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        master.config(menu=menubar)

        self.output_text.tag_configure('header', font=('Arial', 12, 'bold'))
        self.output_text.tag_configure('subheader', font=('Arial', 11, 'bold', 'underline'))
        self.output_text.tag_configure('footer', font=('Arial', 10, 'italic'))
        self.output_text.tag_configure('result_success', foreground='dark green', font=('Arial', 10, 'bold'))
        self.output_text.tag_configure('result_fail', foreground='red', font=('Arial', 10, 'bold'))
        self.output_text.tag_configure('info', foreground='blue')
        self.output_text.tag_configure('label', font=('Arial', 10, 'bold'))

    def update_run_button_text(self, event=None):
        algo = self.selected_algorithm_var.get()
        if algo == "AND-OR Tree Search (8-Puzzle)":
            self.run_button_text_var.set("Run 8-Puzzle AND-OR")
        elif algo == "Unknown/Dynamic Environment Search (Path Demo)":
            self.run_button_text_var.set("Run Path Demo")
        elif algo == "Partially Observable Search (Belief Space Demo)":
            self.run_button_text_var.set("Run Belief Space Demo")
        else:
            self.run_button_text_var.set("Run Conceptual Demo")

    def draw_tkinter_grid(self, canvas, state_matrix, fixed_config, is_target=False):
        canvas.delete("all")
        if not state_matrix or not isinstance(state_matrix, list) or not all(
                isinstance(row, list) for row in state_matrix) or len(state_matrix) != 3:
            canvas.create_text(75, 75, text="N/A", font=("Arial", 16))
            return
        for row in state_matrix:
            if not isinstance(row, list) or len(row) != 3:
                canvas.create_text(75, 75, text="N/A", font=("Arial", 16))
                return
        grid_size = 150;
        tile_size = grid_size // 3;
        padding = 2
        color_tile_bg = "#4a7aBa";
        color_tile_text = "white"
        color_empty_bg = "#d3d3d3";
        color_border = "#303030"
        color_fixed_highlight = "#FFD700";
        color_fixed_text = "black"
        for r in range(3):
            for c in range(3):
                x0, y0 = c * tile_size, r * tile_size;
                x1, y1 = x0 + tile_size, y0 + tile_size
                canvas.create_rectangle(x0, y0, x1, y1, fill=color_border, outline=color_border)
                inner_x0, inner_y0 = x0 + padding, y0 + padding;
                inner_x1, inner_y1 = x1 - padding, y1 - padding
                tile_value = state_matrix[r][c]
                is_fixed_cell_and_value_matches = False
                if fixed_config and (r, c) in fixed_config and fixed_config[(r, c)] == tile_value:
                    is_fixed_cell_and_value_matches = True
                current_fill = color_tile_bg;
                current_text_color = color_tile_text
                if tile_value == 0:
                    current_fill = color_empty_bg
                elif is_fixed_cell_and_value_matches:
                    current_fill = color_fixed_highlight; current_text_color = color_fixed_text
                canvas.create_rectangle(inner_x0, inner_y0, inner_x1, inner_y1, fill=current_fill, outline=current_fill,
                                        width=0)
                if tile_value != 0:  # Vẽ số nếu không phải ô trống
                    canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(tile_value), font=("Arial", 24, "bold"),
                                       fill=current_text_color)

    def _configure_output_text(self, state):
        self.output_text.configure(state=state)

    def add_log_message(self, message, tags=None):
        self._configure_output_text(tk.NORMAL)
        if tags:
            self.output_text.insert(tk.END, message + "\n", tags)
        else:
            self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self._configure_output_text(tk.DISABLED)

    def clear_output(self):
        self._configure_output_text(tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self._configure_output_text(tk.DISABLED)
        self.current_initial_state_to_draw = [row[:] for row in self.initial_state_default]
        self.current_final_state_to_draw = [row[:] for row in self.target_state_default]
        self.active_fixed_config = {}
        self.draw_tkinter_grid(self.canvas1, self.current_initial_state_to_draw, self.active_fixed_config)
        self.draw_tkinter_grid(self.canvas2, self.current_final_state_to_draw, self.active_fixed_config, is_target=True)
        self.label_canvas1.config(text="Initial State")
        self.label_canvas2.config(text="Target State")

    def run_selected_algorithm(self):
        self.clear_output()
        algo = self.selected_algorithm_var.get()

        self.add_log_message(f"--- Running: {algo} ---", ('header',))
        results = None
        initial_for_8puzzle_solver = [row[:] for row in self.current_initial_state_to_draw]
        target_for_8puzzle_solver = [row[:] for row in self.target_state_default]
        self.active_fixed_config = {}

        if algo == "AND-OR Tree Search (8-Puzzle)":
            self.label_canvas1.config(text="Initial State (8-Puzzle)")
            self.label_canvas2.config(text="Target State (8-Puzzle)")
            self.draw_tkinter_grid(self.canvas1, initial_for_8puzzle_solver, {})
            self.draw_tkinter_grid(self.canvas2, target_for_8puzzle_solver, {}, is_target=True)
            solver = AndOrPuzzleSolver()
            results = solver.solve(initial_for_8puzzle_solver, target_for_8puzzle_solver)
            if results and results.get("final_state_list"):
                self.current_final_state_to_draw = [row[:] for row in results["final_state_list"]]
                self.label_canvas2.config(
                    text="Final State (Reached)" if results.get("success") else "Final State (Not Goal)")
                self.draw_tkinter_grid(self.canvas2, self.current_final_state_to_draw, {},
                                       is_target=results.get("success", False))

        elif algo == "Partially Observable Search (Belief Space Demo)":
            demo_runner = PartiallyObservableDemo()
            # Sử dụng hàm path_like_demo mới
            results = demo_runner.run_path_like_demo(num_initial_beliefs=2, path_len_to_generate=random.randint(2, 500))
            self.active_fixed_config = {}  # Demo này không dùng fixed config đặc biệt

            # Hiển thị trạng thái đầu tiên của niềm tin ban đầu
            if results.get("initial_state_sample"):
                self.current_initial_state_to_draw = [row[:] for row in results["initial_state_sample"]]
                self.label_canvas1.config(text="Initial Belief Sample")
            else:
                self.current_initial_state_to_draw = self.initial_state_default
                self.label_canvas1.config(text="Initial (Context)")
            self.draw_tkinter_grid(self.canvas1, self.current_initial_state_to_draw, self.active_fixed_config)

            # Hiển thị trạng thái đích của demo này trên canvas2
            target_for_demo = results.get("target_state", self.target_state_default)
            self.current_final_state_to_draw = [row[:] for row in target_for_demo]  # Đây là target của demo
            self.label_canvas2.config(text="Target State (Demo)")
            # Hoặc có thể hiển thị final_representative_state nếu muốn
            # final_rep_state = results.get("final_representative_state", self.target_state_default)
            # self.current_final_state_to_draw = [row[:] for row in final_rep_state]
            # self.label_canvas2.config(text="Final Rep. State")

            self.draw_tkinter_grid(self.canvas2, self.current_final_state_to_draw, self.active_fixed_config,
                                   is_target=True)  # is_target=True nếu canvas2 là target

            self.display_partially_observable_demo_results(results)

        elif algo == "Unknown/Dynamic Environment Search (Path Demo)":
            demo_runner = BeliefStatePuzzleDemo()
            num_moves_for_demo = random.randint(4, 500)
            self.add_log_message(f"  (Generating solvable puzzle with {num_moves_for_demo} scramble moves)", ('info',))
            results = demo_runner.run_guaranteed_success_demo(num_scramble_moves=num_moves_for_demo)
            current_run_fixed_config = results.get("fixed_config", {})
            self.active_fixed_config = current_run_fixed_config
            if results.get("initial_belief_states") and results["initial_belief_states"]:
                initial_to_draw = results["initial_belief_states"][0]
                self.current_initial_state_to_draw = [row[:] for row in initial_to_draw]
                self.label_canvas1.config(text="Initial State (Generated)")
                self.draw_tkinter_grid(self.canvas1, self.current_initial_state_to_draw, current_run_fixed_config)
            target_as_reference = results.get("target_state", self.target_state_default)
            final_state_from_sim = results.get("final_belief_states", [target_as_reference])[0]
            self.current_final_state_to_draw = [row[:] for row in final_state_from_sim]
            if results.get("success"):
                self.label_canvas2.config(text="Final State (Reached Target)")
            else:
                self.label_canvas2.config(text="Final State (Not Target)")
            self.draw_tkinter_grid(self.canvas2, self.current_final_state_to_draw, current_run_fixed_config,
                                   is_target=results.get("success", False))
            self.display_path_demo_summary_results(results)
        else:
            if algo:
                messagebox.showerror("Error", f"Algorithm '{algo}' not fully integrated.")
            else:
                messagebox.showwarning("Selection Missing", "Please select an algorithm first.")
            self.draw_tkinter_grid(self.canvas1, self.initial_state_default, {})
            self.draw_tkinter_grid(self.canvas2, self.target_state_default, {}, is_target=True)

        # Hiển thị kết quả chung cho AND-OR (các demo khác đã có hàm hiển thị riêng)
        if results and algo == "AND-OR Tree Search (8-Puzzle)":
            self.add_log_message(f"\n--- Results for {algo} ---", ('subheader',))
            self.add_log_message(f"Time Taken: {results.get('time_taken', 0.0):.4f}s", ('info',))
            self.add_log_message(f"Work Done (Nodes Evaluated): {results.get('nodes_expanded', 'N/A')}", ('info',))
            self.add_log_message(f"Reason: {results.get('reason', 'N/A')}", ('info',))
            if results.get("success", False):
                self.add_log_message(f"SUCCESS!", ('result_success',))
                cost_key = "cost" if "cost" in results else "g_cost"
                self.add_log_message(f"Path/Solution Cost (g): {results.get(cost_key, 'N/A')}", ('info',))
                self.add_log_message(f"Final Heuristic (h): {results.get('h_cost', 'N/A')}", ('info',))
                g = results.get(cost_key);
                h = results.get('h_cost')
                if isinstance(g, (int, float)) and isinstance(h, (int, float)):
                    self.add_log_message(f"Final f-cost (g+h): {g + h}", ('info',))
                self.add_log_message("\nPath Actions:", ('subheader',))
                if results.get("path_actions"):
                    for i, action in enumerate(results["path_actions"]):
                        if i < 15:
                            self.add_log_message(f"  Step {i + 1}: {action}")
                        elif i == 15:
                            self.add_log_message("  ..."); break
                else:
                    self.add_log_message("  (No actions, path not applicable/empty)")
            else:
                self.add_log_message(f"FAILED or ended without explicit success.", ('result_fail',))

        self.add_log_message(f"\n--- {algo} Execution Finished ---", ('footer',))

    def display_path_demo_summary_results(self, results):
        self.add_log_message("\n--- Unknown/Dynamic Env (Path Demo) Summary ---", ('subheader',))
        # ... (Nội dung giữ nguyên)
        fixed_conf_str = ", ".join([f"({r},{c}):{v}" for (r, c), v in results.get("fixed_config", {}).items()])
        self.add_log_message(f"Fixed Tiles: {fixed_conf_str if fixed_conf_str else 'None'}", ('info',))
        if results.get("initial_belief_states") and results["initial_belief_states"]:
            self.add_log_message("\nInitial State (derived from Target by scrambling):", ('label',))
            self.add_log_message(f"  State: {results['initial_belief_states'][0]}")
        self.add_log_message("\nTarget State (generated):", ('label',))
        self.add_log_message(f"  {results.get('target_state', 'N/A')}")
        self.add_log_message("\nSimulation Details:", ('label',))
        self.add_log_message(f"  Planned Actions to Solve: {results.get('simulated_actions', [])}")
        self.add_log_message(f"  Number of Actions Applied (Iterations): {results.get('nodes_expanded', 'N/A')}")
        self.add_log_message("\nOutcome:", ('label',))
        success_tag = ('result_success',) if results.get("success") else ('result_fail',)
        self.add_log_message(f"  Goal Reached: {results.get('success', False)}", success_tag)
        self.add_log_message(f"  Reason: {results.get('reason', 'N/A')}")
        if results.get("final_belief_states") and results["final_belief_states"]:
            self.add_log_message("\nFinal State Reached:", ('label',))
            self.add_log_message(f"  State: {results['final_belief_states'][0]}")
        self.add_log_message(f"  Final State Heuristic (h): {results.get('h_cost', 'N/A')}", ('info',))
        self.add_log_message(f"  Path Length (g): {results.get('steps', 'N/A')}", ('info',))

    def display_partially_observable_demo_results(self, results):
        self.add_log_message("\n--- Partially Observable (Belief Space) Demo Summary ---", ('subheader',))

        if results.get("initial_state_sample"):
            self.add_log_message("Initial State Sample (from Belief Set):", ('label',))
            self.add_log_message(f"  Sample 1: {results['initial_state_sample']}")

        if results.get("target_state"):
            self.add_log_message("\nTarget State for this Demo:", ('label',))
            self.add_log_message(f"  {results['target_state']}")

        if results.get("path_actions"):  # Đây là action_observation_log_for_ui
            self.add_log_message("\nSimulated Action-Observation Log:", ('label',))
            for entry_str in results["path_actions"]:
                self.add_log_message(f"  - {entry_str}", ('info',))

        self.add_log_message("\nFinal Outcome:", ('label',))
        success_tag = ('result_success',) if results.get("success") else ('result_fail',)
        self.add_log_message(f"  Reason: {results.get('reason', 'N/A')}")
        self.add_log_message(f"  Demo Success (a belief state matched target): {results.get('success', False)}",
                             success_tag)

        if results.get("final_representative_state"):
            self.add_log_message("Final Representative State (from Belief Set):", ('label',))
            self.add_log_message(f"  State: {results['final_representative_state']}")
        else:
            self.add_log_message("Final Belief Set: Empty or N/A", ('info',))

        self.add_log_message(f"Simulation Cycles Run: {results.get('nodes_expanded', 'N/A')}", ('info',))
        self.add_log_message(f"g_cost (Path Length/Cycles): {results.get('g_cost', 'N/A')}",
                             ('info',))  # g_cost là số bước theo kế hoạch
        self.add_log_message(f"h_cost (of Final Rep. to Target): {results.get('h_cost', 'N/A')}", ('info',))
        if "final_belief_set_size" in results:
            self.add_log_message(f"Final Belief Set Size: {results['final_belief_set_size']}", ('info',))

    def run_belief_space_search_conceptual_demo(self):
        # Chuyển logic gọi solver mới vào run_selected_algorithm
        # Hàm này giờ có thể được coi là không dùng nữa hoặc là một demo text thuần túy nếu muốn
        self.active_fixed_config = {}
        self.draw_tkinter_grid(self.canvas1, self.initial_state_default, self.active_fixed_config)
        self.draw_tkinter_grid(self.canvas2, self.target_state_default, self.active_fixed_config, is_target=True)
        self.label_canvas1.config(text="Initial State (Context)")
        self.label_canvas2.config(text="Target State (Context)")
        self.add_log_message("\n[Concept: Partially Observable Search (Belief Space) - General]", ('subheader',))
        self.add_log_message("  This option now runs a more specific demo with 8-puzzle states.")
        self.add_log_message(
            "  General idea: Agent has a belief (multiple possible states). Actions transform beliefs. Observations refine beliefs.")


def main():
    root = tk.Tk()
    app = Group4UI(root)
    root.mainloop()


if __name__ == '__main__':
    main()