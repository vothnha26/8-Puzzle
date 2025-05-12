# belief_search_demo_solver.py
import random
import time


# import math # Bỏ import math nếu không dùng trực tiếp trong file này

class BeliefStatePuzzleDemo:
    def __init__(self):
        self.log = []
        self.fixed_cells_config = {}
        self.all_tiles = list(range(9))
        self.num_fixed_tiles = 3
        self.grid_size = 3
        self.iterations_count = 0
        self._goal_positions_cache = {}  # Đã có cache cho goal positions
        self.goal_state_list_ref = None  # Sẽ được gán khi solve

    def _is_valid_pos(self, r, c):
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _find_blank_in_state(self, state_list_2d):
        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                if state_list_2d[r_idx][c_idx] == 0:
                    return r_idx, c_idx
        return None

    def _generate_random_fixed_config(self):
        self.fixed_cells_config = {}
        available_cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                available_cells.append((r, c))
        random.shuffle(available_cells)

        fixed_tile_values = random.sample([t for t in self.all_tiles if t != 0], self.num_fixed_tiles)

        for i in range(self.num_fixed_tiles):
            cell_to_fix = available_cells.pop(0)
            tile_value_to_fix = fixed_tile_values[i]
            self.fixed_cells_config[cell_to_fix] = tile_value_to_fix

        self.log.append(f"Fixed Configuration (Cell:Tile): { {str(k): v for k, v in self.fixed_cells_config.items()} }")

    def _precompute_goal_positions(self, goal_state_list):  # Đã có hàm này
        goal_state_tuple_key = tuple(map(tuple, goal_state_list))
        if goal_state_tuple_key in self._goal_positions_cache:
            return self._goal_positions_cache[goal_state_tuple_key]
        positions = {}
        for r_idx, row in enumerate(goal_state_list):
            for c_idx, tile_value in enumerate(row):
                if tile_value != 0:
                    positions[tile_value] = (r_idx, c_idx)
        self._goal_positions_cache[goal_state_tuple_key] = positions
        return positions

    # <<< THÊM PHƯƠNG THỨC BỊ THIẾU VÀO ĐÂY >>>
    def _calculate_manhattan_distance(self, current_state_list):
        # Hàm này cần self.goal_state_list_ref đã được thiết lập
        if not self.goal_state_list_ref:
            # self.log.append("Error: Target state reference not set for Manhattan calculation.") # Ghi log lỗi
            return float('inf')  # Hoặc một giá trị lỗi khác

        goal_positions = self._precompute_goal_positions(self.goal_state_list_ref)
        total_manhattan_distance = 0
        for r_idx, row in enumerate(current_state_list):
            for c_idx, tile_value in enumerate(row):
                if tile_value != 0:  # Bỏ qua ô trống
                    if tile_value in goal_positions:  # Đảm bảo ô số có trong đích
                        goal_r, goal_c = goal_positions[tile_value]
                        total_manhattan_distance += abs(r_idx - goal_r) + abs(c_idx - goal_c)
                    # else:
                    # Ô số này không có trong cấu hình đích (không nên xảy ra với 8-puzzle chuẩn)
                    # Có thể thêm một hình phạt lớn ở đây nếu muốn
                    # total_manhattan_distance += self.grid_size * self.grid_size # Hình phạt lớn
        return total_manhattan_distance

    # <<< KẾT THÚC PHẦN THÊM >>>

    def _generate_state_respecting_fixed(self):
        # ... (Giữ nguyên như trước)
        state = [[-1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        placed_tiles_values = set()
        for (r, c), tile_val in self.fixed_cells_config.items():
            state[r][c] = tile_val
            placed_tiles_values.add(tile_val)
        remaining_tiles_values = [t for t in self.all_tiles if t not in placed_tiles_values]
        random.shuffle(remaining_tiles_values)
        empty_grid_cells = []
        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                if state[r_idx][c_idx] == -1:
                    empty_grid_cells.append((r_idx, c_idx))
        for i, (r_idx, c_idx) in enumerate(empty_grid_cells):
            state[r_idx][c_idx] = remaining_tiles_values[i]
        return state

    def _apply_action_to_state(self, state_list_2d, action_str, log_this_action=True, log_prefix="  "):
        # ... (Giữ nguyên như trước)
        br, bc = self._find_blank_in_state(state_list_2d)
        if br is None:
            if log_this_action: self.log.append(f"{log_prefix}Action {action_str}: No blank tile. State unchanged.")
            return [row[:] for row in state_list_2d], False
        dr, dc = 0, 0
        if action_str == "UP":
            dr = -1
        elif action_str == "DOWN":
            dr = 1
        elif action_str == "LEFT":
            dc = -1
        elif action_str == "RIGHT":
            dc = 1
        target_r, target_c = br + dr, bc + dc
        new_state = [row[:] for row in state_list_2d]
        if not self._is_valid_pos(target_r, target_c):
            if log_this_action: self.log.append(
                f"{log_prefix}Action {action_str}: Blank at ({br},{bc}) hits wall. State unchanged.")
            return new_state, False
        tile_in_target_cell = new_state[target_r][target_c]
        if (target_r, target_c) in self.fixed_cells_config and \
                self.fixed_cells_config[(target_r, target_c)] == tile_in_target_cell:
            if log_this_action: self.log.append(
                f"{log_prefix}Action {action_str}: Blank at ({br},{bc}) attempts to swap with fixed tile {tile_in_target_cell} at ({target_r},{target_c}). Move invalid.")
            return new_state, False
        new_state[br][bc] = tile_in_target_cell
        new_state[target_r][target_c] = 0
        if log_this_action: self.log.append(
            f"{log_prefix}Action {action_str}: Blank from ({br},{bc}) to ({target_r},{target_c}). Swapped with {tile_in_target_cell}.")
        return new_state, True

    def get_inverse_action(self, action_str):
        # ... (Giữ nguyên như trước)
        if action_str == "UP": return "DOWN"
        if action_str == "DOWN": return "UP"
        if action_str == "LEFT": return "RIGHT"
        if action_str == "RIGHT": return "LEFT"
        return None

    def run_guaranteed_success_demo(self, num_scramble_moves=3):
        self.log = []
        start_time = time.time()
        self.iterations_count = 0

        self._generate_random_fixed_config()

        # Gán self.goal_state_list_ref TRƯỚC KHI _calculate_manhattan_distance có thể được gọi (gián tiếp hoặc trực tiếp)
        self.goal_state_list_ref = self._generate_state_respecting_fixed()  # Đây sẽ là target
        target_state = self.goal_state_list_ref  # Sử dụng biến cục bộ cho rõ ràng hơn
        self.log.append(f"\nTarget State (generated randomly, respecting fixed tiles): {target_state}")

        current_scrambled_state = [row[:] for row in target_state]
        actions_to_scramble = []
        possible_agent_actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        for _ in range(num_scramble_moves):
            valid_scramble_action = None
            shuffled_actions = random.sample(possible_agent_actions, len(possible_agent_actions))
            for candidate_action in shuffled_actions:
                if actions_to_scramble and candidate_action == self.get_inverse_action(actions_to_scramble[-1]):
                    continue
                temp_next_state, moved = self._apply_action_to_state(current_scrambled_state, candidate_action,
                                                                     log_this_action=False)
                if moved and temp_next_state != current_scrambled_state:
                    valid_scramble_action = candidate_action
                    current_scrambled_state = temp_next_state
                    break
            if valid_scramble_action:
                actions_to_scramble.append(valid_scramble_action)
            else:
                break

        initial_state = current_scrambled_state
        simulated_actions_to_solve = [self.get_inverse_action(act) for act in reversed(actions_to_scramble)]

        self.log.append(
            f"\nInitial State (generated by {len(actions_to_scramble)} scramble moves from target): {initial_state}")
        if not simulated_actions_to_solve:
            self.log.append("  (Initial state is the same as target or scrambling failed to make distinct state)")

        current_simulation_state = [row[:] for row in initial_state]
        actual_path_taken_in_sim = []
        self.log.append(f"\n--- Simulating Solution Path ({len(simulated_actions_to_solve)} steps) ---")

        for i, action_to_take in enumerate(simulated_actions_to_solve):
            self.iterations_count += 1
            self.log.append(f"Step {i + 1}: Agent applies Action '{action_to_take}'")
            next_sim_state, moved = self._apply_action_to_state(current_simulation_state, action_to_take,
                                                                log_prefix="    ")
            if moved:
                current_simulation_state = next_sim_state
                actual_path_taken_in_sim.append(action_to_take)
            else:
                self.log.append(f"    Action '{action_to_take}' was unexpectedly invalid or ineffective during solve.")
                break

        final_achieved_state = current_simulation_state
        time_taken = time.time() - start_time
        success = (final_achieved_state == target_state)

        reason = "Goal reached successfully via planned path." if success else "Failed to reach goal with planned path."
        if not success and self.iterations_count < len(simulated_actions_to_solve):
            reason = "Simulation stopped prematurely."

        # Tính h_cost của trạng thái cuối cùng đạt được
        # Đảm bảo self.goal_state_list_ref đã được set (đã làm ở trên)
        final_h_cost = self._calculate_manhattan_distance(final_achieved_state)

        return {
            "success": success, "log": self.log,
            "initial_belief_states": [initial_state],
            "target_state": target_state,
            "final_belief_states": [final_achieved_state],
            "fixed_config": self.fixed_cells_config,
            "simulated_actions": actual_path_taken_in_sim,
            "time_taken": time_taken, "nodes_expanded": self.iterations_count,
            "reason": reason, "steps": len(actual_path_taken_in_sim),
            "g_cost": len(actual_path_taken_in_sim), "h_cost": final_h_cost
        }


# ... (phần if __name__ == '__main__': giữ nguyên) ...
if __name__ == '__main__':
    demo = BeliefStatePuzzleDemo()
    results = demo.run_guaranteed_success_demo(num_scramble_moves=3)
    for entry in results["log"]: print(entry)
    print("-" * 20)
    print(f"Fixed tiles config: {results['fixed_config']}")
    print(f"Target State: {results['target_state']}")
    print(f"Initial State (derived): {results['initial_belief_states'][0]}")
    print(f"Actions to Solve: {results['simulated_actions']}")
    print(f"Final State Reached: {results['final_belief_states'][0]}")
    print(f"Success: {results['success']}")
    print(f"Reason: {results['reason']}")
    print(f"Time: {results['time_taken']:.4f}s, Iterations: {results['nodes_expanded']}")
    print(f"Final h_cost: {results['h_cost']}")