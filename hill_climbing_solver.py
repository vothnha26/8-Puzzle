# hill_climbing_solver.py
import time
import random  # Cần thiết cho Stochastic Hill Climbing


class HillClimbingSolver:
    def __init__(self):
        self._goal_positions_cache = {}
        self.goal_state_list_ref = None
        self.nodes_evaluated = 0  # Đếm số trạng thái có heuristic được tính toán

    def _precompute_goal_positions(self, goal_state_list):
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

    def _calculate_manhattan_distance(self, current_state_list):
        goal_positions = self._precompute_goal_positions(self.goal_state_list_ref)
        total_manhattan_distance = 0
        for r_idx, row in enumerate(current_state_list):
            for c_idx, tile_value in enumerate(row):
                if tile_value != 0:
                    if tile_value in goal_positions:
                        goal_r, goal_c = goal_positions[tile_value]
                        total_manhattan_distance += abs(r_idx - goal_r) + abs(c_idx - goal_c)
        return total_manhattan_distance

    def _find_blank_position(self, state_list):
        for r_idx, row in enumerate(state_list):
            for c_idx, val in enumerate(row):
                if val == 0:
                    return r_idx, c_idx
        return None

    def _get_possible_moves(self, state_list):  # Trả về list (action_name, next_state_list)
        moves = []
        blank_r, blank_c = self._find_blank_position(state_list)
        possible_actions = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]
        for dr, dc, action_name in possible_actions:
            new_r, new_c = blank_r + dr, blank_c + dc
            if 0 <= new_r < 3 and 0 <= new_c < 3:
                new_state_list = [row[:] for row in state_list]
                new_state_list[blank_r][blank_c] = new_state_list[new_r][new_c]
                new_state_list[new_r][new_c] = 0
                moves.append((action_name, new_state_list))
        return moves

    def solve_simple_hc(self, initial_state_list, goal_state_list, max_iterations=1000):
        start_time = time.time()
        self.nodes_evaluated = 0
        self.goal_state_list_ref = goal_state_list
        self._precompute_goal_positions(self.goal_state_list_ref)
        current_state = [row[:] for row in initial_state_list]
        current_h = self._calculate_manhattan_distance(current_state)
        self.nodes_evaluated += 1
        path_actions_taken = []
        for iteration in range(max_iterations):
            if current_h == 0:
                time_taken = time.time() - start_time
                return {
                    "path_actions": path_actions_taken, "steps": len(path_actions_taken),
                    "g_cost": len(path_actions_taken), "h_cost": 0,
                    "final_state_list": current_state, "time_taken": time_taken,
                    "nodes_expanded": self.nodes_evaluated,
                    "success": True, "reason": "Goal reached."
                }
            neighbors = self._get_possible_moves(current_state)
            moved_to_better_neighbor = False
            for action, next_state in neighbors:  # Có thể random.shuffle(neighbors) ở đây
                self.nodes_evaluated += 1
                next_h = self._calculate_manhattan_distance(next_state)
                if next_h < current_h:
                    current_state = next_state
                    current_h = next_h
                    path_actions_taken.append(action)
                    moved_to_better_neighbor = True
                    break
            if not moved_to_better_neighbor:
                time_taken = time.time() - start_time
                reason = "Stuck at local optimum or plateau."
                success_status = (current_h == 0);
                if success_status: reason = "Goal reached (no better neighbor)."
                return {
                    "path_actions": path_actions_taken, "steps": len(path_actions_taken),
                    "g_cost": len(path_actions_taken), "h_cost": current_h,
                    "final_state_list": current_state, "time_taken": time_taken,
                    "nodes_expanded": self.nodes_evaluated,
                    "success": success_status, "reason": reason
                }
        time_taken = time.time() - start_time
        success_status = (current_h == 0)
        reason = "Max iterations reached."
        if success_status: reason = "Goal reached at max iterations."
        return {
            "path_actions": path_actions_taken, "steps": len(path_actions_taken),
            "g_cost": len(path_actions_taken), "h_cost": current_h,
            "final_state_list": current_state, "time_taken": time_taken,
            "nodes_expanded": self.nodes_evaluated,
            "success": success_status, "reason": reason
        }

    def solve_steepest_ascent_hc(self, initial_state_list, goal_state_list, max_iterations=1000):
        start_time = time.time()
        self.nodes_evaluated = 0
        self.goal_state_list_ref = goal_state_list
        self._precompute_goal_positions(self.goal_state_list_ref)
        current_state = [row[:] for row in initial_state_list]
        current_h = self._calculate_manhattan_distance(current_state)
        self.nodes_evaluated += 1
        path_actions_taken = []
        for iteration in range(max_iterations):
            if current_h == 0:
                time_taken = time.time() - start_time
                return {
                    "path_actions": path_actions_taken, "steps": len(path_actions_taken),
                    "g_cost": len(path_actions_taken), "h_cost": 0,
                    "final_state_list": current_state, "time_taken": time_taken,
                    "nodes_expanded": self.nodes_evaluated,
                    "success": True, "reason": "Goal reached."
                }
            neighbors_generated = self._get_possible_moves(current_state)
            best_neighbor_state = None
            best_neighbor_h = current_h
            action_to_best_neighbor = None
            if not neighbors_generated: break
            for action, next_state in neighbors_generated:
                self.nodes_evaluated += 1
                h_next = self._calculate_manhattan_distance(next_state)
                if h_next < best_neighbor_h:
                    best_neighbor_h = h_next
                    best_neighbor_state = next_state
                    action_to_best_neighbor = action
            if best_neighbor_state is not None and best_neighbor_h < current_h:
                current_state = best_neighbor_state
                current_h = best_neighbor_h
                path_actions_taken.append(action_to_best_neighbor)
            else:
                time_taken = time.time() - start_time
                reason = "Stuck at local optimum or plateau."
                success_status = (current_h == 0)
                if success_status: reason = "Goal reached (no better neighbor)."
                return {
                    "path_actions": path_actions_taken, "steps": len(path_actions_taken),
                    "g_cost": len(path_actions_taken), "h_cost": current_h,
                    "final_state_list": current_state, "time_taken": time_taken,
                    "nodes_expanded": self.nodes_evaluated,
                    "success": success_status, "reason": reason
                }
        time_taken = time.time() - start_time
        success_status = (current_h == 0)
        reason = "Max iterations reached."
        if success_status: reason = "Goal reached at max iterations."
        return {
            "path_actions": path_actions_taken, "steps": len(path_actions_taken),
            "g_cost": len(path_actions_taken), "h_cost": current_h,
            "final_state_list": current_state, "time_taken": time_taken,
            "nodes_expanded": self.nodes_evaluated,
            "success": success_status, "reason": reason
        }

    def solve_stochastic_hc(self, initial_state_list, goal_state_list, max_iterations=1000):
        """
        Thực hiện Stochastic Hill Climbing.
        Chọn ngẫu nhiên một trong số các láng giềng tốt hơn (uphill).
        """
        start_time = time.time()
        self.nodes_evaluated = 0
        self.goal_state_list_ref = goal_state_list
        self._precompute_goal_positions(self.goal_state_list_ref)

        current_state = [row[:] for row in initial_state_list]
        current_h = self._calculate_manhattan_distance(current_state)
        self.nodes_evaluated += 1

        path_actions_taken = []

        for iteration in range(max_iterations):
            if current_h == 0:  # Đã đạt đến trạng thái đích
                time_taken = time.time() - start_time
                return {
                    "path_actions": path_actions_taken, "steps": len(path_actions_taken),
                    "g_cost": len(path_actions_taken), "h_cost": 0,
                    "final_state_list": current_state, "time_taken": time_taken,
                    "nodes_expanded": self.nodes_evaluated,
                    "success": True, "reason": "Goal reached."
                }

            neighbors_generated = self._get_possible_moves(current_state)
            uphill_neighbors = []  # Lưu trữ các láng giềng tốt hơn: (action, next_state, h_cost_của_next_state)

            for action, next_state in neighbors_generated:
                self.nodes_evaluated += 1
                h_next = self._calculate_manhattan_distance(next_state)
                if h_next < current_h:  # Chỉ xem xét các láng giềng tốt hơn (uphill move)
                    uphill_neighbors.append((action, next_state, h_next))

            if uphill_neighbors:  # Nếu có láng giềng tốt hơn
                # Chọn ngẫu nhiên một trong số các láng giềng tốt hơn đó
                selected_action, selected_next_state, selected_h_next = random.choice(uphill_neighbors)

                current_state = selected_next_state
                current_h = selected_h_next  # Sử dụng h_cost đã tính của láng giềng được chọn
                path_actions_taken.append(selected_action)
            else:
                # Không có láng giềng nào tốt hơn => bị kẹt
                time_taken = time.time() - start_time
                reason = "Stuck (no uphill move)."  # Ngắn gọn hơn cho Stochastic
                success_status = (current_h == 0)
                if success_status: reason = "Goal reached (no uphill move)."
                return {
                    "path_actions": path_actions_taken, "steps": len(path_actions_taken),
                    "g_cost": len(path_actions_taken), "h_cost": current_h,
                    "final_state_list": current_state, "time_taken": time_taken,
                    "nodes_expanded": self.nodes_evaluated,
                    "success": success_status, "reason": reason
                }

        # Đạt đến số vòng lặp tối đa
        time_taken = time.time() - start_time
        success_status = (current_h == 0)
        reason = "Max iterations reached."
        if success_status: reason = "Goal reached at max iterations."
        return {
            "path_actions": path_actions_taken, "steps": len(path_actions_taken),
            "g_cost": len(path_actions_taken), "h_cost": current_h,
            "final_state_list": current_state, "time_taken": time_taken,
            "nodes_expanded": self.nodes_evaluated,
            "success": success_status, "reason": reason
        }


if __name__ == '__main__':
    solver = HillClimbingSolver()
    initial = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    # ... (Test code cho Simple và Steepest-Ascent HC giữ nguyên) ...

    print("\n--- Stochastic Hill Climbing ---")
    # Chạy nhiều lần để thấy tính ngẫu nhiên
    successful_runs = 0
    for i in range(5):  # Ví dụ chạy 5 lần
        print(f"Run {i + 1}:")
        # Tạo một solver mới mỗi lần để reset self.nodes_evaluated nếu nó không được reset trong solve
        # Hoặc đảm bảo solver.nodes_evaluated được reset đúng cách.
        # Trong class hiện tại, nó được reset trong mỗi hàm solve_..._hc
        run_solver = HillClimbingSolver()
        result_stochastic = run_solver.solve_stochastic_hc(initial, goal, max_iterations=200)
        if result_stochastic['success']: successful_runs += 1
        print(f"  Success: {result_stochastic['success']}, Reason: {result_stochastic['reason']}")
        print(f"  Steps: {result_stochastic['steps']}, Final h: {result_stochastic['h_cost']}")
        # print(f"  Path: {result_stochastic['path_actions']}") # Có thể khá dài
        print(f"  Time: {result_stochastic['time_taken']:.4f}s, Nodes Eval: {result_stochastic['nodes_expanded']}")
    print(f"Stochastic HC successful runs: {successful_runs}/5")