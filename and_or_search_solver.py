# and_or_puzzle_solver.py
import time
import math


class AndOrPuzzleSolver:
    def __init__(self):
        self._goal_positions_cache = {}
        self.goal_state_list_ref = None
        self.nodes_evaluated = 0

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
        # Sử dụng self.goal_state_list_ref đã được thiết lập
        if not self.goal_state_list_ref: return float('inf')
        goal_positions = self._precompute_goal_positions(self.goal_state_list_ref)
        if not goal_positions: return float('inf')

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

    def _get_possible_moves(self, state_list):
        moves = []
        blank_r, blank_c = self._find_blank_position(state_list)
        possible_actions_order = [
            ("UP", -1, 0), ("DOWN", 1, 0),
            ("LEFT", 0, -1), ("RIGHT", 0, 1)
        ]
        for action_name, dr, dc in possible_actions_order:
            tile_to_swap_r, tile_to_swap_c = blank_r + dr, blank_c + dc
            if 0 <= tile_to_swap_r < 3 and 0 <= tile_to_swap_c < 3:
                new_state_list = [row[:] for row in state_list]
                new_state_list[blank_r][blank_c] = new_state_list[tile_to_swap_r][tile_to_swap_c]
                new_state_list[tile_to_swap_r][tile_to_swap_c] = 0
                moves.append((action_name, new_state_list, 1))
        return moves

    def _search_recursive_with_path(self, current_path_states_list_of_lists, g_cost_so_far, f_bound):
        self.nodes_evaluated += 1
        current_state_list = current_path_states_list_of_lists[-1]
        # current_state_tuple = tuple(map(tuple, current_state_list)) # Dùng cho visited_path_tuples nếu dùng set

        h = self._calculate_manhattan_distance(current_state_list)
        f = g_cost_so_far + h

        if f > f_bound:
            return f, None

        if current_state_list == self.goal_state_list_ref:
            return g_cost_so_far, []

        min_f_exceeded_for_next_iteration = float('inf')

        for action, next_state_l, action_cost in self._get_possible_moves(current_state_list):
            next_state_t_for_cycle_check = tuple(map(tuple, next_state_l))

            # Kiểm tra chu trình bằng cách xem next_state_l có trong current_path_states_list_of_lists không
            is_cycle = False
            for prev_state_in_path_list in current_path_states_list_of_lists:
                if prev_state_in_path_list == next_state_l:  # So sánh list of lists
                    is_cycle = True
                    break
            if is_cycle:
                continue

            current_path_states_list_of_lists.append(next_state_l)

            cost_from_next_or_new_bound, path_segment_from_next = self._search_recursive_with_path(
                current_path_states_list_of_lists,
                g_cost_so_far + action_cost,
                f_bound
            )

            current_path_states_list_of_lists.pop()

            if path_segment_from_next is not None:
                # cost_from_next_or_new_bound bây giờ là g_cost_to_goal từ gốc
                return cost_from_next_or_new_bound, [action] + path_segment_from_next

            if cost_from_next_or_new_bound < min_f_exceeded_for_next_iteration:
                min_f_exceeded_for_next_iteration = cost_from_next_or_new_bound

        return min_f_exceeded_for_next_iteration, None

    def solve(self, initial_state_list, goal_state_list, max_f_limit_iterations=40):
        start_time = time.time()
        self.nodes_evaluated = 0
        self.goal_state_list_ref = goal_state_list
        self._precompute_goal_positions(self.goal_state_list_ref)

        current_f_bound = self._calculate_manhattan_distance(initial_state_list)

        iteration_count = 0
        while iteration_count < max_f_limit_iterations:
            iteration_count += 1

            # Đường đi ban đầu chỉ chứa trạng thái ban đầu (dưới dạng list of lists)
            cost_or_new_bound, path_actions = self._search_recursive_with_path(
                [[row[:] for row in initial_state_list]],  # Path là list các state_list
                0,
                current_f_bound
            )

            if path_actions is not None:
                time_taken = time.time() - start_time
                return {
                    "success": True, "cost": cost_or_new_bound,
                    "g_cost": cost_or_new_bound, "h_cost": 0,
                    "path_actions": path_actions,
                    "final_state_list": goal_state_list,
                    "nodes_expanded": self.nodes_evaluated,
                    "time_taken": time_taken,
                    "reason": f"Goal reached with f-bound: {current_f_bound}"
                }

            if cost_or_new_bound == float('inf'): break
            current_f_bound = cost_or_new_bound

        time_taken = time.time() - start_time
        return {
            "success": False, "cost": float('inf'), "g_cost": float('inf'),
            "h_cost": self._calculate_manhattan_distance(initial_state_list),
            "path_actions": [], "final_state_list": initial_state_list,
            "nodes_expanded": self.nodes_evaluated, "time_taken": time_taken,
            "reason": f"Max iterations or no solution. Last f-bound: {current_f_bound}"
        }


if __name__ == '__main__':
    solver = AndOrPuzzleSolver()
    initial = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    result = solver.solve(initial, goal)