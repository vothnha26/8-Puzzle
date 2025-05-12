# ida_star_solver.py
import time


class IDAStarSolver:
    def __init__(self):
        self._goal_positions_cache = {}
        self.nodes_expanded_total = 0
        # path_came_from sẽ được xây dựng trong lần lặp thành công của Cost-Limited Search
        self.path_came_from_solution = {}
        self.goal_state_list_ref = None  # Tham chiếu đến trạng thái đích

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
        # Sử dụng self.goal_state_list_ref đã được gán
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

    def _get_possible_moves(self, state_list):
        moves = []
        blank_r, blank_c = self._find_blank_position(state_list)
        possible_actions = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]
        for dr, dc, action_name in possible_actions:
            new_r, new_c = blank_r + dr, blank_c + dc
            if 0 <= new_r < 3 and 0 <= new_c < 3:
                new_state_list = [row[:] for row in state_list]
                new_state_list[blank_r][blank_c] = new_state_list[new_r][new_c]
                new_state_list[new_r][new_c] = 0
                moves.append((action_name, new_state_list, 1))
        return moves

    def _reconstruct_path(self, goal_state_tuple):
        path_actions = []
        temp_current_tuple = goal_state_tuple
        # Sử dụng self.path_came_from_solution đã được xây dựng khi tìm thấy đích
        while temp_current_tuple in self.path_came_from_solution and \
                self.path_came_from_solution[temp_current_tuple] is not None:
            prev_state_tuple, action = self.path_came_from_solution[temp_current_tuple]
            path_actions.append(action)
            temp_current_tuple = prev_state_tuple
        return path_actions[::-1]

    def _cost_limited_search_recursive(self, current_path_tuples, g_cost, f_limit_current_iteration):
        """
        Tìm kiếm giới hạn chi phí (Cost-Limited Search - CLS) đệ quy.
        current_path_tuples: list các state_tuple của đường đi hiện tại để tránh chu trình.
        Trả về:
            - "FOUND" nếu tìm thấy đích.
            - Giá trị f nhỏ nhất vượt quá f_limit_current_iteration.
            - float('inf') nếu tất cả các nhánh đều được khám phá và không vượt f_limit (không tìm thấy đích).
        """
        self.nodes_expanded_total += 1

        current_state_tuple = current_path_tuples[-1]
        # Chuyển đổi tuple về list để tính toán và so sánh (nếu cần)
        current_state_list = [list(row) for row in current_state_tuple]

        h_cost = self._calculate_manhattan_distance(current_state_list)
        f_cost = g_cost + h_cost

        if f_cost > f_limit_current_iteration:
            return f_cost  # Trả về f_cost này để có thể làm f_limit cho vòng lặp sau

        if current_state_list == self.goal_state_list_ref:  # So sánh với trạng thái đích toàn cục
            return "FOUND"

        min_next_f_threshold = float('inf')

        for action, next_state_list, step_cost in self._get_possible_moves(current_state_list):
            next_state_tuple = tuple(map(tuple, next_state_list))

            if next_state_tuple not in current_path_tuples:  # Tránh chu trình trong đường đi hiện tại
                current_path_tuples.append(next_state_tuple)
                # Ghi nhận đường đi tiềm năng, sẽ được dùng nếu nhánh này dẫn đến đích
                self.path_came_from_solution[next_state_tuple] = (current_state_tuple, action)

                result = self._cost_limited_search_recursive(
                    current_path_tuples,
                    g_cost + step_cost,
                    f_limit_current_iteration
                )

                if result == "FOUND":
                    return "FOUND"  # Lan truyền tín hiệu tìm thấy

                if result < min_next_f_threshold:
                    min_next_f_threshold = result  # Cập nhật ngưỡng f nhỏ nhất bị vượt qua

                current_path_tuples.pop()  # Backtrack
                # Không cần xóa self.path_came_from_solution[next_state_tuple] ở đây,
                # vì nếu một nhánh khác tốt hơn tìm thấy đích, nó sẽ ghi đè.
                # Hoặc, chỉ xây dựng came_from khi "FOUND". Cách hiện tại đơn giản hơn.

        return min_next_f_threshold

    def solve(self, initial_state_list, goal_state_list, max_iterations_ida=100):  # Giới hạn số lần tăng f_limit
        start_time = time.time()
        self.nodes_expanded_total = 0  # Reset cho mỗi lần giải mới
        self.goal_state_list_ref = goal_state_list  # Lưu trữ tham chiếu trạng thái đích
        self._precompute_goal_positions(self.goal_state_list_ref)  # Tính toán trước vị trí đích

        initial_state_tuple = tuple(map(tuple, initial_state_list))

        current_f_limit = self._calculate_manhattan_distance(initial_state_list)

        iteration_count = 0
        while iteration_count < max_iterations_ida:
            iteration_count += 1
            # print(f"IDA* Iteration: {iteration_count}, f_limit = {current_f_limit}") # Để debug

            # path_came_from_solution cần được quản lý cẩn thận.
            # Nó được xây dựng bởi CLS, nhưng chỉ có ý nghĩa khi CLS trả về "FOUND".
            # Ta có thể reset nó ở đây hoặc để CLS ghi đè. Reset ở đây an toàn hơn.
            self.path_came_from_solution = {initial_state_tuple: None}

            result_cls = self._cost_limited_search_recursive(
                [initial_state_tuple],  # Đường đi hiện tại bắt đầu với trạng thái ban đầu (dạng tuple)
                0,  # g_cost ban đầu là 0
                current_f_limit
            )

            if result_cls == "FOUND":
                # Đích được tìm thấy, self.path_came_from_solution chứa đường đi đúng
                final_goal_state_tuple = tuple(map(tuple, self.goal_state_list_ref))
                path_actions = self._reconstruct_path(final_goal_state_tuple)
                actual_g_cost = len(path_actions)  # Với chi phí mỗi bước là 1
                final_h = 0  # h(đích) = 0

                time_taken = time.time() - start_time
                return {
                    "path_actions": path_actions,
                    "steps": actual_g_cost,
                    "g_cost": actual_g_cost,
                    "h_cost": final_h,
                    "time_taken": time_taken,
                    "nodes_expanded": self.nodes_expanded_total,
                    "success": True,
                    "f_limit_final": current_f_limit  # f_limit mà tại đó tìm thấy giải pháp
                }

            if result_cls == float('inf'):  # Tất cả các nhánh đã được khám phá và không tìm thấy đích
                break  # Không có giải pháp

            current_f_limit = result_cls  # Cập nhật f_limit cho vòng lặp tiếp theo

        # Nếu vòng lặp kết thúc (do vượt quá max_iterations_ida hoặc result_cls == float('inf'))
        time_taken = time.time() - start_time
        return {
            "path_actions": [], "steps": 0, "g_cost": float('inf'), "h_cost": float('inf'),
            "time_taken": time_taken, "nodes_expanded": self.nodes_expanded_total,
            "success": False, "f_limit_final": current_f_limit
        }


if __name__ == '__main__':
    solver = IDAStarSolver()
    initial = [
        [1, 8, 2],
        [0, 4, 3],
        [7, 6, 5]
    ]
    # initial = [[2,8,3],[1,6,4],[7,0,5]] # Optimal: 5 steps
    # initial = [[8,6,7],[2,5,4],[3,0,1]] # Hard: 31 steps. Cần max_iterations_ida cao.

    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    print(f"Solving puzzle with IDA* (Manhattan) from {initial} to {goal}")
    # Đặt max_iterations_ida hợp lý, ví dụ 40 (vì f_limit có thể tăng khá nhanh với heuristic tốt)
    result = solver.solve(initial, goal, max_iterations_ida=40)

    if result["success"]:
        print(
            f"Solution found with final f_limit {result['f_limit_final']}. Steps (g): {result['g_cost']}, Final h: {result['h_cost']}")
        print(f"Path (actions): {result['path_actions']}")
    else:
        print(f"No solution found. Last f_limit tried: {result['f_limit_final']}.")

    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Nodes expanded: {result['nodes_expanded']}")