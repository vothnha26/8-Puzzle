# greedy_solver.py
import heapq
import time
import itertools  # Để tạo số thứ tự duy nhất cho các phần tử trong hàng đợi ưu tiên


class GreedySolver:
    def __init__(self):
        # Cache để lưu vị trí đích của các ô số, key là goal_state_tuple
        self._goal_positions_cache = {}

    def _precompute_goal_positions(self, goal_state_list):
        """
        Tính toán trước và lưu trữ vị trí đích của mỗi ô số (trừ ô trống).
        Sử dụng cache để tránh tính toán lại nếu goal_state không đổi.
        """
        goal_state_tuple_key = tuple(map(tuple, goal_state_list))
        if goal_state_tuple_key in self._goal_positions_cache:
            return self._goal_positions_cache[goal_state_tuple_key]

        positions = {}
        for r_idx, row in enumerate(goal_state_list):
            for c_idx, tile_value in enumerate(row):
                if tile_value != 0:  # Bỏ qua ô trống
                    positions[tile_value] = (r_idx, c_idx)

        self._goal_positions_cache[goal_state_tuple_key] = positions
        return positions

    def _calculate_manhattan_distance(self, current_state_list, goal_state_list):
        """Tính tổng khoảng cách Manhattan cho trạng thái hiện tại so với trạng thái đích."""
        goal_positions = self._precompute_goal_positions(goal_state_list)
        total_manhattan_distance = 0
        for r_idx, row in enumerate(current_state_list):
            for c_idx, tile_value in enumerate(row):
                if tile_value != 0:  # Bỏ qua ô trống
                    if tile_value in goal_positions:  # Đảm bảo ô số có trong đích
                        goal_r, goal_c = goal_positions[tile_value]
                        total_manhattan_distance += abs(r_idx - goal_r) + abs(c_idx - goal_c)
        return total_manhattan_distance

    def _find_blank_position(self, state_list):
        """Tìm vị trí của ô trống (số 0)."""
        for r_idx, row in enumerate(state_list):
            for c_idx, val in enumerate(row):
                if val == 0:
                    return r_idx, c_idx
        return None  # Nên luôn tìm thấy

    def _get_possible_moves(self, state_list):
        """Tạo các trạng thái kế tiếp hợp lệ."""
        moves = []
        blank_r, blank_c = self._find_blank_position(state_list)
        # (dr, dc, action_name)
        possible_actions = [
            (-1, 0, "UP"), (1, 0, "DOWN"),
            (0, -1, "LEFT"), (0, 1, "RIGHT")
        ]
        for dr, dc, action_name in possible_actions:
            new_r, new_c = blank_r + dr, blank_c + dc
            if 0 <= new_r < 3 and 0 <= new_c < 3:
                new_state_list = [row[:] for row in state_list]
                new_state_list[blank_r][blank_c] = new_state_list[new_r][new_c]
                new_state_list[new_r][new_c] = 0
                moves.append((action_name, new_state_list, 1))  # action, next_state, step_cost (luôn là 1)
        return moves

    def _reconstruct_path(self, came_from, current_state_tuple):
        """Truy vết đường đi và tính chi phí thực tế (g)."""
        path_actions = []
        actual_g_cost = 0
        temp_current_tuple = current_state_tuple
        while temp_current_tuple in came_from and came_from[temp_current_tuple] is not None:
            prev_state_tuple, action = came_from[temp_current_tuple]
            path_actions.append(action)
            actual_g_cost += 1  # Mỗi hành động tốn 1 chi phí
            temp_current_tuple = prev_state_tuple
        return path_actions[::-1], actual_g_cost

    def solve(self, initial_state_list, goal_state_list):
        """Thực hiện Greedy Best-First Search."""
        start_time = time.time()

        # Tính toán trước vị trí đích cho heuristic
        self._precompute_goal_positions(goal_state_list)

        initial_state_tuple = tuple(map(tuple, initial_state_list))

        # Hàng đợi ưu tiên: (h_cost, unique_id, state_list, current_g_cost_to_state)
        # current_g_cost_to_state là chi phí thực tế để đến state_list, dùng để báo cáo.
        pq = []
        unique_id_counter = itertools.count()

        initial_h_cost = self._calculate_manhattan_distance(initial_state_list, goal_state_list)
        heapq.heappush(pq, (initial_h_cost, next(unique_id_counter), initial_state_list, 0))

        # came_from: current_state_tuple -> (parent_state_tuple, action)
        came_from = {initial_state_tuple: None}

        # visited_expanded: set chứa các state_tuple đã được lấy ra khỏi pq và mở rộng.
        # Greedy không cần cost_so_far để quyết định, chỉ cần tránh mở rộng lại nút đã mở rộng.
        visited_expanded = set()

        nodes_expanded_count = 0

        while pq:
            h_cost, _, current_state_list, current_g_cost = heapq.heappop(pq)
            current_state_tuple = tuple(map(tuple, current_state_list))

            if current_state_tuple in visited_expanded:  # Nếu đã mở rộng trạng thái này rồi thì bỏ qua
                continue

            visited_expanded.add(current_state_tuple)  # Đánh dấu đã mở rộng
            nodes_expanded_count += 1

            if current_state_list == goal_state_list:  # Tìm thấy đích
                path_actions, actual_g_cost = self._reconstruct_path(came_from, current_state_tuple)
                final_h = self._calculate_manhattan_distance(current_state_list, goal_state_list)  # Phải bằng 0
                time_taken = time.time() - start_time
                return {
                    "path_actions": path_actions,
                    "steps": actual_g_cost,  # Số bước thực tế
                    "g_cost": actual_g_cost,  # Chi phí g thực tế
                    "h_cost": final_h,  # Heuristic của trạng thái đích (nên là 0)
                    "time_taken": time_taken,
                    "nodes_expanded": nodes_expanded_count,
                    "success": True
                }

            for action, next_state_list, step_cost in self._get_possible_moves(current_state_list):
                next_state_tuple = tuple(map(tuple, next_state_list))
                if next_state_tuple not in visited_expanded:  # Chỉ xem xét nếu chưa được mở rộng
                    new_g_cost_to_next = current_g_cost + step_cost
                    next_h_cost = self._calculate_manhattan_distance(next_state_list, goal_state_list)

                    # Với Greedy, ta thêm vào hàng đợi mà không cần kiểm tra chi phí cũ (nếu có).
                    # `visited_expanded` sẽ xử lý việc không mở rộng lại.
                    # Tuy nhiên, để `came_from` chính xác cho đường đi Greedy tìm được,
                    # ta nên cập nhật `came_from` khi thêm vào PQ.
                    came_from[next_state_tuple] = (current_state_tuple, action)
                    heapq.heappush(pq, (next_h_cost, next(unique_id_counter), next_state_list, new_g_cost_to_next))

        # Không tìm thấy giải pháp
        time_taken = time.time() - start_time
        return {
            "path_actions": [], "steps": 0, "g_cost": float('inf'), "h_cost": float('inf'),
            "time_taken": time_taken, "nodes_expanded": nodes_expanded_count, "success": False
        }


if __name__ == '__main__':
    solver = GreedySolver()
    initial = [
        [1, 8, 2],
        [0, 4, 3],  # H = 9 for this initial state with std goal
        [7, 6, 5]
    ]
    # initial = [[8,1,2],[0,4,3],[7,6,5]] # H = 8 + 0 + 0 + 1 + 1 + 0 + 0 = 10
    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    print(f"Solving puzzle with Greedy (Manhattan) from {initial} to {goal}")
    result = solver.solve(initial, goal)

    if result["success"]:
        print(f"Solution found. Steps (g): {result['g_cost']}, Final h: {result['h_cost']}")
        print(f"Path (actions): {result['path_actions']}")
    else:
        print("No solution found.")

    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Nodes expanded: {result['nodes_expanded']}")