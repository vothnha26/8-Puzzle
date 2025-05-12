# a_star_solver.py
import heapq
import time
import itertools  # Để tạo số thứ tự duy nhất cho các phần tử trong hàng đợi ưu tiên


class AStarSolver:
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
        # Đảm bảo goal_positions được tính cho goal_state_list cụ thể này
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
        return None

    def _get_possible_moves(self, state_list):
        """Tạo các trạng thái kế tiếp hợp lệ."""
        moves = []
        blank_r, blank_c = self._find_blank_position(state_list)
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
        # Chi phí g thực tế sẽ được lấy từ current_g_cost của nút đích khi nó được pop khỏi PQ.
        # Hàm này chỉ cần trả về các hành động.
        temp_current_tuple = current_state_tuple
        while temp_current_tuple in came_from and came_from[temp_current_tuple] is not None:
            prev_state_tuple, action = came_from[temp_current_tuple]
            path_actions.append(action)
            temp_current_tuple = prev_state_tuple
        return path_actions[::-1]

    def solve(self, initial_state_list, goal_state_list):
        """Thực hiện thuật toán A*."""
        start_time = time.time()

        self._precompute_goal_positions(goal_state_list)  # Đảm bảo cache vị trí đích được khởi tạo

        initial_state_tuple = tuple(map(tuple, initial_state_list))

        # Hàng đợi ưu tiên: (f_cost, unique_id, state_list, current_g_cost_to_state)
        pq = []
        unique_id_counter = itertools.count()

        initial_g_cost = 0
        initial_h_cost = self._calculate_manhattan_distance(initial_state_list, goal_state_list)
        initial_f_cost = initial_g_cost + initial_h_cost

        heapq.heappush(pq, (initial_f_cost, next(unique_id_counter), initial_state_list, initial_g_cost))

        # came_from: current_state_tuple -> (parent_state_tuple, action)
        came_from = {initial_state_tuple: None}
        # cost_so_far_g: lưu trữ chi phí g_cost thấp nhất tìm được để đến một state_tuple
        cost_so_far_g = {initial_state_tuple: initial_g_cost}

        nodes_expanded_count = 0

        while pq:
            f_cost, _, current_state_list, current_g_cost = heapq.heappop(pq)
            current_state_tuple = tuple(map(tuple, current_state_list))

            # Nếu chi phí g hiện tại để đến nút này lớn hơn chi phí g đã lưu trữ
            # (tức là đã tìm thấy đường đi ngắn hơn đến nút này trước đó), thì bỏ qua.
            if current_g_cost > cost_so_far_g.get(current_state_tuple, float('inf')):
                continue

            nodes_expanded_count += 1

            if current_state_list == goal_state_list:  # Tìm thấy đích
                path_actions = self._reconstruct_path(came_from, current_state_tuple)
                # current_g_cost là chi phí tối ưu đến đích
                final_h = self._calculate_manhattan_distance(current_state_list, goal_state_list)  # Phải bằng 0
                time_taken = time.time() - start_time
                return {
                    "path_actions": path_actions,
                    "steps": current_g_cost,  # Số bước tối ưu
                    "g_cost": current_g_cost,  # Chi phí g tối ưu
                    "h_cost": final_h,  # Heuristic của trạng thái đích
                    "time_taken": time_taken,
                    "nodes_expanded": nodes_expanded_count,
                    "success": True
                }

            for action, next_state_list, step_cost in self._get_possible_moves(current_state_list):
                new_g_cost = current_g_cost + step_cost  # Chi phí g mới để đến trạng thái con
                next_state_tuple = tuple(map(tuple, next_state_list))

                # Nếu tìm thấy đường đi mới tốt hơn (chi phí g thấp hơn) đến trạng thái con
                if new_g_cost < cost_so_far_g.get(next_state_tuple, float('inf')):
                    cost_so_far_g[next_state_tuple] = new_g_cost  # Cập nhật chi phí g thấp nhất
                    came_from[next_state_tuple] = (current_state_tuple, action)  # Lưu vết đường đi

                    h_next = self._calculate_manhattan_distance(next_state_list, goal_state_list)
                    f_next = new_g_cost + h_next  # Tính f_cost cho trạng thái con
                    heapq.heappush(pq, (f_next, next(unique_id_counter), next_state_list, new_g_cost))

        # Không tìm thấy giải pháp
        time_taken = time.time() - start_time
        return {
            "path_actions": [], "steps": 0, "g_cost": float('inf'), "h_cost": float('inf'),
            "time_taken": time_taken, "nodes_expanded": nodes_expanded_count, "success": False
        }


if __name__ == '__main__':
    solver = AStarSolver()
    initial = [
        [1, 8, 2],
        [0, 4, 3],
        [7, 6, 5]
    ]
    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    print(f"Solving puzzle with A* (Manhattan) from {initial} to {goal}")
    result = solver.solve(initial, goal)

    if result["success"]:
        print(
            f"Solution found. Steps (g): {result['g_cost']}, Final h: {result['h_cost']}, Final f: {result['g_cost'] + result['h_cost']}")
        print(f"Path (actions): {result['path_actions']}")
    else:
        print("No solution found.")

    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Nodes expanded: {result['nodes_expanded']}")