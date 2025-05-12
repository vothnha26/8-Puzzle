# ucs_solver.py
import heapq  # Thư viện cho hàng đợi ưu tiên (min-heap)
import time
import itertools  # Để tạo số thứ tự duy nhất cho các phần tử trong hàng đợi ưu tiên


class UCSSolver:
    def __init__(self):
        pass

    def _find_blank_position(self, state_list):
        """Tìm vị trí của ô trống (số 0) trong một state_list."""
        for r_idx, row in enumerate(state_list):
            for c_idx, val in enumerate(row):
                if val == 0:
                    return r_idx, c_idx
        return None

    def _get_possible_moves(self, state_list):
        """
        Tạo ra các trạng thái kế tiếp hợp lệ từ trạng thái hiện tại.
        Trả về một danh sách các tuple (action, next_state_list, step_cost).
        Đối với 8-puzzle, step_cost luôn là 1.
        """
        moves = []
        blank_r, blank_c = self._find_blank_position(state_list)

        # Các hướng di chuyển có thể của ô trống: (dr, dc, action_name)
        possible_actions = [
            (-1, 0, "UP"),  # Ô trống đi lên
            (1, 0, "DOWN"),  # Ô trống đi xuống
            (0, -1, "LEFT"),  # Ô trống đi qua trái
            (0, 1, "RIGHT")  # Ô trống đi qua phải
        ]

        for dr, dc, action_name in possible_actions:
            new_r, new_c = blank_r + dr, blank_c + dc

            if 0 <= new_r < 3 and 0 <= new_c < 3:  # Kiểm tra vị trí mới hợp lệ
                new_state_list = [row[:] for row in state_list]  # Sao chép trạng thái hiện tại
                # Thực hiện di chuyển
                new_state_list[blank_r][blank_c] = new_state_list[new_r][new_c]
                new_state_list[new_r][new_c] = 0
                moves.append((action_name, new_state_list, 1))  # step_cost là 1
        return moves

    def _reconstruct_path(self, came_from, current_state_tuple):
        """
        Truy vết đường đi từ trạng thái đích về trạng thái ban đầu.
        Trả về danh sách các hành động.
        """
        path_actions = []
        # current_state_tuple là trạng thái đã được chuyển thành tuple để dùng làm key
        while current_state_tuple in came_from and came_from[current_state_tuple] is not None:
            prev_state_tuple, action = came_from[current_state_tuple]
            path_actions.append(action)
            current_state_tuple = prev_state_tuple
        return path_actions[::-1]  # Đảo ngược để có thứ tự từ đầu đến cuối

    def solve(self, initial_state_list, goal_state_list):
        """
        Thực hiện thuật toán Uniform Cost Search (UCS).
        Trả về một dictionary chứa: path_actions, steps, g_cost, time_taken, nodes_expanded, success.
        """
        start_time = time.time()

        initial_state_tuple = tuple(map(tuple, initial_state_list))
        # goal_state_tuple = tuple(map(tuple, goal_state_list)) # Không cần thiết nếu so sánh list

        # Hàng đợi ưu tiên (min-heap): lưu (g_cost, unique_id, state_list)
        # unique_id dùng để tránh lỗi so sánh khi g_cost bằng nhau và state_list không so sánh được.
        pq = []
        unique_id_counter = itertools.count()  # Bộ đếm cho unique_id

        # Thêm trạng thái ban đầu vào hàng đợi
        heapq.heappush(pq, (0, next(unique_id_counter), initial_state_list))

        # came_from: dict lưu state_tuple_hiện_tại -> (state_tuple_cha, hành_động_tới_hiện_tại)
        came_from = {initial_state_tuple: None}
        # cost_so_far: dict lưu state_tuple -> g_cost thấp nhất tìm được để tới state đó
        cost_so_far = {initial_state_tuple: 0}

        nodes_expanded = 0

        while pq:
            current_g_cost, _, current_state_list = heapq.heappop(pq)

            current_state_tuple = tuple(map(tuple, current_state_list))

            # Nếu chi phí hiện tại để đến nút này lớn hơn chi phí đã biết (từ một đường khác)
            # thì bỏ qua nút này (vì đã có đường tốt hơn được xử lý hoặc đang trong hàng đợi)
            if current_g_cost > cost_so_far.get(current_state_tuple, float('inf')):
                continue  # Bỏ qua nếu đã tìm thấy đường đi tốt hơn đến trạng thái này rồi

            nodes_expanded += 1

            if current_state_list == goal_state_list:  # Đã tìm thấy trạng thái đích
                path_actions = self._reconstruct_path(came_from, current_state_tuple)
                time_taken = time.time() - start_time
                return {
                    "path_actions": path_actions,
                    "steps": len(path_actions),  # Số bước bằng độ dài path_actions
                    "g_cost": current_g_cost,  # Chi phí thực tế từ điểm bắt đầu
                    "time_taken": time_taken,
                    "nodes_expanded": nodes_expanded,
                    "success": True
                }

            # Duyệt các trạng thái con
            for action, next_state_list, step_cost in self._get_possible_moves(current_state_list):
                new_g_cost = current_g_cost + step_cost  # Chi phí mới để đến trạng thái con
                next_state_tuple = tuple(map(tuple, next_state_list))

                # Nếu trạng thái con chưa được thăm hoặc tìm thấy đường đi mới tốt hơn (chi phí thấp hơn)
                if new_g_cost < cost_so_far.get(next_state_tuple, float('inf')):
                    cost_so_far[next_state_tuple] = new_g_cost  # Cập nhật chi phí thấp nhất
                    came_from[next_state_tuple] = (current_state_tuple, action)  # Lưu vết đường đi
                    heapq.heappush(pq, (new_g_cost, next(unique_id_counter), next_state_list))  # Thêm vào hàng đợi

        # Nếu không tìm thấy đường đi
        time_taken = time.time() - start_time
        return {
            "path_actions": [],
            "steps": 0,
            "g_cost": float('inf'),  # Chi phí là vô cùng nếu không tới được đích
            "time_taken": time_taken,
            "nodes_expanded": nodes_expanded,
            "success": False
        }


if __name__ == '__main__':
    # Ví dụ cách sử dụng (để test)
    solver = UCSSolver()
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
    print(f"Solving puzzle with UCS from {initial} to {goal}")
    result = solver.solve(initial, goal)

    if result["success"]:
        print(f"Solution found. Cost (g): {result['g_cost']}, Steps: {result['steps']}")
        print(f"Path (actions): {result['path_actions']}")
    else:
        print("No solution found.")

    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Nodes expanded: {result['nodes_expanded']}")