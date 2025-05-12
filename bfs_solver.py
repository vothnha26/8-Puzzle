# bfs_solver.py
import collections
import time


class BFSSolver:
    def __init__(self):
        pass

    def _find_blank_position(self, state):
        """Tìm vị trí của ô trống (số 0)."""
        for r_idx, row in enumerate(state):
            for c_idx, val in enumerate(row):
                if val == 0:
                    return r_idx, c_idx
        return None

    def _get_possible_moves(self, state):
        """
        Tạo ra các trạng thái kế tiếp hợp lệ từ trạng thái hiện tại.
        Trả về một danh sách các tuple (action, next_state).
        """
        moves = []
        blank_r, blank_c = self._find_blank_position(state)

        # Các hướng di chuyển có thể của ô trống: (dr, dc, action_name)
        # dr: delta row, dc: delta col
        possible_actions = [
            (-1, 0, "UP"),  # Ô trống đi lên (số bên dưới ô trống đi xuống)
            (1, 0, "DOWN"),  # Ô trống đi xuống
            (0, -1, "LEFT"),  # Ô trống đi qua trái
            (0, 1, "RIGHT")  # Ô trống đi qua phải
        ]

        for dr, dc, action_name in possible_actions:
            new_r, new_c = blank_r + dr, blank_c + dc  # Vị trí mới của ô trống

            # Kiểm tra xem vị trí mới của ô trống có hợp lệ không
            # (tức là vị trí của ô số sẽ tráo đổi với ô trống)
            if 0 <= new_r < 3 and 0 <= new_c < 3:
                # Tạo trạng thái mới bằng cách sao chép trạng thái hiện tại
                new_state = [row[:] for row in state]
                # Di chuyển ô số vào vị trí ô trống cũ
                new_state[blank_r][blank_c] = new_state[new_r][new_c]
                # Đặt ô trống vào vị trí mới
                new_state[new_r][new_c] = 0
                moves.append((action_name, new_state))
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

    def solve(self, initial_state, goal_state):
        """
        Thực hiện thuật toán BFS.
        Trả về một dictionary chứa: path (danh sách hành động), steps, time_taken, nodes_expanded, success.
        """
        start_time = time.time()

        # Hàng đợi cho BFS, mỗi phần tử là một trạng thái (dạng list 2D)
        queue = collections.deque([initial_state])

        # Set để lưu các trạng thái đã thăm (dạng tuple để có thể hash)
        # và `came_from` để lưu trữ (trạng thái trước đó, hành động) dẫn đến trạng thái hiện tại
        # Giá trị cho initial_state là None vì nó không có trạng thái trước đó
        visited_states = {tuple(map(tuple, initial_state))}
        came_from = {tuple(map(tuple, initial_state)): None}

        nodes_expanded = 0

        while queue:
            current_state_list = queue.popleft()
            nodes_expanded += 1

            current_state_tuple = tuple(map(tuple, current_state_list))

            if current_state_list == goal_state:
                path_actions = self._reconstruct_path(came_from, current_state_tuple)
                time_taken = time.time() - start_time
                return {
                    "path_actions": path_actions,
                    "steps": len(path_actions),
                    "time_taken": time_taken,
                    "nodes_expanded": nodes_expanded,
                    "success": True
                }

            for action, next_state_list in self._get_possible_moves(current_state_list):
                next_state_tuple = tuple(map(tuple, next_state_list))
                if next_state_tuple not in visited_states:
                    visited_states.add(next_state_tuple)
                    came_from[next_state_tuple] = (current_state_tuple, action)
                    queue.append(next_state_list)

        time_taken = time.time() - start_time
        return {
            "path_actions": [],
            "steps": 0,
            "time_taken": time_taken,
            "nodes_expanded": nodes_expanded,
            "success": False  # Không tìm thấy đường đi
        }


if __name__ == '__main__':
    # Ví dụ cách sử dụng (để test)
    solver = BFSSolver()

    initial = [
        [1, 8, 2],
        [0, 4, 3],
        [7, 6, 5]
    ]
    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]  # Solvable in 9 steps by A* (Manhattan), BFS might take more nodes but same path length

    print(f"Solving puzzle from {initial} to {goal}")
    result = solver.solve(initial, goal)

    if result["success"]:
        print(f"Solution found in {result['steps']} steps.")
        print(f"Path (actions): {result['path_actions']}")
        # print(f"Solution path (states):")
        # current_s = [r[:] for r in initial]
        # for i, action in enumerate(result['path_actions']):
        #     print(f"Step {i+1}: {action}")
        #     # (Trong thực tế, bạn cần hàm để áp dụng action vào state để thấy state mới)
        #     # Hoặc BFS trả về path_states
    else:
        print("No solution found.")

    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Nodes expanded: {result['nodes_expanded']}")