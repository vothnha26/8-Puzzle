# iddfs_solver.py
import time
import itertools  # Mặc dù không dùng trong phiên bản này, giữ lại cho các ý tưởng tương lai


class IDDFSSolver:
    def __init__(self):
        pass

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
        ]  # Thứ tự này có thể ảnh hưởng đến nhánh nào được duyệt trước trong DLS

        for dr, dc, action_name in possible_actions:
            new_r, new_c = blank_r + dr, blank_c + dc
            if 0 <= new_r < 3 and 0 <= new_c < 3:
                new_state_list = [row[:] for row in state_list]
                new_state_list[blank_r][blank_c] = new_state_list[new_r][new_c]
                new_state_list[new_r][new_c] = 0
                # action, next_state, step_cost (luôn là 1 cho 8-puzzle)
                moves.append((action_name, new_state_list, 1))
        return moves

    def _reconstruct_path(self, came_from, current_state_tuple):
        """Truy vết đường đi từ đích về nguồn."""
        path_actions = []
        while current_state_tuple in came_from and came_from[current_state_tuple] is not None:
            prev_state_tuple, action = came_from[current_state_tuple]
            path_actions.append(action)
            current_state_tuple = prev_state_tuple
        return path_actions[::-1]

    def _dls(self, initial_state_list, goal_state_list, current_depth_limit, nodes_expanded_ref):
        """
        Depth-Limited Search (DLS).
        nodes_expanded_ref là một list [count] để truyền giá trị kiểu tham chiếu.
        Trả về một dictionary chứa kết quả của DLS cho depth_limit hiện tại.
        """
        initial_state_tuple = tuple(map(tuple, initial_state_list))

        # Stack lưu: (state_list, current_path_depth_to_this_state)
        stack = [(initial_state_list, 0)]

        # came_from_this_dls: dùng để truy vết đường đi NẾU tìm thấy giải pháp ở độ sâu này.
        # Nó phải được làm mới cho mỗi lần gọi DLS (tức là cho mỗi depth_limit mới).
        came_from_this_dls = {initial_state_tuple: None}

        # visited_in_this_dls: Set để tránh lặp trong một đường dẫn của DLS hiện tại.
        # Đối với IDDFS, việc "reset" visited cho mỗi DLS iteration (mỗi depth_limit mới) là quan trọng.
        # visited_in_this_dls lưu trữ các state_tuple đã được đưa vào stack để xử lý trong DLS này.
        visited_in_this_dls = {initial_state_tuple}

        while stack:
            current_state_list, path_depth = stack.pop()
            nodes_expanded_ref[0] += 1  # Tăng tổng số nút đã duyệt qua tất cả các DLS calls

            current_state_tuple = tuple(map(tuple, current_state_list))

            if current_state_list == goal_state_list:
                # Tìm thấy giải pháp, trả về thông tin cần thiết để dựng lại đường đi
                return {"found": True, "goal_state_tuple": current_state_tuple, "came_from_dict": came_from_this_dls}

            if path_depth < current_depth_limit:  # Chỉ mở rộng nếu chưa vượt quá giới hạn độ sâu
                # Để duyệt theo thứ tự "UP, DOWN, LEFT, RIGHT" (nếu _get_possible_moves trả về theo thứ tự đó)
                # thì cần push vào stack theo thứ tự ngược lại (RIGHT, LEFT, DOWN, UP)
                successors = self._get_possible_moves(current_state_list)
                for action, next_state_list, _ in reversed(successors):
                    next_state_tuple = tuple(map(tuple, next_state_list))

                    # Chỉ thêm vào stack nếu trạng thái này chưa được thăm TRONG DLS HIỆN TẠI
                    # Hoặc, một cách tiếp cận khác là không dùng visited_in_this_dls và cho phép lặp lại
                    # việc duyệt các nút trên các nhánh khác nhau (nhưng DLS vẫn bị giới hạn bởi depth_limit).
                    # Sử dụng visited_in_this_dls giúp tránh lãng phí trong một DLS call.
                    if next_state_tuple not in visited_in_this_dls:
                        visited_in_this_dls.add(next_state_tuple)  # Đánh dấu đã thăm trong DLS này
                        came_from_this_dls[next_state_tuple] = (current_state_tuple, action)
                        stack.append((next_state_list, path_depth + 1))

        return {"found": False}  # Không tìm thấy giải pháp ở độ sâu này

    def solve(self, initial_state_list, goal_state_list, max_iterations=30):
        """
        Thực hiện Iterative Deepening DFS.
        max_iterations hoạt động như một giới hạn độ sâu tối đa tổng thể.
        """
        start_time = time.time()
        total_nodes_expanded = [0]  # Dùng list để truyền kiểu tham chiếu cho số nguyên

        for current_max_depth in range(max_iterations + 1):  # Lặp qua các giới hạn độ sâu
            # Mỗi lần gọi DLS, came_from và visited phải được "reset" (DLS tự quản lý came_from của nó)
            dls_result = self._dls(initial_state_list, goal_state_list, current_max_depth, total_nodes_expanded)

            if dls_result["found"]:
                final_came_from = dls_result["came_from_dict"]
                goal_state_tuple = dls_result["goal_state_tuple"]
                path_actions = self._reconstruct_path(final_came_from, goal_state_tuple)
                time_taken = time.time() - start_time
                return {
                    "path_actions": path_actions,
                    "steps": len(path_actions),  # Số bước bằng giới hạn độ sâu khi tìm thấy
                    "g_cost": len(path_actions),  # Chi phí g bằng số bước (vì chi phí mỗi bước là 1)
                    "time_taken": time_taken,
                    "nodes_expanded": total_nodes_expanded[0],
                    "success": True,
                    "depth_reached": current_max_depth  # Độ sâu mà giải pháp được tìm thấy
                }

        # Nếu vòng lặp kết thúc mà không tìm thấy giải pháp
        time_taken = time.time() - start_time
        return {
            "path_actions": [],
            "steps": 0,
            "g_cost": float('inf'),
            "time_taken": time_taken,
            "nodes_expanded": total_nodes_expanded[0],
            "success": False,
            "depth_reached": max_iterations  # Độ sâu cuối cùng đã thử
        }


if __name__ == '__main__':
    solver = IDDFSSolver()
    initial = [
        [1, 8, 2],
        [0, 4, 3],
        [7, 6, 5]
    ]
    # initial = [[1,2,3],[0,4,6],[7,5,8]] # 2 steps to goal
    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    print(f"Solving puzzle with IDDFS from {initial} to {goal}")
    # Giới hạn độ sâu tối đa hợp lý cho 8-puzzle, ví dụ 20-25.
    # Lời giải cho trạng thái ban đầu mặc định thường là 9 bước.
    result = solver.solve(initial, goal, max_iterations=20)

    if result["success"]:
        print(
            f"Solution found at depth {result['depth_reached']}. Cost (g): {result['g_cost']}, Steps: {result['steps']}")
        print(f"Path (actions): {result['path_actions']}")
    else:
        print(f"No solution found within max depth {result['depth_reached']}.")

    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Nodes expanded: {result['nodes_expanded']}")