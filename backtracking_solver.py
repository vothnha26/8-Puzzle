# backtracking_solver.py
import time


class BacktrackingSolver:
    def __init__(self):
        self.nodes_expanded_count = 0
        # Không cần self.path_actions_solution nữa vì đường đi sẽ được lưu trên stack

    def _find_blank_position(self, state_list):
        for r_idx, row in enumerate(state_list):
            for c_idx, val in enumerate(row):
                if val == 0:
                    return r_idx, c_idx
        return None

    def _get_possible_moves(self, state_list):
        """Trả về list các tuple (action_name, next_state_list)."""
        moves = []
        blank_r, blank_c = self._find_blank_position(state_list)

        # Thứ tự các hành động này có thể ảnh hưởng đến giải pháp đầu tiên được tìm thấy
        # Ưu tiên các hướng thường dùng trong DFS: UP, LEFT, DOWN, RIGHT (hoặc tương tự)
        action_deltas = [
            ("UP", -1, 0),
            ("LEFT", 0, -1),
            ("DOWN", 1, 0),
            ("RIGHT", 0, 1)
        ]

        for action_name, dr, dc in action_deltas:
            tile_to_swap_r, tile_to_swap_c = blank_r + dr, blank_c + dc
            if 0 <= tile_to_swap_r < 3 and 0 <= tile_to_swap_c < 3:
                new_state_list = [row[:] for row in state_list]
                new_state_list[blank_r][blank_c] = new_state_list[tile_to_swap_r][tile_to_swap_c]
                new_state_list[tile_to_swap_r][tile_to_swap_c] = 0
                moves.append((action_name, new_state_list))
        return moves

    def solve(self, initial_state_list, goal_state_list):
        start_time = time.time()
        self.nodes_expanded_count = 0

        initial_state_tuple = tuple(map(tuple, initial_state_list))

        # Stack lưu trữ: (current_state_list, actions_taken_so_far, visited_on_this_path_tuples)
        # visited_on_this_path_tuples là một set các state_tuple đã thăm trên đường đi DẪN ĐẾN current_state_list
        stack = []
        stack.append(
            (initial_state_list, [], {initial_state_tuple}))  # Trạng thái ban đầu, không hành động, đã thăm chính nó

        while stack:
            current_state_l, current_actions, visited_path = stack.pop()  # Lấy phần tử từ đỉnh stack (LIFO)
            self.nodes_expanded_count += 1

            current_state_t = tuple(map(tuple, current_state_l))  # Chuyển về tuple để so sánh (nếu cần)

            if current_state_l == goal_state_list:
                time_taken = time.time() - start_time
                return {
                    "path_actions": current_actions,
                    "steps": len(current_actions),
                    "g_cost": len(current_actions),
                    "h_cost": 0,
                    "final_state_list": goal_state_list,
                    "time_taken": time_taken,
                    "nodes_expanded": self.nodes_expanded_count,
                    "success": True,
                    "reason": "Goal reached."
                }

            # Để mô phỏng đúng thứ tự duyệt của DFS đệ quy (ví dụ: UP trước, rồi LEFT,...),
            # khi thêm các con vào stack (LIFO), ta nên thêm chúng theo thứ tự ngược lại
            # so với thứ tự mà _get_possible_moves trả về nếu ta muốn xử lý chúng theo thứ tự đó.
            # Ví dụ: nếu _get_possible_moves trả về [UP_move, LEFT_move, DOWN_move, RIGHT_move]
            # và ta muốn thử UP_move trước, thì phải push RIGHT_move, rồi DOWN_move, rồi LEFT_move, rồi UP_move.

            possible_next_moves = self._get_possible_moves(current_state_l)
            for action, next_state_l in reversed(possible_next_moves):  # Duyệt ngược để push vào stack
                next_state_t = tuple(map(tuple, next_state_l))
                if next_state_t not in visited_path:  # Chỉ đi tiếp nếu chưa thăm trên đường đi này
                    new_actions = current_actions + [action]
                    new_visited_path = visited_path.copy()  # Tạo bản sao set cho nhánh mới
                    new_visited_path.add(next_state_t)  # Thêm trạng thái con vào tập đã thăm của nhánh mới
                    stack.append((next_state_l, new_actions, new_visited_path))

        # Nếu stack rỗng mà không tìm thấy giải pháp
        time_taken = time.time() - start_time
        return {
            "path_actions": [], "steps": 0, "g_cost": 0, "h_cost": "N/A",
            "final_state_list": initial_state_list,
            "time_taken": time_taken, "nodes_expanded": self.nodes_expanded_count,
            "success": False, "reason": "No solution found."
        }


if __name__ == '__main__':
    solver = BacktrackingSolver()
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

    print(f"Solving puzzle with Iterative Backtracking from {initial} to {goal}")
    result = solver.solve(initial, goal)

    print(f"Success: {result['success']}")
    print(f"Reason: {result['reason']}")
    if result['success']:
        print(f"Steps (g): {result['g_cost']}")
        print(f"Path (actions): {result['path_actions']}")
    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Nodes expanded: {result['nodes_expanded']}")

    initial_is_goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    print(f"\nSolving Iterative Backtracking from {initial_is_goal} to {goal}")
    result_ig = solver.solve(initial_is_goal, goal)
    print(f"Success: {result_ig['success']}")  # True
    print(f"Reason: {result_ig['reason']}")  # Goal reached.
    print(f"Steps: {result_ig['steps']}")  # 0
    print(f"Nodes expanded: {result_ig['nodes_expanded']}")  # 1