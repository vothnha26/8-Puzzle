# forward_checking_solver.py
import time


class ForwardCheckingSolver:
    def __init__(self):
        self.nodes_expanded_count = 0
        self.path_actions_solution = []

    def _find_blank_position(self, state_list):
        for r_idx, row in enumerate(state_list):
            for c_idx, val in enumerate(row):
                if val == 0:
                    return r_idx, c_idx
        return None

    def _get_possible_moves(self, state_list):
        moves = []
        blank_r, blank_c = self._find_blank_position(state_list)
        # (action_name, dr_for_blank, dc_for_blank)
        possible_actions_order = [
            ("UP", -1, 0), ("DOWN", 1, 0),
            ("LEFT", 0, -1), ("RIGHT", 0, 1)
        ]
        for action_name, dr, dc in possible_actions_order:
            new_blank_r, new_blank_c = blank_r + dr, blank_c + dc  # Vị trí mới của ô trống

            if 0 <= new_blank_r < 3 and 0 <= new_blank_c < 3:  # Kiểm tra vị trí mới của ô trống có hợp lệ không
                new_state_list = [row[:] for row in state_list]
                # Tráo đổi ô trống với ô số tại vị trí mới của ô trống
                new_state_list[blank_r][blank_c] = new_state_list[new_blank_r][new_blank_c]
                new_state_list[new_blank_r][new_blank_c] = 0
                moves.append((action_name, new_state_list))
        return moves

    def _perform_forward_check(self, state_to_check_list, parent_of_state_to_check_tuple,
                               visited_on_current_path_tuples):
        """
        Kiểm tra xem có phải tất cả các con của state_to_check_list đều là trạng thái cha của nó
        (parent_of_state_to_check_tuple) hoặc đã có trong visited_on_current_path_tuples (bao gồm cả cha và các nút trước đó).
        Trả về True nếu state_to_check_list là 'ngõ cụt cục bộ', False nếu không.
        """
        children_moves = self._get_possible_moves(state_to_check_list)

        if not children_moves:
            return True

        all_children_problematic = True
        for _, grandchild_list in children_moves:
            grandchild_tuple = tuple(map(tuple, grandchild_list))

            # Một 'cháu' là 'an toàn' (không problematic) nếu nó KHÔNG PHẢI là cha trực tiếp (parent_of_state_to_check_tuple)
            # VÀ nó cũng KHÔNG nằm trên đường đi hiện tại (visited_on_current_path_tuples,
            # mà lúc này đã chứa parent_of_state_to_check_tuple và các nút trước đó).
            # Nói cách khác, một 'cháu' là problematic nếu nó là cha, HOẶC nó đã nằm trên đường đi.
            if grandchild_tuple == parent_of_state_to_check_tuple or \
                    grandchild_tuple in visited_on_current_path_tuples:
                continue
            else:
                all_children_problematic = False
                break

        return all_children_problematic

    def _fc_recursive(self, current_state_list, goal_state_list, visited_on_current_path_tuples, current_depth,
                      max_depth):
        if current_depth > max_depth:
            return "DEPTH_LIMIT_REACHED"

        self.nodes_expanded_count += 1
        current_state_tuple = tuple(map(tuple, current_state_list))

        if current_state_list == goal_state_list:
            return "FOUND"

        visited_on_current_path_tuples.add(current_state_tuple)
        possible_next_moves = self._get_possible_moves(current_state_list)

        # Theo dõi trạng thái trả về từ các lệnh gọi đệ quy con
        # Nếu bất kỳ nhánh con nào bị giới hạn độ sâu, chúng ta cần biết điều đó.
        hit_depth_limit_in_subtree = False

        for action, next_state_l in possible_next_moves:
            next_state_t = tuple(map(tuple, next_state_l))
            if next_state_t not in visited_on_current_path_tuples:

                # --- Bước Forward Checking ---
                # visited_on_current_path_tuples hiện chứa current_state_tuple và các cha của nó.
                if self._perform_forward_check(next_state_l, current_state_tuple, visited_on_current_path_tuples):
                    continue

                self.path_actions_solution.append(action)
                result = self._fc_recursive(next_state_l, goal_state_list, visited_on_current_path_tuples,
                                            current_depth + 1, max_depth)

                if result == "FOUND":
                    return "FOUND"
                if result == "DEPTH_LIMIT_REACHED":
                    hit_depth_limit_in_subtree = True  # Ghi nhận nếu một nhánh con bị giới hạn độ sâu

                self.path_actions_solution.pop()  # Quay lui hành động

        visited_on_current_path_tuples.remove(current_state_tuple)

        if hit_depth_limit_in_subtree:
            return "DEPTH_LIMIT_REACHED"  # Nếu có nhánh con bị giới hạn, báo cáo điều đó
        return "NO_SOLUTION_IN_PATH"  # Nếu không tìm thấy và không bị giới hạn độ sâu ở đây

    def solve(self, initial_state_list, goal_state_list, recursion_depth_limit=50):  # Giới hạn độ sâu mặc định
        start_time = time.time()
        self.nodes_expanded_count = 0
        self.path_actions_solution = []

        visited_on_path = set()
        status = self._fc_recursive(initial_state_list, goal_state_list, visited_on_path, 0, recursion_depth_limit)
        time_taken = time.time() - start_time

        reason_msg = ""
        if status == "FOUND":
            reason_msg = "Goal reached."
            g_cost = len(self.path_actions_solution)
            return {
                "path_actions": list(self.path_actions_solution),
                "steps": g_cost, "g_cost": g_cost, "h_cost": 0,
                "final_state_list": goal_state_list,
                "time_taken": time_taken, "nodes_expanded": self.nodes_expanded_count,
                "success": True, "reason": reason_msg
            }
        else:
            if status == "DEPTH_LIMIT_REACHED":
                reason_msg = f"Search stopped at depth limit ({recursion_depth_limit})."
            else:  # NO_SOLUTION_IN_PATH
                reason_msg = "No solution found (all paths explored up to depth limit)."

            return {
                "path_actions": [], "steps": 0, "g_cost": 0, "h_cost": "N/A",
                "final_state_list": initial_state_list,
                "time_taken": time_taken, "nodes_expanded": self.nodes_expanded_count,
                "success": False, "reason": reason_msg
            }


if __name__ == '__main__':
    solver = ForwardCheckingSolver()
    initial = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    print(f"Solving puzzle with Forward Checking enhanced Backtracking from {initial} to {goal}")
    # Tăng recursion_depth_limit nếu cần, nhưng cẩn thận
    result = solver.solve(initial, goal, recursion_depth_limit=35)  # 8-puzzle thường có giải pháp <30

    print(f"Success: {result['success']}")
    print(f"Reason: {result['reason']}")
    if result['success']:
        print(f"Steps (g): {result['g_cost']}")
        print(f"Path (actions): {result['path_actions']}")
    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Nodes expanded: {result['nodes_expanded']}")