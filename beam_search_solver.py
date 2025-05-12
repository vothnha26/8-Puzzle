# beam_search_solver.py
import time
import itertools  # Để tạo unique_id cho các phần tử trong beam


class BeamSearchSolver:
    def __init__(self):
        self._goal_positions_cache = {}
        self.goal_state_list_ref = None
        self.nodes_evaluated_count = 0  # Đếm số trạng thái có heuristic được tính

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
                moves.append((action_name, new_state_list, 1))  # action, next_state, step_cost
        return moves

    def _reconstruct_path(self, came_from_dict, goal_state_tuple):
        path_actions = []
        temp_current_tuple = goal_state_tuple
        while temp_current_tuple in came_from_dict and came_from_dict[temp_current_tuple] is not None:
            prev_state_tuple, action = came_from_dict[temp_current_tuple]
            path_actions.append(action)
            temp_current_tuple = prev_state_tuple
        return path_actions[::-1]

    def solve(self, initial_state_list, goal_state_list, beam_width=3, max_steps=100):
        start_time = time.time()
        self.nodes_evaluated_count = 0
        self.goal_state_list_ref = goal_state_list
        self._precompute_goal_positions(self.goal_state_list_ref)

        initial_state_tuple = tuple(map(tuple, initial_state_list))
        initial_h = self._calculate_manhattan_distance(initial_state_list)
        self.nodes_evaluated_count += 1

        unique_id_counter = itertools.count()

        # Chùm tia (beam) lưu trữ: (heuristic_cost, unique_id, state_list, g_cost)
        # unique_id dùng để đảm bảo thứ tự ổn định nếu heuristic_cost bằng nhau
        # và để heapq không cố gắng so sánh state_list (nếu nó là list)
        beam = [(initial_h, next(unique_id_counter), initial_state_list, 0)]

        # came_from: current_state_tuple -> (parent_state_tuple, action)
        # Dùng để truy vết đường đi cho BẤT KỲ trạng thái nào được thêm vào beam
        came_from = {initial_state_tuple: None}

        # visited_states: lưu các state_tuple đã từng được mở rộng (tức là con của nó đã được tạo ra)
        # để tránh lặp lại và chu trình trong quá trình tìm kiếm tổng thể.
        visited_states = {initial_state_tuple}

        best_state_overall = initial_state_list  # Trạng thái tốt nhất tìm được nếu không đạt đích
        best_h_overall = initial_h
        best_g_overall = 0

        for step_count in range(max_steps):
            if not beam:  # Chùm tia rỗng, không còn trạng thái để khám phá
                break

            next_beam_candidates = []  # (h_cost, unique_id, state_list, g_cost, parent_tuple, action)

            # Kiểm tra xem có trạng thái đích nào trong chùm tia hiện tại không
            for h_val, _, state_l, g_val in beam:
                if state_l == self.goal_state_list_ref:
                    path_actions = self._reconstruct_path(came_from, tuple(map(tuple, state_l)))
                    time_taken = time.time() - start_time
                    return {
                        "path_actions": path_actions, "steps": g_val, "g_cost": g_val,
                        "h_cost": 0, "final_state_list": state_l, "time_taken": time_taken,
                        "nodes_expanded": self.nodes_evaluated_count,
                        "success": True, "reason": "Goal reached."
                    }
                # Cập nhật trạng thái tốt nhất (không phải đích) đã gặp
                if h_val < best_h_overall:
                    best_h_overall = h_val
                    best_state_overall = state_l
                    best_g_overall = g_val  # g_cost để đến trạng thái tốt nhất này

            # Mở rộng các nút trong chùm tia hiện tại
            for _, _, current_s_list, current_g in beam:
                current_s_tuple = tuple(map(tuple, current_s_list))

                for action, next_s_list, step_c in self._get_possible_moves(current_s_list):
                    next_s_tuple = tuple(map(tuple, next_s_list))

                    # Chỉ xem xét nếu trạng thái này chưa từng được mở rộng trước đó
                    # (Beam Search có thể tạo lại các trạng thái đã thăm qua các đường khác nhau)
                    if next_s_tuple not in visited_states:
                        visited_states.add(next_s_tuple)  # Đánh dấu đã mở rộng

                        h_next = self._calculate_manhattan_distance(next_s_list)
                        self.nodes_evaluated_count += 1
                        g_next = current_g + step_c

                        # Thêm vào danh sách ứng cử viên cho chùm tia tiếp theo
                        # Lưu cả parent và action để cập nhật came_from nếu nó được chọn
                        next_beam_candidates.append(
                            (h_next, next(unique_id_counter), next_s_list, g_next, current_s_tuple, action))

            if not next_beam_candidates:
                break  # Không có ứng cử viên mới

            # Sắp xếp các ứng cử viên: ưu tiên heuristic thấp, sau đó là g_cost thấp (để phá vỡ sự bằng nhau)
            next_beam_candidates.sort(key=lambda x: (x[0], x[3]))

            # Xây dựng chùm tia mới và cập nhật came_from
            new_beam_temp = []
            # Set để đảm bảo không có trạng thái trùng lặp trong chùm tia mới (chọn đường đi tốt nhất đến nó)
            # Tuy nhiên, với visited_states ở trên, điều này ít xảy ra hơn.
            # Cách đơn giản là chỉ lấy top k.

            temp_came_from_updates = {}  # Các cập nhật came_from tiềm năng cho beam mới

            for h_val, uid, state_l, g_val, parent_t, act in next_beam_candidates[:beam_width]:
                new_beam_temp.append((h_val, uid, state_l, g_val))
                # Chỉ cập nhật came_from cho những nút thực sự được chọn vào beam mới
                state_t = tuple(map(tuple, state_l))
                # Nếu một nút có thể được tạo từ nhiều cha trong beam trước,
                # came_from nên phản ánh đường đi dẫn đến nó được chọn vào beam mới.
                # Do next_beam_candidates đã được sắp xếp, đường đi đầu tiên đến state_t sẽ là "tốt nhất" theo tiêu chí sắp xếp.
                if state_t not in temp_came_from_updates:  # Chỉ lấy đường đi đầu tiên (tốt nhất)
                    temp_came_from_updates[state_t] = (parent_t, act)

            beam = new_beam_temp
            came_from.update(temp_came_from_updates)  # Áp dụng các cập nhật came_from

        # Kết thúc vòng lặp (do max_steps, chùm tia rỗng, hoặc không có ứng cử viên)
        time_taken = time.time() - start_time

        # Trả về thông tin về trạng thái tốt nhất tìm được nếu không đến đích
        path_to_best_overall = self._reconstruct_path(came_from, tuple(map(tuple, best_state_overall)))

        reason_text = "Max steps reached or beam became empty."
        if best_h_overall == 0: reason_text = "Goal found by best_state_overall tracker (rare)."

        return {
            "path_actions": path_to_best_overall,
            "steps": len(path_to_best_overall),
            "g_cost": best_g_overall,  # g_cost của best_state_overall
            "h_cost": best_h_overall,  # h_cost của best_state_overall
            "final_state_list": best_state_overall,
            "time_taken": time_taken,
            "nodes_expanded": self.nodes_evaluated_count,
            "success": (best_h_overall == 0),  # Thành công nếu trạng thái tốt nhất là đích
            "reason": reason_text + f" Best h={best_h_overall}."
        }


if __name__ == '__main__':
    solver = BeamSearchSolver()
    initial = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    beam_k = 5  # Độ rộng chùm tia
    max_s = 50  # Số bước tối đa
    print(f"Solving puzzle with Beam Search (k={beam_k}, max_steps={max_s}, Manhattan) from {initial} to {goal}")
    result = solver.solve(initial, goal, beam_width=beam_k, max_steps=max_s)

    print(f"Success: {result['success']}")
    print(f"Reason: {result['reason']}")
    print(f"Path steps (to best/goal): {result['steps']}, Best state h: {result['h_cost']}")
    print(f"Path (actions): {result['path_actions']}")
    # print(f"Best state found: {result['final_state_list']}")
    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Nodes evaluated: {result['nodes_expanded']}")