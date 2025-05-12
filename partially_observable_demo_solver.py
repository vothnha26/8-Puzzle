# partially_observable_demo_solver.py
import random
import time


class PartiallyObservableDemo:
    def __init__(self):
        self.grid_size = 3
        self.all_tiles = list(range(9))  # Các ô số từ 0 đến 8
        self.simulation_cycles_count = 0
        self._goal_positions_cache = {}
        self.target_state_for_heuristic_calc = None  # Dùng để tính heuristic

    def _find_blank_in_state(self, state_list_2d):
        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                if state_list_2d[r_idx][c_idx] == 0:
                    return r_idx, c_idx
        return None

    def _is_valid_pos(self, r, c):
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _generate_random_state(self):
        """Tạo một trạng thái 8-puzzle ngẫu nhiên hợp lệ."""
        tiles = list(self.all_tiles)
        random.shuffle(tiles)
        state = []
        for i in range(0, self.grid_size * self.grid_size, self.grid_size):
            state.append(tiles[i:i + self.grid_size])
        return state

    def _apply_action_to_state(self, state_list_2d, action_str):
        """
        Áp dụng hành động lên một trạng thái. Trả về (new_state, moved_successfully).
        """
        br, bc = self._find_blank_in_state(state_list_2d)
        if br is None: return [row[:] for row in state_list_2d], False  # Không có ô trống

        dr, dc = 0, 0
        if action_str == "UP":
            dr = -1
        elif action_str == "DOWN":
            dr = 1
        elif action_str == "LEFT":
            dc = -1
        elif action_str == "RIGHT":
            dc = 1

        target_r, target_c = br + dr, bc + dc  # Ô mà ô trống muốn tráo đổi với
        new_state = [row[:] for row in state_list_2d]

        if not self._is_valid_pos(target_r, target_c):
            return new_state, False  # Nước đi không hợp lệ (ra ngoài biên)

        # Nước đi hợp lệ: tráo đổi ô trống với ô ở target_r, target_c
        tile_in_target_cell = new_state[target_r][target_c]
        new_state[br][bc] = tile_in_target_cell
        new_state[target_r][target_c] = 0  # Ô trống di chuyển đến vị trí mới
        return new_state, True

    def _check_observation(self, state_list_2d, observation_type, observation_param=None):
        """
        Kiểm tra xem một trạng thái có nhất quán với một quan sát không.
        """
        br_obs, bc_obs = self._find_blank_in_state(state_list_2d)
        if br_obs is None and observation_type is not None and "BLANK" in observation_type:
            return False  # Trạng thái không hợp lệ nếu cần thông tin ô trống

        if observation_type == "BLANK_IN_ROW":
            return br_obs is not None and br_obs == observation_param
        elif observation_type == "BLANK_IN_COL":
            return bc_obs is not None and bc_obs == observation_param
        elif observation_type == "TILE_AT_POS":
            if observation_param is None: return False
            (r, c), tile_val = observation_param
            if self._is_valid_pos(r, c):
                return state_list_2d[r][c] == tile_val
        return False  # Loại quan sát không xác định hoặc tham số không hợp lệ

    def _precompute_goal_positions(self, goal_state_for_heuristic):
        goal_state_tuple_key = tuple(map(tuple, goal_state_for_heuristic))
        if goal_state_tuple_key in self._goal_positions_cache:
            return self._goal_positions_cache[goal_state_tuple_key]
        positions = {}
        for r_idx, row in enumerate(goal_state_for_heuristic):
            for c_idx, tile_value in enumerate(row):
                if tile_value != 0:
                    positions[tile_value] = (r_idx, c_idx)
        self._goal_positions_cache[goal_state_tuple_key] = positions
        return positions

    def _calculate_manhattan_distance(self, current_state_list, target_state_for_h):
        if not target_state_for_h or not current_state_list: return float('inf')
        # Đảm bảo vị trí đích được tính toán trước cho target_state_for_h này
        self._precompute_goal_positions(target_state_for_h)

        goal_positions = self._goal_positions_cache.get(tuple(map(tuple, target_state_for_h)))
        if not goal_positions: return float('inf')  # Không nên xảy ra nếu precompute hoạt động

        total_manhattan_distance = 0
        for r_idx, row in enumerate(current_state_list):
            for c_idx, tile_value in enumerate(row):
                if tile_value != 0:
                    if tile_value in goal_positions:
                        goal_r, goal_c = goal_positions[tile_value]
                        total_manhattan_distance += abs(r_idx - goal_r) + abs(c_idx - goal_c)
        return total_manhattan_distance

    def get_inverse_action(self, action_str):
        if action_str == "UP": return "DOWN"
        if action_str == "DOWN": return "UP"
        if action_str == "LEFT": return "RIGHT"
        if action_str == "RIGHT": return "LEFT"
        return None

    def run_path_like_demo(self, num_initial_beliefs=2, path_len_to_generate=3):
        start_time = time.time()
        self.simulation_cycles_count = 0

        # 1. Tạo trạng thái đích ngẫu nhiên
        self.target_state_for_heuristic_calc = self._generate_random_state()

        # 2. Tạo một "trạng thái thật ban đầu" bằng cách đi ngược từ đích
        true_initial_state_for_path = [row[:] for row in self.target_state_for_heuristic_calc]
        scramble_actions = []

        possible_agent_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        for _ in range(path_len_to_generate):
            valid_scramble_action = None
            # Thử các hành động ngẫu nhiên để xáo trộn
            shuffled_actions = random.sample(possible_agent_actions, len(possible_agent_actions))
            for candidate_action in shuffled_actions:
                # Tránh hành động ngược lại ngay lập tức nếu có thể
                if scramble_actions and candidate_action == self.get_inverse_action(scramble_actions[-1]):
                    if len(shuffled_actions) > 1: continue

                temp_next_state, moved = self._apply_action_to_state(true_initial_state_for_path, candidate_action)
                if moved and temp_next_state != true_initial_state_for_path:  # Đảm bảo di chuyển hiệu quả
                    valid_scramble_action = candidate_action
                    true_initial_state_for_path = temp_next_state
                    break  # Đã tìm thấy nước đi xáo trộn hợp lệ

            if valid_scramble_action:
                scramble_actions.append(valid_scramble_action)
            else:  # Không tìm thấy nước đi xáo trộn hợp lệ nào nữa
                break

                # Chuỗi hành động để giải là ngược lại của scramble_actions, theo thứ tự ngược lại
        solution_actions_for_true_path = [self.get_inverse_action(act) for act in reversed(scramble_actions)]

        # Xử lý trường hợp không xáo trộn được (ví dụ: path_len_to_generate = 0)
        if not solution_actions_for_true_path and true_initial_state_for_path == self.target_state_for_heuristic_calc:
            action_if_stuck = random.choice(possible_agent_actions)
            temp_s, moved = self._apply_action_to_state(true_initial_state_for_path, action_if_stuck)
            if moved:
                true_initial_state_for_path = temp_s
                solution_actions_for_true_path = [self.get_inverse_action(action_if_stuck)]

        # 3. Tạo tập niềm tin ban đầu
        # Trạng thái đầu tiên trong niềm tin là "trạng thái thật" đã được tạo
        initial_belief_states_list = [[row[:] for row in true_initial_state_for_path]]
        # Thêm các trạng thái niềm tin ngẫu nhiên khác (nếu num_initial_beliefs > 1)
        for _ in range(1, num_initial_beliefs):
            s = self._generate_random_state()
            # Đảm bảo các trạng thái niềm tin khác nhau và khác trạng thái đích
            while any(s == bs for bs in initial_belief_states_list) or s == self.target_state_for_heuristic_calc:
                s = self._generate_random_state()
            initial_belief_states_list.append(s)

        current_belief_set = [[row[:] for row in bs] for bs in initial_belief_states_list]
        current_true_state = [row[:] for row in true_initial_state_for_path]  # Theo dõi trạng thái "thật"

        action_observation_log_for_ui = []  # Lưu (Hành động, Quan sát) để hiển thị

        for i, action_to_take in enumerate(solution_actions_for_true_path):
            self.simulation_cycles_count += 1

            # Áp dụng hành động lên trạng thái "thật" để biết quan sát "đúng"
            next_true_state, moved_true = self._apply_action_to_state(current_true_state, action_to_take)
            if not moved_true:  # Hành động theo kế hoạch không hợp lệ với trạng thái thật (hiếm)
                action_observation_log_for_ui.append(
                    f"Cycle {i + 1}: Planned Act='{action_to_take}' invalid for true state. Demo ends.")
                break
            current_true_state = next_true_state

            # Tạo quan sát dựa trên trạng thái thật mới (ví dụ: vị trí ô trống)
            br_true, bc_true = self._find_blank_in_state(current_true_state)
            obs_type, obs_param = "BLANK_IN_ROW", br_true
            # Thay đổi loại quan sát để đa dạng hơn
            if i % 2 == 1 and bc_true is not None: obs_type, obs_param = "BLANK_IN_COL", bc_true

            action_observation_log_for_ui.append(
                f"Cycle {i + 1}: Agent Acts='{action_to_take}', Receives Obs='{obs_type}({obs_param})'")

            # Bước dự đoán: Áp dụng hành động cho tất cả trạng thái trong niềm tin
            predicted_belief_set_after_action = []
            for belief_instance in current_belief_set:
                new_instance_state, _ = self._apply_action_to_state(belief_instance, action_to_take)
                predicted_belief_set_after_action.append(new_instance_state)

            # Bước cập nhật: Lọc tập niềm tin dựa trên quan sát
            updated_belief_set_after_observation = []
            for predicted_state in predicted_belief_set_after_action:
                if self._check_observation(predicted_state, obs_type, obs_param):
                    updated_belief_set_after_observation.append(predicted_state)

            current_belief_set = updated_belief_set_after_observation
            if not current_belief_set:  # Tập niềm tin rỗng, dừng mô phỏng
                action_observation_log_for_ui.append(f"  -> Belief set became empty after observation.")
                break
            else:
                action_observation_log_for_ui.append(f"  -> Updated belief set size: {len(current_belief_set)}")

        time_taken = time.time() - start_time

        success_this_demo = False
        final_representative_state = None
        final_h_cost = float('inf')

        if current_belief_set:
            # Thành công nếu trạng thái thật cuối cùng (là đích) CÓ trong tập niềm tin cuối
            if current_true_state == self.target_state_for_heuristic_calc and \
                    any(bs == self.target_state_for_heuristic_calc for bs in current_belief_set):
                success_this_demo = True
                final_representative_state = self.target_state_for_heuristic_calc

            if not final_representative_state:  # Nếu không, lấy cái đầu tiên làm đại diện
                final_representative_state = current_belief_set[0]
            final_h_cost = self._calculate_manhattan_distance(final_representative_state,
                                                              self.target_state_for_heuristic_calc)
        else:  # Tập niềm tin rỗng
            # Lấy trạng thái ban đầu làm đại diện nếu không còn gì
            final_representative_state = initial_belief_states_list[
                0] if initial_belief_states_list else self._generate_random_state()
            final_h_cost = self._calculate_manhattan_distance(final_representative_state,
                                                              self.target_state_for_heuristic_calc)

        reason = f"{self.simulation_cycles_count} cycles completed."
        if success_this_demo:
            reason = "Target state consistent with final belief after planned actions."
        elif not current_belief_set:
            reason = "Belief set became empty during simulation."

        return {
            "initial_state_sample": initial_belief_states_list[0] if initial_belief_states_list else None,
            "target_state": self.target_state_for_heuristic_calc,  # Trạng thái đích của demo
            "final_representative_state": final_representative_state,  # Trạng thái đại diện cuối cùng
            "path_actions": action_observation_log_for_ui,  # Log các (Hành động, Quan sát)
            "time_taken": time_taken,
            "nodes_expanded": self.simulation_cycles_count,  # Số chu trình mô phỏng
            "reason": reason,
            "success": success_this_demo,
            "g_cost": len(solution_actions_for_true_path),  # Số bước của đường đi dự kiến để giải
            "h_cost": final_h_cost,  # h_cost của trạng thái đại diện cuối so với đích
            "final_belief_set_size": len(current_belief_set)
        }