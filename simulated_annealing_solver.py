# simulated_annealing_solver.py
import time
import random
import math  # Để sử dụng math.exp()


class SimulatedAnnealingSolver:
    def __init__(self):
        self._goal_positions_cache = {}
        self.goal_state_list_ref = None
        # "nodes_expanded" trong SA thường được hiểu là số vòng lặp (iterations)
        self.iterations_count = 0

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

    def _get_random_neighbor(self, state_list):
        """Chọn một láng giềng ngẫu nhiên và trả về (action, next_state_list)."""
        blank_r, blank_c = self._find_blank_position(state_list)
        possible_moves_tuples = []  # Lưu trữ (dr, dc, action_name)

        # Lên
        if blank_r > 0: possible_moves_tuples.append((-1, 0, "UP"))
        # Xuống
        if blank_r < 2: possible_moves_tuples.append((1, 0, "DOWN"))
        # Trái
        if blank_c > 0: possible_moves_tuples.append((0, -1, "LEFT"))
        # Phải
        if blank_c < 2: possible_moves_tuples.append((0, 1, "RIGHT"))

        if not possible_moves_tuples:
            return None, None  # Không nên xảy ra với 8-puzzle

        dr, dc, action_name = random.choice(possible_moves_tuples)

        new_state_list = [row[:] for row in state_list]
        # Vị trí của ô số sẽ tráo đổi với ô trống
        tile_to_move_r, tile_to_move_c = blank_r + dr, blank_c + dc

        # Thực hiện tráo đổi
        new_state_list[blank_r][blank_c] = new_state_list[tile_to_move_r][tile_to_move_c]
        new_state_list[tile_to_move_r][tile_to_move_c] = 0  # Ô trống di chuyển đến vị trí của ô số

        return action_name, new_state_list

    def solve(self, initial_state_list, goal_state_list,
              initial_temp=100.0, cooling_rate=0.995, min_temp=0.01, max_iterations_sa=30000):
        """
        Thực hiện Simulated Annealing.
        Lưu ý: SA không đảm bảo tìm được lời giải tối ưu hoặc thậm chí là lời giải.
        Nó trả về trạng thái tốt nhất mà nó tìm thấy trong quá trình.
        """
        start_time = time.time()
        self.iterations_count = 0
        self.goal_state_list_ref = goal_state_list  # Tham chiếu để hàm heuristic sử dụng
        self._precompute_goal_positions(self.goal_state_list_ref)

        current_state = [row[:] for row in initial_state_list]
        current_h = self._calculate_manhattan_distance(current_state)  # Chi phí/Năng lượng của trạng thái hiện tại

        # Theo dõi trạng thái tốt nhất từng gặp
        best_state_overall = [row[:] for row in current_state]
        best_h_overall = current_h

        path_actions_taken = []  # Lưu các hành động đã thực hiện để đến current_state
        temp = initial_temp

        iteration = 0
        while temp > min_temp and iteration < max_iterations_sa:
            iteration += 1
            self.iterations_count = iteration

            if current_h == 0:  # Trạng thái hiện tại là đích
                best_state_overall = [row[:] for row in current_state]  # Cập nhật best nếu current là goal
                best_h_overall = 0
                break  # Thoát vòng lặp chính

            action_to_neighbor, neighbor_state = self._get_random_neighbor(current_state)

            if neighbor_state is None:  # Không có láng giềng (không nên xảy ra)
                break

            neighbor_h = self._calculate_manhattan_distance(neighbor_state)

            delta_e = neighbor_h - current_h  # Thay đổi năng lượng/chi phí

            if delta_e < 0:  # Nếu láng giềng tốt hơn, luôn chấp nhận
                current_state = neighbor_state
                current_h = neighbor_h
                path_actions_taken.append(action_to_neighbor)
                if current_h < best_h_overall:  # Cập nhật trạng thái tốt nhất toàn cục
                    best_state_overall = [row[:] for row in current_state]
                    best_h_overall = current_h
            else:  # Nếu láng giềng tệ hơn hoặc bằng
                # Chấp nhận với một xác suất P = exp(-delta_E / T)
                if temp > 0:  # Tránh chia cho 0 nếu nhiệt độ quá thấp
                    acceptance_probability = math.exp(-delta_e / temp)
                    if random.random() < acceptance_probability:
                        current_state = neighbor_state
                        current_h = neighbor_h
                        path_actions_taken.append(action_to_neighbor)
                        # Không cần cập nhật best_state_overall ở đây vì đây là nước đi tệ hơn
                # Nếu không chấp nhận, current_state không đổi

            temp *= cooling_rate  # Giảm nhiệt độ

        time_taken = time.time() - start_time

        # Sau khi vòng lặp kết thúc, best_h_overall là heuristic của trạng thái tốt nhất tìm được
        success = (best_h_overall == 0)  # Thành công nếu trạng thái tốt nhất là đích
        reason = ""
        if success:
            reason = "Goal reached (by best state found)."
        elif iteration >= max_iterations_sa:
            reason = "Max iterations reached."
        else:  # temp <= min_temp
            reason = "Temperature cooled down."

        if best_h_overall > 0 and success is False and reason != "Max iterations reached.":
            reason += " Did not find goal."

        return {
            "path_actions": path_actions_taken,  # Đường đi của current_state
            "steps": len(path_actions_taken),  # Số bước di chuyển của current_state
            "g_cost": len(path_actions_taken),
            "h_cost": best_h_overall,  # Heuristic của trạng thái TỐT NHẤT tìm được
            "final_state_list": best_state_overall,  # Trạng thái TỐT NHẤT tìm được
            "time_taken": time_taken,
            "nodes_expanded": self.iterations_count,  # Số vòng lặp chính
            "success": success,
            "reason": reason
        }


if __name__ == '__main__':
    solver = SimulatedAnnealingSolver()
    initial = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]
    # initial = [[8,6,7],[2,5,4],[3,0,1]] # Trạng thái khó, cần nhiều iterations và tinh chỉnh tham số
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    print(f"Solving puzzle with Simulated Annealing (Manhattan) from {initial} to {goal}")

    # Tham số SA có thể cần được tinh chỉnh kỹ lưỡng
    # result = solver.solve(initial, goal, initial_temp=100.0, cooling_rate=0.99, min_temp=0.001, max_iterations_sa=50000)
    result = solver.solve(initial, goal)  # Sử dụng tham số mặc định

    print(f"Success: {result['success']}")
    print(f"Reason: {result['reason']}")
    print(f"Current_state path steps: {result['steps']}, Best_state_overall h: {result['h_cost']}")
    # print(f"Path (actions of current_state): {result['path_actions']}") # Có thể rất dài
    print(f"Best state found: {result['final_state_list']}")
    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Iterations: {result['nodes_expanded']}")