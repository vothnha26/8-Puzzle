# q_learning_solver.py
import random
import time
import pickle  # Để lưu/tải bảng Q

class QLearningSolver:
    def __init__(self, learning_rate=0.1, discount_factor=0.9,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_rate=0.9995, # Giảm epsilon chậm hơn một chút
                 q_table_file='q_table_8puzzle.pkl'):

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay_rate

        self.q_table = {}  # Dùng dictionary: state_tuple -> {action_str: q_value}
        self.q_table_file = q_table_file

        self.actions_possible = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.training_episodes_completed = 0
        self.total_training_time = 0.0
        self.is_trained = False

    def _state_to_tuple(self, state_list_2d):
        return tuple(map(tuple, state_list_2d))

    def _get_q_action_values(self, state_tuple):
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {action: 0.0 for action in self.actions_possible}
        return self.q_table[state_tuple]

    def _find_blank(self, state_list_2d):
        for r, row in enumerate(state_list_2d):
            for c, tile in enumerate(row):
                if tile == 0:
                    return r, c
        return None

    def _get_valid_actions(self, state_list_2d):
        br, bc = self._find_blank(state_list_2d)
        if br is None: return []
        valid_actions = []
        # Giả sử grid_size là 3x3
        grid_size = len(state_list_2d)
        if br > 0: valid_actions.append("UP")
        if br < grid_size - 1: valid_actions.append("DOWN")
        if bc > 0: valid_actions.append("LEFT")
        if bc < grid_size - 1: valid_actions.append("RIGHT")
        return valid_actions

    def _apply_action(self, state_list_2d, action_str):
        br, bc = self._find_blank(state_list_2d)
        if br is None: return [r[:] for r in state_list_2d], False

        dr, dc = 0, 0
        if action_str == "UP": dr = -1
        elif action_str == "DOWN": dr = 1
        elif action_str == "LEFT": dc = -1
        elif action_str == "RIGHT": dc = 1
        else: return [r[:] for r in state_list_2d], False

        tile_to_swap_r, tile_to_swap_c = br + dr, bc + dc
        new_state = [r[:] for r in state_list_2d]
        grid_size = len(state_list_2d)

        if 0 <= tile_to_swap_r < grid_size and 0 <= tile_to_swap_c < grid_size:
            new_state[br][bc] = new_state[tile_to_swap_r][tile_to_swap_c]
            new_state[tile_to_swap_r][tile_to_swap_c] = 0
            return new_state, True
        return new_state, False

    def _generate_random_solvable_state_for_training(self, goal_state_list_2d, num_shuffles=100):
        """
        Tạo một trạng thái ngẫu nhiên có thể giải được bằng cách
        bắt đầu từ trạng thái đích và thực hiện các nước đi ngẫu nhiên.
        """
        current_state = [r[:] for r in goal_state_list_2d]
        last_action = None
        for _ in range(num_shuffles):
            valid_actions = self._get_valid_actions(current_state)
            # Tránh thực hiện hành động ngược lại với hành động vừa rồi ngay lập tức (nếu có thể)
            # để việc xáo trộn hiệu quả hơn
            if last_action and len(valid_actions) > 1:
                inverse_last_action = None
                if last_action == "UP": inverse_last_action = "DOWN"
                elif last_action == "DOWN": inverse_last_action = "UP"
                elif last_action == "LEFT": inverse_last_action = "RIGHT"
                elif last_action == "RIGHT": inverse_last_action = "LEFT"
                if inverse_last_action in valid_actions:
                    valid_actions.remove(inverse_last_action)

            if not valid_actions: break # Bị kẹt, không nên xảy ra nếu goal_state hợp lệ

            action_to_shuffle = random.choice(valid_actions)
            next_state, moved = self._apply_action(current_state, action_to_shuffle)
            if moved:
                current_state = next_state
                last_action = action_to_shuffle
            # Nếu không di chuyển được (dù hiếm khi xảy ra với valid_actions), thử lại
        return current_state

    def choose_action(self, state_tuple, current_epsilon_val, state_list_for_valid_actions):
        valid_actions = self._get_valid_actions(state_list_for_valid_actions)
        if not valid_actions: return None

        if random.random() < current_epsilon_val:
            return random.choice(valid_actions)
        else:
            q_values_for_state = self._get_q_action_values(state_tuple)
            valid_q_values = {act: q_values_for_state.get(act, -float('inf')) for act in valid_actions} # Dùng -inf cho hành động chưa có trong q_table[state]
            if not valid_q_values : return random.choice(valid_actions) # Dự phòng

            max_q = -float('inf')
            best_actions = []
            for act, q_val in valid_q_values.items():
                if q_val > max_q:
                    max_q = q_val
                    best_actions = [act]
                elif q_val == max_q:
                    best_actions.append(act)
            return random.choice(best_actions)

    def train(self, initial_state_to_solve_list_2d, goal_state_list_2d,
              num_episodes=20000, max_steps_per_episode=200,
              start_random_episode_ratio=0.8, num_shuffles_for_random_start=100):
        """
        Huấn luyện bảng Q.
        Args:
            initial_state_to_solve_list_2d: Trạng thái ban đầu của bài toán CỤ THỂ mà người dùng muốn giải sau này.
                                            Cũng được dùng để bắt đầu một số episodes.
            goal_state_list_2d: Trạng thái đích chung cho việc huấn luyện.
            start_random_episode_ratio: Tỷ lệ episodes bắt đầu từ trạng thái ngẫu nhiên.
            num_shuffles_for_random_start: Số lần xáo trộn từ trạng thái đích để tạo trạng thái bắt đầu ngẫu nhiên.
        """
        print(f"Q-Learning: Bắt đầu huấn luyện cho {num_episodes} episodes...")
        start_train_time = time.time()
        self.training_episodes_completed = 0
        self.epsilon = self.epsilon_start

        goal_state_tuple = self._state_to_tuple(goal_state_list_2d)
        rewards_per_episode = [] # Để theo dõi

        for episode in range(num_episodes):
            self.training_episodes_completed += 1
            total_episode_reward = 0

            # Quyết định trạng thái bắt đầu cho episode này
            if random.random() < start_random_episode_ratio:
                current_state_list = self._generate_random_solvable_state_for_training(goal_state_list_2d, num_shuffles_for_random_start)
            else:
                current_state_list = [r[:] for r in initial_state_to_solve_list_2d]

            current_state_tuple = self._state_to_tuple(current_state_list)

            for step in range(max_steps_per_episode):
                action = self.choose_action(current_state_tuple, self.epsilon, current_state_list)
                if action is None: break

                next_state_list, moved_successfully = self._apply_action(current_state_list, action)
                next_state_tuple = self._state_to_tuple(next_state_list)

                reward = -1  # Chi phí cho mỗi bước
                is_done = (next_state_list == goal_state_list_2d)

                if is_done:
                    reward = 100
                elif not moved_successfully:
                    reward = -20 # Phạt nặng hơn cho việc đâm vào tường

                total_episode_reward += reward

                # Cập nhật Q-value
                current_q_values_dict = self._get_q_action_values(current_state_tuple)
                old_q_value = current_q_values_dict.get(action, 0.0)

                next_max_q = 0.0
                if not is_done:
                    next_q_values_dict = self._get_q_action_values(next_state_tuple)
                    valid_next_actions = self._get_valid_actions(next_state_list)
                    if valid_next_actions:
                        next_max_q = max(next_q_values_dict.get(act, 0.0) for act in valid_next_actions)
                    # else: next_max_q = 0.0 (nếu không có hành động hợp lệ từ trạng thái tiếp theo, coi như không có giá trị tương lai)

                new_q_value = old_q_value + self.lr * (reward + self.gamma * next_max_q - old_q_value)
                self.q_table[current_state_tuple][action] = new_q_value

                current_state_list = next_state_list
                current_state_tuple = next_state_tuple

                if is_done:
                    break

            rewards_per_episode.append(total_episode_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (episode + 1) % (num_episodes // 20 if num_episodes >= 20 else 1) == 0:
                avg_reward = sum(rewards_per_episode[-(num_episodes // 20 if num_episodes >=20 else 1):]) / (num_episodes // 20 if num_episodes >=20 else 1)
                print(f"  Q-Train: Episode {episode + 1}/{num_episodes}. Epsilon: {self.epsilon:.4f}. Avg Reward (last batch): {avg_reward:.2f}")

        self.total_training_time = time.time() - start_train_time
        self.is_trained = True
        print(f"Q-Learning: Huấn luyện hoàn thành trong {self.total_training_time:.2f}s. Kích thước bảng Q: {len(self.q_table)}")


    def get_optimal_path(self, start_state_list, goal_state_list, max_path_len=50):
        if not self.is_trained:
            print("Q-Learning: Bảng Q chưa được huấn luyện!")
            print("Vui lòng chạy hàm train() trước hoặc tải bảng Q đã huấn luyện.")
            # Để ví dụ này chạy được ngay cả khi chưa train lâu, có thể train nhanh:
            # print("Chạy một phiên huấn luyện nhanh mặc định...")
            # self.train(start_state_list, goal_state_list, num_episodes=1000, max_steps_per_episode=100)
            # if not self.is_trained: # Vẫn chưa train được thì chịu
            return {"success": False, "reason": "Bảng Q chưa được huấn luyện.", "path_actions": [], "steps":0, "g_cost":float('inf')}


        path_actions = []
        current_state = [r[:] for r in start_state_list]
        visited_states_in_path = set() # Để tránh lặp vô hạn nếu bảng Q chưa tốt

        for step_count in range(max_path_len):
            current_state_tuple = self._state_to_tuple(current_state)
            visited_states_in_path.add(current_state_tuple)

            if current_state == goal_state_list:
                break

            action = self.choose_action(current_state_tuple, 0.0, current_state) # Epsilon = 0 để khai thác

            if action is None:
                print("Q-Learning Path: Bị kẹt hoặc không có hành động hợp lệ từ chính sách Q.")
                break

            path_actions.append(action)
            current_state, moved = self._apply_action(current_state, action)

            if not moved:
                print(f"Q-Learning Path: Cảnh báo - Chính sách Q gợi ý nước đi không hợp lệ '{action}' từ {current_state_tuple}")
                path_actions.pop()
                break
            if self._state_to_tuple(current_state) in visited_states_in_path:
                print("Q-Learning Path: Phát hiện lặp trạng thái trong đường đi. Dừng lại.")
                # Có thể coi là thất bại hoặc trả về đường đi hiện tại
                path_actions.pop() # Xoá hành động gây lặp
                break


        success = (current_state == goal_state_list)
        g_cost = len(path_actions) if success else float('inf')

        return {
            "path_actions": path_actions if success else [],
            "steps": len(path_actions), # số bước thực sự đã đi
            "g_cost": g_cost,
            "h_cost": 0 if success else "N/A", # Q-learning không dùng h_cost trực tiếp khi giải
            "final_state_list": current_state,
            "nodes_expanded": self.training_episodes_completed, # Là số episodes đã huấn luyện
            "success": success,
            "reason": "Đã đến đích bằng chính sách Q." if success else "Không đến được đích hoặc vượt quá độ dài tối đa.",
            "time_taken": self.total_training_time # Trả về thời gian huấn luyện đã thực hiện
        }

    def save_q_table(self, filename=None):
        """Lưu bảng Q vào file."""
        file_to_save = filename if filename else self.q_table_file
        try:
            with open(file_to_save, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon, # Lưu cả epsilon cuối cùng
                    'episodes_completed': self.training_episodes_completed
                }, f)
            print(f"Bảng Q đã được lưu vào {file_to_save}")
        except Exception as e:
            print(f"Lỗi khi lưu bảng Q: {e}")

    def load_q_table(self, filename=None):
        """Tải bảng Q từ file."""
        file_to_load = filename if filename else self.q_table_file
        try:
            with open(file_to_load, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', self.epsilon_start) # Tải lại epsilon
                self.training_episodes_completed = data.get('episodes_completed', 0)
                self.is_trained = bool(self.q_table) # Nếu q_table có dữ liệu thì coi như đã train
            print(f"Bảng Q đã được tải từ {file_to_load}. Kích thước: {len(self.q_table)}. Epsilon hiện tại: {self.epsilon:.4f}")
        except FileNotFoundError:
            print(f"Không tìm thấy file {file_to_load}. Bảng Q sẽ bắt đầu trống.")
        except Exception as e:
            print(f"Lỗi khi tải bảng Q: {e}")

# --- Hàm trợ giúp để in trạng thái ---
def print_puzzle_state(state_list_2d, title="Trạng thái"):
    print(f"--- {title} ---")
    if not state_list_2d:
        print(" (Không có)")
        return
    for row in state_list_2d:
        print(" ".join(map(str, row)).replace('0', '.')) # Thay 0 bằng . cho dễ nhìn
    print("---------------")

if __name__ == '__main__':
    # Khởi tạo solver
    solver = QLearningSolver(
        learning_rate=0.1,        # Tỷ lệ học
        discount_factor=0.95,      # Yếu tố chiết khấu (quan tâm nhiều hơn đến tương lai)
        epsilon_start=1.0,
        epsilon_end=0.05,         # Giữ lại một chút khám phá
        epsilon_decay_rate=0.9999 # Giảm epsilon rất chậm để khám phá nhiều hơn
    )

    # Trạng thái ban đầu và đích cho bài toán 8-puzzle cụ thể
    # Đây là trạng thái ban đầu của BÀI TOÁN CỤ THỂ người dùng muốn giải
    # và cũng là một trong các điểm bắt đầu episode (nếu start_random_episode_ratio < 1)
    initial_problem_state = [[1, 2, 3], [0, 4, 6], [7, 5, 8]] # Một ví dụ
    # initial_problem_state = [[8, 1, 2], [0, 4, 3], [7, 6, 5]] # Ví dụ khó hơn

    # Trạng thái đích CHUNG cho việc huấn luyện
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    # Tùy chọn: Tải bảng Q đã huấn luyện trước (nếu có)
    # solver.load_q_table()

    # Huấn luyện (Cần nhiều episodes để có kết quả tốt cho 8-puzzle)
    # num_episodes_train = 50000 # Thử nghiệm với số lượng lớn hơn
    num_episodes_train = 30000 # Giảm để chạy thử nhanh hơn
    print(f"Chuẩn bị huấn luyện với {num_episodes_train} episodes.")
    solver.train(
        initial_state_to_solve_list_2d=initial_problem_state,
        goal_state_list_2d=goal_state,
        num_episodes=num_episodes_train,
        max_steps_per_episode=250, # Cho phép nhiều bước hơn mỗi episode
        start_random_episode_ratio=0.9, # 90% episodes bắt đầu từ trạng thái ngẫu nhiên
        num_shuffles_for_random_start=150 # Xáo trộn nhiều hơn để có trạng thái đa dạng
    )

    # Tùy chọn: Lưu bảng Q sau khi huấn luyện
    solver.save_q_table()

    # Trích xuất và hiển thị đường đi cho bài toán cụ thể
    print("\n--- Trích xuất đường đi bằng Bảng Q đã học ---")
    print_puzzle_state(initial_problem_state, "Trạng thái BAN ĐẦU của bài toán")
    print_puzzle_state(goal_state, "Trạng thái ĐÍCH của bài toán")

    result = solver.get_optimal_path(initial_problem_state, goal_state, max_path_len=60)

    print(f"\nThành công: {result['success']}")
    print(f"Lý do: {result['reason']}")
    if result['success']:
        print(f"Độ dài đường đi (g_cost): {result['g_cost']}")
        print(f"Các hành động: {result['path_actions']}")

        # Hiển thị các bước đi (tùy chọn)
        print("\n--- Các bước đi chi tiết ---")
        current_display_state = [r[:] for r in initial_problem_state]
        print_puzzle_state(current_display_state, f"Bước 0 (Ban đầu)")
        for i, action in enumerate(result['path_actions']):
            current_display_state, _ = solver._apply_action(current_display_state, action)
            print_puzzle_state(current_display_state, f"Bước {i+1}: Hành động '{action}'")
    else:
        print(f"Số bước đã thực hiện trước khi thất bại/dừng: {result['steps']}")

    print(f"\nThời gian huấn luyện đã báo cáo: {result['time_taken']:.2f}s")
    print(f"Số episodes đã huấn luyện: {result['nodes_expanded']}")
    print(f"Kích thước bảng Q cuối cùng: {len(solver.q_table)}")