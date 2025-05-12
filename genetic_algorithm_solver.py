# genetic_algorithm_solver.py
import time
import random
import math


class Individual:
    def __init__(self, moves):
        self.moves = moves  # List of actions, e.g., ["UP", "LEFT", ...]
        self.fitness = 0.0

    def __repr__(self):
        # Giới hạn số bước hiển thị để tránh làm rối console
        display_moves = self.moves[:10] + (['...'] if len(self.moves) > 10 else [])
        return f"Moves: {display_moves}, Fitness: {self.fitness:.4f}"


class GeneticAlgorithmSolver:
    def __init__(self):
        self._goal_positions_cache = {}
        self.goal_state_list_ref = None
        self.generations_count = 0

        # Tham số GA mặc định (có thể được ghi đè khi gọi solve)
        self.population_size = 60
        self.max_initial_chromosome_length = 25  # Độ dài tối đa của chuỗi hành động ban đầu
        self.num_generations = 150
        self.crossover_rate = 0.85
        self.mutation_rate = 0.05  # Tỷ lệ đột biến cho mỗi gen (hành động) trong một cá thể
        self.gene_mutation_rate = 0.1  # Tỷ lệ đột biến thực sự cho một gene nếu cá thể được chọn để đột biến
        self.elitism_count = 2  # Số cá thể tốt nhất được giữ lại cho thế hệ sau

        self.possible_actions = ["UP", "DOWN", "LEFT", "RIGHT"]

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

    def _apply_move(self, state_list, action):
        blank_r, blank_c = self._find_blank_position(state_list)
        dr, dc = 0, 0
        if action == "UP":
            dr = -1
        elif action == "DOWN":
            dr = 1
        elif action == "LEFT":
            dc = -1
        elif action == "RIGHT":
            dc = 1

        tile_to_move_r, tile_to_move_c = blank_r + dr, blank_c + dc
        new_state_list = [row[:] for row in state_list]
        if 0 <= tile_to_move_r < 3 and 0 <= tile_to_move_c < 3:
            new_state_list[blank_r][blank_c] = new_state_list[tile_to_move_r][tile_to_move_c]
            new_state_list[tile_to_move_r][tile_to_move_c] = 0
            return new_state_list
        return state_list

    def _apply_sequence_of_moves(self, initial_state_list, moves_sequence):
        current_state = [row[:] for row in initial_state_list]
        for move in moves_sequence:
            current_state = self._apply_move(current_state, move)
        return current_state

    def _calculate_fitness(self, individual, initial_state_list):
        final_state = self._apply_sequence_of_moves(initial_state_list, individual.moves)
        h_cost = self._calculate_manhattan_distance(final_state)

        fitness_score = 0
        if h_cost == 0:  # Đã đến đích
            # Thưởng rất lớn, trừ đi một chút cho độ dài đường đi
            fitness_score = 10000.0 - len(individual.moves) * 0.1
        else:
            # Nghịch đảo của Manhattan distance, cộng 1 để tránh chia cho 0
            fitness_score = 1.0 / (1.0 + h_cost)

        # Phạt nhẹ cho các chuỗi quá dài, ngay cả khi chưa đến đích
        # Điều này giúp hướng GA đến các giải pháp ngắn hơn nếu có thể
        fitness_score -= len(individual.moves) * 0.0001
        return max(0, fitness_score)  # Đảm bảo fitness không âm

    def _initialize_population(self, initial_state_list):
        population = []
        for _ in range(self.population_size):
            # Độ dài ngẫu nhiên cho chromosome ban đầu, trong một khoảng hợp lý
            chromosome_length = random.randint(max(1, self.max_initial_chromosome_length // 2),
                                               self.max_initial_chromosome_length)
            moves = [random.choice(self.possible_actions) for _ in range(chromosome_length)]
            individual = Individual(moves)
            # Fitness được tính sau khi quần thể được khởi tạo hoàn chỉnh
            population.append(individual)
        return population

    def _selection(self, population):  # Tournament selection
        tournament_size = 5
        if self.population_size < tournament_size: tournament_size = self.population_size

        selected_parents = []
        for _ in range(self.population_size):  # Chọn đủ cha mẹ để tạo thế hệ mới
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected_parents.append(winner)
        return selected_parents

    def _crossover(self, parent1, parent2):
        child1_moves, child2_moves = list(parent1.moves), list(parent2.moves)

        if random.random() < self.crossover_rate:
            len1, len2 = len(parent1.moves), len(parent2.moves)
            if len1 > 0 and len2 > 0:  # Cả hai cha mẹ phải có gen
                # Single-point crossover
                point1 = random.randint(0, len1)  # Điểm cắt có thể ở đầu hoặc cuối
                point2 = random.randint(0, len2)

                c1m = parent1.moves[:point1] + parent2.moves[point2:]
                c2m = parent2.moves[:point2] + parent1.moves[point1:]

                child1_moves = c1m if c1m else [random.choice(self.possible_actions)]
                child2_moves = c2m if c2m else [random.choice(self.possible_actions)]

        return Individual(child1_moves), Individual(child2_moves)

    def _mutate(self, individual):
        if random.random() < self.mutation_rate:  # Quyết định xem cá thể này có đột biến không
            mutated_moves = list(individual.moves)
            if not mutated_moves: mutated_moves = [random.choice(self.possible_actions)]  # Đảm bảo có gen để đột biến

            for i in range(len(mutated_moves)):
                if random.random() < self.gene_mutation_rate:  # Đột biến gen này
                    mutated_moves[i] = random.choice(self.possible_actions)

            # Đột biến thay đổi độ dài (tùy chọn, có thể thêm)
            # if random.random() < 0.1: # Tỷ lệ nhỏ để thay đổi độ dài
            #     if len(mutated_moves) > 1 and random.random() < 0.5:
            #         mutated_moves.pop(random.randrange(len(mutated_moves)))
            #     elif len(mutated_moves) < self.max_initial_chromosome_length * 2: # Giới hạn độ dài tối đa
            #         mutated_moves.insert(random.randrange(len(mutated_moves) + 1), random.choice(self.possible_actions))

            individual.moves = mutated_moves if mutated_moves else [random.choice(self.possible_actions)]

    def solve(self, initial_state_list, goal_state_list, **kwargs):
        # Ghi đè tham số nếu được cung cấp
        self.population_size = kwargs.get('pop_size', self.population_size)
        self.num_generations = kwargs.get('generations', self.num_generations)
        self.mutation_rate = kwargs.get('mut_rate', self.mutation_rate)  # Tỷ lệ đột biến cá thể
        self.gene_mutation_rate = kwargs.get('gene_mut_rate', self.gene_mutation_rate)  # Tỷ lệ đột biến gen
        self.crossover_rate = kwargs.get('cross_rate', self.crossover_rate)
        self.elitism_count = kwargs.get('elitism_k', self.elitism_count)
        self.max_initial_chromosome_length = kwargs.get('max_len', self.max_initial_chromosome_length)

        start_time = time.time()
        self.goal_state_list_ref = goal_state_list
        self._precompute_goal_positions(self.goal_state_list_ref)

        population = self._initialize_population(initial_state_list)
        # Tính fitness cho quần thể ban đầu
        for ind in population:
            ind.fitness = self._calculate_fitness(ind, initial_state_list)

        best_overall_individual = max(population, key=lambda ind: ind.fitness)

        for gen in range(self.num_generations):
            self.generations_count = gen + 1

            parents = self._selection(population)
            next_population = []

            # Elitism
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            for i in range(min(self.elitism_count, len(population))):  # Đảm bảo không vượt quá kích thước quần thể
                next_population.append(population[i])

            # Tạo con cái
            while len(next_population) < self.population_size:
                p1, p2 = random.sample(parents, 2)  # Chọn 2 cha mẹ ngẫu nhiên từ danh sách đã chọn lọc
                child1, child2 = self._crossover(p1, p2)

                self._mutate(child1)
                self._mutate(child2)

                child1.fitness = self._calculate_fitness(child1, initial_state_list)
                child2.fitness = self._calculate_fitness(child2, initial_state_list)

                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)

            population = next_population[:self.population_size]

            current_gen_best = max(population, key=lambda ind: ind.fitness)
            if current_gen_best.fitness > best_overall_individual.fitness:
                # Sao chép sâu để best_overall_individual không bị thay đổi nếu current_gen_best thay đổi
                best_overall_individual = Individual(list(current_gen_best.moves))
                best_overall_individual.fitness = current_gen_best.fitness

            final_state_of_best = self._apply_sequence_of_moves(initial_state_list, best_overall_individual.moves)
            if self._calculate_manhattan_distance(final_state_of_best) == 0:
                # print(f"GA: Goal reached by best individual at generation {gen+1}!") # Debug
                break

        time_taken = time.time() - start_time
        final_state_of_best = self._apply_sequence_of_moves(initial_state_list, best_overall_individual.moves)
        final_h_cost = self._calculate_manhattan_distance(final_state_of_best)
        success = (final_h_cost == 0)

        reason = "Goal reached by best individual." if success else "Max generations reached."
        if not success and self.generations_count < self.num_generations:
            reason = "Stopped early (unknown reason)."

        return {
            "path_actions": best_overall_individual.moves,
            "steps": len(best_overall_individual.moves),
            "g_cost": len(best_overall_individual.moves),
            "h_cost": final_h_cost,
            "final_state_list": final_state_of_best,
            "time_taken": time_taken,
            "nodes_expanded": self.generations_count,  # Số thế hệ đã chạy
            "success": success,
            "reason": reason,
            "best_fitness": best_overall_individual.fitness
        }


if __name__ == '__main__':
    solver = GeneticAlgorithmSolver()
    initial = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    print(f"Solving puzzle with Genetic Algorithm from {initial} to {goal}")

    # Các tham số GA có thể cần tinh chỉnh rất nhiều để có kết quả tốt
    result = solver.solve(initial, goal,
                          pop_size=100,
                          generations=200,
                          mut_rate=0.1,  # Tỷ lệ cá thể bị đột biến
                          gene_mut_rate=0.05,  # Tỷ lệ gen trong cá thể đó bị đột biến
                          cross_rate=0.8,
                          elitism_k=3,
                          max_len=30)  # Max_len cho chromosome ban đầu

    print(f"Success: {result['success']}")
    print(f"Reason: {result['reason']}")
    print(f"Best Fitness found: {result.get('best_fitness', -1):.4f}")
    print(f"Path length (g): {result['g_cost']}, Final state h: {result['h_cost']}")
    # print(f"Path (actions): {result['path_actions']}")
    print(f"Final state reached by best: {result['final_state_list']}")
    print(f"Time taken: {result['time_taken']:.4f} seconds")
    print(f"Generations run: {result['nodes_expanded']}")