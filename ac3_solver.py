# ac3_solver.py
import time
from collections import deque


class AC3Solver:
    def __init__(self):
        self.revisions_count = 0
        self.arcs_processed_count = 0
        self.log = []  # Để lưu các bước thực hiện của thuật toán

    def _revise(self, xi_name, xj_name, domains, constraints_xi_xj):
        """
        Làm cho cung (xi_name, xj_name) trở nên nhất quán.
        constraints_xi_xj: list các hàm ràng buộc func(val_xi, val_xj) giữa Xi và Xj.
        Trả về True nếu miền của Xi bị thay đổi, False nếu không.
        """
        revised = False
        # Phải duyệt trên một bản sao của domains[xi_name] nếu ta định xóa phần tử khỏi nó
        for x_val in list(domains[xi_name]):
            found_satisfying_y = False
            for y_val in domains[xj_name]:
                # Kiểm tra xem (x_val, y_val) có thỏa mãn TẤT CẢ các ràng buộc giữa Xi và Xj không
                all_constraints_met_for_this_y = True
                for constraint_func in constraints_xi_xj:
                    if not constraint_func(x_val, y_val):
                        all_constraints_met_for_this_y = False
                        break

                if all_constraints_met_for_this_y:
                    found_satisfying_y = True
                    break  # Tìm thấy một y_val phù hợp cho x_val này

            if not found_satisfying_y:
                self.log.append(
                    f"REVISE({xi_name},{xj_name}): Loại bỏ {x_val} khỏi D({xi_name}) vì không có giá trị trong D({xj_name}) thỏa mãn.")
                domains[xi_name].remove(x_val)
                revised = True
                self.revisions_count += 1
        return revised

    def solve(self, variables, initial_domains, constraints_list):
        """
        Thực hiện thuật toán AC-3.
        variables: list tên các biến (ví dụ: ["X", "Y", "Z"])
        initial_domains: dict {tên_biến: [list_giá_trị_miền]}
        constraints_list: list các tuple, mỗi tuple là (tên_biến_1, tên_biến_2, hàm_ràng_buộc)
                          hàm_ràng_buộc(giá_trị_1, giá_trị_2) trả về True nếu ràng buộc được thỏa mãn.
        """
        start_time = time.time()
        self.revisions_count = 0
        self.arcs_processed_count = 0
        self.log = []

        # Sao chép miền giá trị để không thay đổi dict gốc
        domains = {var: list(domain_vals) for var, domain_vals in initial_domains.items()}

        self.log.append("AC-3 Algorithm Started.")
        self.log.append("Initial Domains:")
        for var, domain_vals in domains.items():
            self.log.append(f"  D({var}) = {domain_vals}")

        queue = deque()  # Hàng đợi các cung (Xi, Xj)

        # Khởi tạo hàng đợi với tất cả các cung có thể có từ ràng buộc
        # Một ràng buộc C(Var1, Var2) ngụ ý có các cung (Var1, Var2) và (Var2, Var1) cần xem xét
        # (trừ khi ràng buộc chỉ có một chiều một cách rõ ràng).
        # Để đơn giản, ta sẽ thêm cả hai chiều cho mỗi ràng buộc được định nghĩa.
        for var1, var2, _ in constraints_list:
            if (var1, var2) not in queue:
                queue.append((var1, var2))
            if (var2, var1) not in queue:  # Giả sử ràng buộc có thể cần kiểm tra cả hai chiều
                queue.append((var2, var1))

        self.log.append(f"Initial queue size: {len(queue)}")
        self.log.append("--- Processing Arcs ---")

        while queue:
            xi_name, xj_name = queue.popleft()
            self.arcs_processed_count += 1
            self.log.append(f"Processing arc: ({xi_name}, {xj_name})")

            # Lấy các hàm ràng buộc cụ thể cho cung (xi_name, xj_name)
            # Hàm ràng buộc trong constraints_list được định nghĩa cho (var1, var2)
            # Khi xét cung (Xi, Xj), hàm phải là func(val_xi, val_xj)
            relevant_constraints_for_arc = []
            for c_var1, c_var2, c_func in constraints_list:
                if c_var1 == xi_name and c_var2 == xj_name:
                    relevant_constraints_for_arc.append(c_func)
                # Nếu ràng buộc là đối xứng và hàm func xử lý được cả hai chiều,
                # hoặc nếu chúng ta định nghĩa ràng buộc một chiều, chỉ cần điều kiện trên.
                # Để chặt chẽ, nếu ràng buộc C(X,Y) là X<Y, thì khi xét cung (Y,X), ràng buộc là Y>X.
                # Ta giả định rằng constraints_list chứa các định nghĩa một chiều nếu cần.
                # Ví dụ: nếu có X<Y, người dùng cũng phải cung cấp Y>X nếu muốn kiểm tra cung (Y,X) với ràng buộc đó.
                # Cách đơn giản hơn là hàm revise tự xử lý.
                # Cách hiện tại: chỉ lấy ràng buộc đúng chiều.

            if self._revise(xi_name, xj_name, domains, relevant_constraints_for_arc):
                if not domains[xi_name]:  # Miền của Xi rỗng
                    self.log.append(f"Domain of {xi_name} became empty. CSP has no solution.")
                    time_taken = time.time() - start_time
                    return {"consistent": False, "domains": domains, "log": self.log,
                            "revisions": self.revisions_count, "arcs_processed": self.arcs_processed_count,
                            "time_taken": time_taken}

                # Thêm lại các cung (Xk, Xi) vào hàng đợi, với Xk là láng giềng của Xi (Xk != Xj)
                for c_var1_neighbor, c_var2_neighbor, _ in constraints_list:
                    # Tìm Xk sao cho có ràng buộc C(Xk, Xi) hoặc C(Xi, Xk)
                    # và Xk != Xj
                    neighbor_to_add = None
                    if c_var1_neighbor == xi_name and c_var2_neighbor != xj_name:  # C(Xi, Xk), Xk = c_var2_neighbor
                        neighbor_to_add = c_var2_neighbor
                    elif c_var2_neighbor == xi_name and c_var1_neighbor != xj_name:  # C(Xk, Xi), Xk = c_var1_neighbor
                        neighbor_to_add = c_var1_neighbor

                    if neighbor_to_add and (neighbor_to_add, xi_name) not in queue:
                        self.log.append(
                            f"  Domain of {xi_name} changed, adding arc ({neighbor_to_add}, {xi_name}) to queue.")
                        queue.append((neighbor_to_add, xi_name))

            # self.log.append("Current Domains:") # Log này có thể quá dài
            # for var_name_log, domain_vals_log in domains.items():
            #      self.log.append(f"  D({var_name_log}) = {domain_vals_log}")

        self.log.append("--- AC-3 Finished: Queue is empty ---")
        time_taken = time.time() - start_time
        return {"consistent": True, "domains": domains, "log": self.log,
                "revisions": self.revisions_count, "arcs_processed": self.arcs_processed_count,
                "time_taken": time_taken}


if __name__ == '__main__':
    solver = AC3Solver()

    print("--- Example CSP for AC-3 ---")
    variables = ["X", "Y", "Z"]
    domains = {
        "X": [1, 2, 3],
        "Y": [1, 2, 3],
        "Z": [1, 2, 3]
    }


    # Định nghĩa các hàm ràng buộc
    def x_neq_y(val_x, val_y):
        return val_x != val_y


    def y_neq_z(val_y, val_z):
        return val_y != val_z


    def x_lt_z(val_x, val_z):
        return val_x < val_z  # X < Z


    # Danh sách các ràng buộc: (Biến 1, Biến 2, Hàm kiểm tra cho Biến 1 và Biến 2)
    # AC-3 sẽ cần kiểm tra các cung theo cả hai chiều nếu ràng buộc là hai chiều.
    # Ví dụ, X!=Y ngụ ý Y!=X. X<Z ngụ ý Z>X.
    # Để đơn giản cho hàm _revise, ta nên định nghĩa các ràng buộc một cách rõ ràng cho từng cung.
    # Hoặc, hàm _revise phải "thông minh" hơn.
    # Hiện tại, _revise(Xi, Xj, constraints) tìm các constraint(Xi, Xj)

    # Để chạy đúng, queue phải được khởi tạo với các cung (Xi,Xj) và (Xj,Xi) nếu ràng buộc là đối xứng.
    # Và _revise phải lấy đúng hàm cho cung (Xi,Xj) cụ thể.
    # Cách đơn giản nhất là định nghĩa các constraint function cho mỗi chiều nếu không đối xứng.

    # Ràng buộc: X!=Y, Y!=Z, X < Z
    constraints = [
        ("X", "Y", x_neq_y),
        ("Y", "X", x_neq_y),  # Đối xứng
        ("Y", "Z", y_neq_z),
        ("Z", "Y", y_neq_z),  # Đối xứng
        ("X", "Z", x_lt_z),  # X < Z
        ("Z", "X", lambda val_z, val_x: val_z > val_x)  # Z > X (tương đương X < Z)
    ]

    result = solver.solve(variables, domains, constraints)

    print(f"Consistent: {result['consistent']}")
    print("Final Domains:")
    for var, domain_vals in result['domains'].items():
        print(f"  D({var}) = {domain_vals}")
    print(f"Revisions: {result['revisions']}, Arcs Processed: {result['arcs_processed']}")
    print(f"Time: {result['time_taken']:.4f}s")
    # print("\nLog:")
    # for entry in result['log']:
    #     print(entry)