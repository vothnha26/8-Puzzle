# 8-Puzzle

2. Nội dung

  Các thành phần chính của bài toán tìm kiếm:

   Không gian trạng thái: ma trận 3x3 gồm các số 1 đến 8 được sắp xếp ngẫu nhiên và ô trống

   Trạng thái ban đầu: Là trạng thái xuất phát của bài toán

   Hành động: bao gồm di chuyển ô trống lên, xuống, sang trái hoặc sang phải

   Trạng thái đích: Là trạng thái mà người chơi muốn giải ra (ví dụ: các số được sắp xếp theo thứ tự từ 1 đến 8, ô trống ở cuối cùng)

   Chi phí đường đi: Mỗi hành động di chuyển được tính là 1 chi phí

  Solution (giải pháp):

    Giải pháp ở đây có thể từ trạng thái ban đầu đến một trạng thái đích (tùy theo các nhóm thuật toán sẽ cho ra các trạng thái đích/ lời giải khác nhau). Một       
    solution được coi là tối ưu (optimal) nếu nó có chi phí đường đi thấp nhất trong số tất cả các solution có thể. 

2.1. Các thuật toán Tìm kiếm không có thông tin:

   ![image](https://github.com/user-attachments/assets/213bafae-0edb-4769-86d8-4487cef898dc)


  Dựa vào hình ảnh có thể nhận xét như sau:

  Thuật toán BFS: đảm bảo tìm ra giải pháp tối ưu (ngắn nhất) nhưng chi phí về không gian bộ nhớ sẽ lớn (do mỗi lần sẽ duyệt theo độ sâu)

  Thuật toán DFS: không đảm bảo tìm ra giải pháp tối ưu (ngắn nhất) nhưng sẽ đảm bảo về không gian bộ nhớ (duyệt theo độ sâu)

  Thuật toán UCS: đảm bảo tìm ra giải pháp tối ưu (sử dụng bfs), chi phí về không gian bộ nhớ sẽ lớn (do mỗi lần sẽ duyệt theo độ sâu) nhưng có thể tìm thấy lời giải nhanh hơn so với bfs

  Thuật toán IDDFS: đảm bảo tìm ra giải pháp tối ưu (ngắn nhất) và tối ưu về không gian bộ nhớ (giống DFS), nhưng có thể chậm hơn BFS/UCS một chút do phải duyệt lại các nút ở độ sâu nông hơn.

  Nhìn chung, đối với bài toán 8-puzzle, khi yêu cầu giải pháp tối ưu và đối mặt với giới hạn bộ nhớ, IDDFS thường là lựa chọn cân bằng và hiệu quả nhất trong nhóm thuật toán không thông tin này, mặc dù BFS/UCS có thể nhanh hơn nếu bộ nhớ không phải là vấn đề. DFS chỉ phù hợp khi không yêu cầu tính tối ưu.
   
