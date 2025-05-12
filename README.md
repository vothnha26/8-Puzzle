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
   ![image](https://github.com/user-attachments/assets/bc0a5c93-cc5e-43f4-bd8e-1415e4f091d3)

   
