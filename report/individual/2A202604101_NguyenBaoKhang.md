# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: *2A202604101*
- **Họ và tên: *Nguyễn Bảo Khang*
- **SEED** (in ở ô đầu notebook): 211

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                   precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.782      0.878        0.889   
2. BoW + MLP (deep learning)         0.946   0.871      0.907        0.919   
3a. Encoder (MiniLM) + kNN           0.894   0.921      0.907        1.000   
3b. Encoder (MiniLM) + LogReg        0.968   0.911      0.939        1.000   
3c. Encoder (MiniLM) + MLP head      0.958   0.901      0.929        0.974   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)           12.588       5249.010       0.048   
2. BoW + MLP (deep learning)        23.338       8507.327       0.431   
3a. Encoder (MiniLM) + kNN         111.310         17.005     105.153   
3b. Encoder (MiniLM) + LogReg      114.713         16.934      29.831   
3c. Encoder (MiniLM) + MLP head    112.362         32.966      34.395   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.158           0.0  
2. BoW + MLP (deep learning)          5.634           0.0  
3a. Encoder (MiniLM) + kNN          156.786           0.0  
3b. Encoder (MiniLM) + LogReg        45.526           0.0  
3c. Encoder (MiniLM) + MLP head      53.123           0.0 

## 3. Dự đoán và thực tế

- F1 của BoW + Logistic Regression. Em dự đoán F1 khoảng 0.75, recall cao hơn precision. Thực tế F1 là 0.878, precision 1.000 và recall 0.782, tức precision cao hơn recall. Em đã đánh giá thấp hiệu quả của mô hình và nghĩ mô hình sẽ ưu tiên bắt nhiều spam hơn. Với ngưỡng mặc định 0.5 trên lần chia dữ liệu này, mô hình không tạo false positive nào nhưng vẫn bỏ sót một số spam.
- Learning rate 50. Em dự đoán loss giảm nhanh rồi chững lại. Thực tế đồ thị có một cú tăng vọt ban đầu lên khoảng 2.5, sau đó mới giảm rất nhanh và gần như chạm 0; loss cuối là 0.005. Dự đoán đúng xu hướng giảm về sau nhưng bỏ qua giai đoạn mất ổn định ban đầu. Bước cập nhật quá lớn khiến mô hình ban đầu overshoot trước khi hội tụ đến vùng loss thấp.
- Đường Pareto và SLA 50 ms. Em dự đoán MLP dùng embedding nằm trên Pareto. Kết quả thực tế là Pareto gồm BoW + Logistic Regression, BoW + MLP và Encoder + Logistic Regression; MLP head trên embedding có p95 53.12 ms, vượt SLA, trong khi Encoder + Logistic Regression có p95 45.53 ms và F1 trên SUB 1.000. Em đã dựa nhiều vào kỳ vọng về chất lượng của MLP mà chưa cân nhắc đủ cả độ trễ lẫn F1 thực đo. Phần LLM không có kết quả trong báo cáo nên chưa thể xác nhận dự đoán về LLM.

## 4. Phân tích 50 case khó

BoW + MLP tụt nhiều nhất: F1 giảm từ 0.919 trên SUB xuống 0.324 trên 50 case khó, tức giảm khoảng 0.595 điểm. Nhóm yếu nhất là Spam giọng tự nhiên và Né từ khóa, cùng chỉ 0% chính xác cho mô hình này.
Hai ví dụ thuộc nhóm spam giọng tự nhiên mà cả các mô hình đều đoán sai:
- Case #12: “my friend gave me your number… I'm new in town and looking for…” Đây là spam dụ vào trang hẹn hò, nhưng được viết như tin nhắn cá nhân thân thiện. BoW + MLP dựa nhiều vào các từ riêng lẻ; những từ như “friend” và “number” không thể hiện rõ ý đồ spam.
- Case #13: “Your photos from the event are ready, view & order prints.” Tin nhắn quảng cáo được ngụy trang thành lời nhắc thân thiện. Các từ như “photos” và “event” có thể xuất hiện trong ham; mô hình khó nhận ra ý đồ bán hàng nếu không có dấu hiệu spam rõ ràng.
## 5. Thử thách cá nhân

Cách làm: Em tạo lại vocab với max_size=100 và max_size=5000, rồi vector hóa cùng tập train/test và huấn luyện Logistic Regression bằng cùng cấu hình. Nhờ giữ nguyên seed và cách chia dữ liệu, hai kết quả có thể so sánh trực tiếp.
Kết quả: Khi tăng vocab từ 100 lên 5000 từ, F1 tăng từ 0.846 lên 0.884 (+0.038); precision tăng từ 0.951 lên 1.000, recall tăng từ 0.762 lên 0.792. Thời gian train tăng từ 0.14 giây lên 11.89 giây.
Giải thích: Vocab 100 chỉ giữ các từ phổ biến, nên có thể bỏ mất những từ ít gặp nhưng báo hiệu spam rõ. Vocab 5000 giữ thêm các đặc trưng này, giúp mô hình bắt thêm spam mà không tạo false positive trên tập test lần chạy này. Đổi lại, số chiều lớn hơn khiến phép tính tốn thời gian hơn đáng kể. Kết quả chỉ phản ánh seed và lần chia dữ liệu này; vocab lớn hơn không đảm bảo luôn tốt hơn trên mọi tập.
## 6. Phương án cho sản phẩm thật

Chọn BoW + MLP cho yêu cầu này. 10 triệu SMS/ngày tương đương khoảng 116 tin/giây. Trong lần đo, MLP đạt p95 5.63 ms/tin, thấp hơn đáng kể so với SLA 50 ms; thông lượng theo benchmark là 8,507 tin/giây. F1 test đạt 0.907, không phát sinh chi phí API.

Đánh đổi: Encoder + Logistic Regression có F1 cao hơn (0.939) nhưng p95 45.53 ms, chỉ còn khoảng 4.5 ms dự phòng trước SLA, và thông lượng benchmark thấp hơn (16.9 tin/giây). Có thể chọn phương án này nếu ưu tiên chất lượng và triển khai đủ worker để đáp ứng tải; còn MLP cho dư địa độ trễ và công suất lớn hơn với mức giảm F1 khoảng 0.032.

Các số thông lượng phụ thuộc máy và cách triển khai, nên cần benchmark lại trên hạ tầng production, đặc biệt khi bật tải đồng thời.

## 7. AI log và điều đã học

- Stress test	Model nào tụt F1 nhiều nhất, nhóm khó nào và ví dụ cụ thể.	AI xác định BoW + MLP giảm F1 nhiều nhất, nêu nhóm spam giọng tự nhiên và hai case sai.	Em kiểm tra bảng theo nhóm và output các case sai; ghi rõ nhóm “Né từ khóa” cũng có accuracy 0%.
- Chọn model theo SLA	Chọn phương án cho 10 triệu SMS/ngày với p95 dưới 50 ms.	AI đề xuất BoW + MLP dựa trên F1, p95 và thông lượng đo được.	Em đối chiếu số liệu benchmark và bổ sung lưu ý cần đo lại trên hạ tầng production, nhất là khi có tải đồng thời.
- Điều em hiểu ra hôm nay: Chọn mô hình không thể chỉ nhìn F1; còn phải xét độ trễ, thông lượng, chi phí và khả năng chịu các case khó. Vocab lớn hơn cải thiện kết quả trong lần chia dữ liệu này nhưng làm thời gian train tăng mạnh, nên cần cân bằng chất lượng với tài nguyên và kiểm chứng trên dữ liệu đại diện.
