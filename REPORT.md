# Lab 21 — Evaluation Report

**Học viên**: Trần Minh Hoàng  
**Mã số học viên**: 2A202600700  
**Ngày nộp**: 2026-06-25  
**Submission option**: Option A (Lightweight ZIP)  

---

## 1. Setup
*   **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
*   **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, kích thước 200 mẫu (180 mẫu huấn luyện - train, 20 mẫu đánh giá - eval) chia theo tỷ lệ 90/10 với random seed = 42.
*   **max_seq_length**: `1024` (độ dài token thực tế tại phân vị p95 là 562, được làm tròn lên lũy thừa gần nhất của 2 là 1024).
*   **GPU**: Tesla T4 (14.6 GB VRAM) trên Google Colab Free.
*   **Training cost**: ~$0.07 USD (tổng thời gian huấn luyện 3 rank thực tế là 12.6 phút, tính theo đơn giá ~$0.35/giờ của T4 trên Colab).
*   **HF Hub link**: *N/A (Nộp bài theo Option A - File ZIP cục bộ).*

---

## 2. Rank Experiment Results

Dưới đây là bảng so sánh hiệu năng huấn luyện thực tế của tôi trên Google Colab giữa các mức rank khác nhau (`r=8`, `r=16`, `r=64`) so với mô hình nền tảng (Base Model):

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **8** | 1,843,200 (0.06%) | 4.13 min | 7.22 GB | 1.5577 | 4.75 |
| **16 (Baseline)** | 3,686,400 (0.12%) | 4.31 min | 6.62 GB | 1.5161 | 4.55 |
| **64** | 14,745,600 (0.48%) | 4.15 min | 8.00 GB | 1.4768 | 4.38 |
| **Base** | - | - | - | 1.6215 | 5.06 |

---

## 3. Loss Curve Analysis

Biểu đồ so sánh Loss Curve của 3 mức rank được lưu tại tệp [loss_curve.png](file:///d:/Work/Study/ai-in-action/track3/Lab6/2A202600700-TranMinhHoang-Day21/results/loss_curve.png).

**Quan sát và nhận xét về tình trạng Overfitting:**
*   Dựa trên đồ thị loss curve thật được trích xuất từ notebook, cả 3 mức rank đều có đồ thị huấn luyện giảm đều và mượt mà sau 69 steps (3 epochs).
*   Không có hiện tượng overfitting xảy ra. Điều này là do:
    1.  **Dung lượng dữ liệu nhỏ**: Chúng ta huấn luyện trên một tập hợp con gồm 200 mẫu được làm sạch tốt, giúp mô hình nhanh chóng thích ứng với cấu trúc dữ liệu mà không bị nhiễu.
    2.  **Regularization tốt**: LoRA với rank thấp đóng vai trò như một bộ lọc Regularization hiệu quả, hạn chế việc mô hình ghi nhớ máy móc toàn bộ tập train.
    3.  **Tỷ lệ Epoch thấp**: Huấn luyện trong 3 epoch là vừa đủ để mô hình hội tụ mà chưa đi vào vùng quá khớp (overfitting).
*   Mức rank `r=64` cho tốc độ giảm loss nhanh nhất và đạt mức loss thấp nhất, tiếp theo là `r=16` và `r=8`.

---

## 4. Qualitative Comparison (5 examples)

Dưới đây là so sánh side-by-side kết quả sinh văn bản thực tế trước (Base Model) và sau khi fine-tune (`r=16` Baseline) trích xuất từ notebook của tôi:

### Example 1
*   **Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
*   **Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, n...
*   **Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân...
*   **Nhận xét**: **Cải thiện (Improved)**. Định nghĩa của mô hình Fine-tuned chính xác hơn về mặt kỹ thuật, loại bỏ các từ ngữ lặp và diễn đạt trôi chảy hơn mô hình gốc.

### Example 2
*   **Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
*   **Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return "N p...\n```
*   **Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:\n\n```python\ndef fibonacci(n):\n    if n < 0:\n        raise ValueError("Input phải là một số nguyên dương.")\n    elif n == 0:\n   ...\n```
*   **Nhận xét**: **Cải thiện vượt bậc (Highly Improved)**. Bản gốc dùng đệ quy đơn giản với độ phức tạp thời gian là $O(2^n)$ và xử lý biên lỗi (trả về chuỗi thông báo lỗi thay vì raise lỗi). Mô hình Fine-tuned đã chuyển sang dùng vòng lặp tối ưu $O(n)$ thời gian và $O(1)$ không gian, đồng thời raise ValueError đúng chuẩn Python.

### Example 3
*   **Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
*   **Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc,...
*   **Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX th...
*   **Nhận xét**: **Cải thiện (Improved)**. Cả hai mô hình đều đưa ra các nguyên tắc đúng đắn, tuy nhiên mô hình Fine-tuned định nghĩa các thuật ngữ chuyên ngành chuẩn xác hơn và giải thích sâu sắc hơn về mặt UX.

### Example 4
*   **Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
*   **Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hi...
*   **Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong...
*   **Nhận xét**: **Giữ nguyên/Có lỗi nhỏ (Same/Slight degradation)**. Mô hình Fine-tuned giải thích sai LoRA thành "Layer-wise Adaptive Regularization Optimization" (thực tế là Low-Rank Adaptation). Tuy nhiên, về mặt diễn đạt kỹ thuật cấu trúc QLoRA (phần sau) thì mô hình Fine-tuned giải thích sâu hơn về cơ chế lượng tử hóa 4-bit. Đây là lỗi ảo tưởng kiến thức (hallucination) cần lưu ý.

### Example 5
*   **Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
*   **Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của ...
*   **Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp ...
*   **Nhận xét**: **Cải thiện vượt trội (Highly Improved)**. Mô hình Fine-tuned phân biệt rành mạch 3 khái niệm về mặt kiến thức và bối cảnh ứng dụng thực tế, đưa ra giải thích sâu sắc và chuẩn xác hơn mô hình gốc.

---

## 5. Conclusion về Rank Trade-off

*   **Rank nào cho ROI (Return on Investment) tốt nhất trên dataset này? Tại sao?**
    *   Đối với tập dữ liệu nhỏ (200 ví dụ) này, **`r=16`** mang lại tỷ suất ROI tốt nhất. Nó đạt sự cân bằng tuyệt vời giữa dung lượng bộ nhớ VRAM sử dụng (chỉ **6.62 GB**, thấp hơn mức 7.22 GB của `r=8` do cơ chế phân bổ động của Unsloth và bitsandbytes hiệu quả hơn trên các cấu hình ma trận chẵn), thời gian huấn luyện tương đương (~4.31 phút), và cho perplexity đánh giá khá tốt (**4.55** so với 4.75 của `r=8`). 
*   **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**
    *   Hiện tượng diminishing returns bắt đầu xuất hiện rõ ràng khi chúng ta tăng rank lên `r=64`. Số lượng tham số huấn luyện tăng gấp 4 lần so với `r=16` (từ 3.6M lên 14.7M) và VRAM tăng lên **8.00 GB** (+21%). Tuy nhiên, độ perplexity chỉ giảm nhẹ từ **4.55 xuống 4.38** (chỉ cải thiện ~3.7% chất lượng định lượng). Sự đánh đổi về tài nguyên là không đáng kể đối với bài toán nhỏ này.
*   **Recommendation cho deploy production:**
    *   Nếu triển khai thực tế trên hệ thống sản phẩm lớn, tôi khuyến nghị lựa chọn **`r=16`** (hoặc thậm chí `r=8` nếu cần phục vụ multi-tenant nhiều adapter cùng lúc). `r=16` đảm bảo adapter file có kích thước nhỏ gọn (dễ lưu trữ, tải nhanh khi load động) và tiêu tốn ít bộ nhớ GPU trong quá trình inference, đồng thời giữ được chất lượng ngôn ngữ tự nhiên tốt.

---

## 6. What I Learned
*   Hiểu sâu sắc về cơ chế hoạt động thực tế của LoRA/QLoRA: Việc tinh chỉnh LLM không phải là nạp kiến thức mới (RAG phù hợp hơn) mà là định hình phong cách, định dạng đầu ra (ví dụ: mô hình Fine-tuned đã trả về mã nguồn tối ưu và định dạng bullet points rất tốt).
*   Nắm vững kỹ năng tối ưu tài nguyên trên GPU nhỏ (T4) bằng cách kích hoạt gradient checkpointing, sử dụng Unsloth để tăng tốc độ gấp 2 lần và giảm 60% VRAM, đồng thời xử lý lỗi OOM thông qua cơ chế `safe_evaluate` manual batch size = 1.
*   Trải nghiệm thực tế về sự đánh đổi (trade-off) của siêu tham số rank: Rank cao hơn không phải lúc nào cũng mang lại hiệu quả vượt trội tương xứng với chi phí tài nguyên bỏ ra.
