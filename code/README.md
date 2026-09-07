# Mã nguồn thực nghiệm

Cài đặt PINNs bằng PyTorch thuần, không dùng thư viện PINNs đóng gói sẵn, dùng
cho toàn bộ Chương 5 của luận văn.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

```bash
python -m pinns.exp1_relu        # TN1: hàm kích hoạt, sự suy biến của ReLU
python -m pinns.exp2_lambda      # TN2: cái giá của ràng buộc mềm
python -m pinns.exp3_spectral    # TN3: thiên kiến phổ, đảo chiều do toán tử
python -m pinns.exp4_heat        # TN4: phương trình khuếch tán
python -m pinns.exp5_burgers     # TN5: Burgers độ nhớt nhỏ
python -m pinns.exp6_stability   # TN6: kiểm chứng chặn ổn định
python -m pinns.exp7_nu_sweep    # TN7: quét ν, kiểm chứng chặn suy biến 1/ν
python -m pinns.exp8_seeds       # TN8: lặp TN4/TN5 trên năm hạt giống
python -m pinns.exp9_chiphi      # TN9: chi phí tính toán, PINN so với sai phân
python -m pinns.run_all          # chạy tất cả
```

`exp8_seeds` ghi kết quả ra `results/` **sau mỗi hạt giống** và đọc lại phần đã
có khi chạy lại, nên một lần chạy bị ngắt không làm mất gì. Nó nhận thêm tham số
`heat` hoặc `burgers` để chạy riêng một nửa.

Kết quả số ghi ra `results/*.json`.

## Quy ước bám theo luận văn

| Thành phần | Thiết lập | Truy về |
|---|---|---|
| Hàm kích hoạt | `tanh` | điều kiện $\sigma''\not\equiv0$ cho toán tử bậc hai |
| Khởi tạo | Xavier chuẩn tắc, $\mathrm{Var}(w)=2/(n_{in}+n_{out})$ | `eq:xavier` |
| Độ chệch | $0$ | giả thiết (A2) |
| Lớp ra | tuyến tính | `eq:fnn-output` |
| Độ chính xác | `float64` | phân biệt sai số xấp xỉ với sai số làm tròn |
| Tối ưu | Adam $10^{-3}$ → L-BFGS (Wolfe mạnh) | Thuật toán hai giai đoạn |
| Điểm phối trí 1D | `linspace(a, b, n)`, **kể cả hai điểm biên** | xem ghi chú dưới |
| Lưới đánh giá | khác tập huấn luyện | Định nghĩa chỉ tiêu sai số |

**Ghi chú về lưới điểm phối trí.** Lưới 1D bao gồm cả hai điểm biên. Với
$n=256$ trên $(0,1)$ và $f=4\pi^2\sin(2\pi x)$, trung bình $f^2$ trên lưới này
bằng $776{,}229$ — đúng giá trị $J_r$ mà TN1 báo cáo cho mạng ReLU. Quy ước này
được xác định bằng cách đối chiếu ngược với số liệu trong luận văn.

## Tái lập

Mọi thí nghiệm dùng một hạt giống cố định (`seed = 0`) và `float64`.

**Nhưng hạt giống cố định là chưa đủ để tái lập từng chữ số.** Kết quả còn phụ
thuộc **số luồng BLAS**. Đo trực tiếp trên bài toán Poisson, mạng `(1,32,32,32,1)`:

| | `OMP_NUM_THREADS=1` | `=2` | `=4` |
|---|---|---|---|
| `‖W⁽¹⁾‖` khởi tạo | `1.5813022424358212` | giống hệt | giống hệt |
| `J` tại vòng 0 | `777.60307687791374` | giống hệt | giống hệt |
| `J` tại vòng 100 | `28.153617819651501` | `...505` | `...505` |
| `J` tại vòng 1499 | `0.023262825` | `0.023287981` | `0.023260501` |

Khởi tạo và vòng lặp đầu tiên giống nhau từng bit; sai khác chỉ xuất hiện khi
tích luỹ qua nhiều bước, vì số luồng đổi **thứ tự cộng dồn** trong phép nhân ma
trận. Sai khác cỡ epsilon máy ban đầu được quỹ đạo tối ưu hoá khuếch đại: sau
$1\,500$ vòng đã lệch ở chữ số có nghĩa thứ tư.

Hệ quả thực hành:

- Muốn tái lập từng chữ số, phải cố định **cả** hạt giống **lẫn** số luồng
  (`OMP_NUM_THREADS`), và dùng cùng phiên bản PyTorch.
- Với các thí nghiệm dùng lấy mẫu ngẫu nhiên (TN3, TN4, TN5), chỉ nên kỳ vọng
  tái lập được **kết luận định tính**, không phải con số.
- Các thí nghiệm dùng lưới tất định (TN1, TN2, TN6) và bộ giải tham chiếu sai
  phân hữu hạn của TN5 thì ổn định hơn hẳn: chúng tái lập tới ba đến bốn chữ số
  có nghĩa so với số liệu trong luận văn.

Xem thêm phần Hạn chế của luận văn về việc một hạt giống duy nhất là chưa đủ để
rút kết luận thống kê.
