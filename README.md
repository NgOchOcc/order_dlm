# LLaDA Diffusion Language Model - Math500 Evaluation

Script để inference và đánh giá **LLaDA Diffusion Language Model** trên bộ dataset Math500.

**LLaDA là một Diffusion Language Model**, không phải autoregressive model thông thường. Script này sử dụng đúng phương pháp diffusion sampling của LLaDA với:
- Mask prediction và iterative refinement
- Semi-autoregressive generation với block sampling
- Gumbel noise cho categorical sampling
- Configurable remasking strategies

**Sử dụng thư viện `math_verify` để đánh giá chính xác tương đương toán học.**

## Cài đặt

```bash
pip install -r requirements.txt
```

**Lưu ý:** Script sử dụng `math_verify` để kiểm tra tương đương toán học (ví dụ: `1/2` = `0.5`, `x+x` = `2x`). Nếu `math_verify` không có sẵn, script sẽ tự động fallback về so sánh chuỗi đơn giản.

## Sử dụng

### Test nhanh:

```bash
# Test xem model có hoạt động không
python test_llada_quick.py
```

### Chạy evaluation cơ bản:

```bash
# Sử dụng script diffusion (KHUYẾN NGHỊ cho LLaDA)
python inference_llada_math500_diffusion.py
```

### Chạy với model cụ thể:

```bash
# LLaDA-8B-Base
python inference_llada_math500_diffusion.py --model_name GSAI-ML/LLaDA-8B-Base

# LLaDA-8B-Instruct (có chat template)
python inference_llada_math500_diffusion.py --model_name GSAI-ML/LLaDA-8B-Instruct
```

### Các tùy chọn diffusion:

```bash
# Tùy chỉnh diffusion parameters
python inference_llada_math500_diffusion.py \
  --steps 128 \              # Số bước diffusion (càng nhiều càng chất lượng cao)
  --gen_length 256 \         # Độ dài tối đa sinh ra
  --block_length 32 \        # Block length cho semi-autoregressive
  --temperature 0.0 \        # Temperature (0 = greedy)
  --cfg_scale 0.0            # Classifier-free guidance scale

# Chỉ đánh giá 100 mẫu đầu tiên
python inference_llada_math500_diffusion.py --max_samples 100

# Sử dụng dataset local
python inference_llada_math500_diffusion.py --dataset_path ./math500.json

# Lưu kết quả vào file khác
python inference_llada_math500_diffusion.py --output_file my_results.json
```

## Tính năng chính

1. **Diffusion Sampling**: Triển khai đúng phương pháp diffusion sampling của LLaDA
   - Mask prediction với iterative refinement
   - Semi-autoregressive generation
   - Gumbel noise sampling với float64 precision
   - Low-confidence và random remasking strategies

2. **Model Support**:
   - LLaDA-8B-Base
   - LLaDA-8B-Instruct (tự động sử dụng chat template)
   - Dream models (nếu tương thích)

3. **Answer Extraction**: Tự động trích xuất đáp án từ output
   - Hỗ trợ nhiều format: `\boxed{}`, `####`, plain text
   - Regex patterns cho final answer

4. **Mathematical Verification**: Sử dụng `math_verify`
   - So sánh chính xác: `1/2` ≡ `0.5`, `x+x` ≡ `2x`, `(x+1)^2` ≡ `x^2+2x+1`
   - Fallback tự động nếu thư viện không có sẵn

5. **Evaluation Metrics**: Tính accuracy và lưu kết quả chi tiết
6. **Progress Tracking**: Hiển thị tiến độ và accuracy theo thời gian thực
7. **Flexible Dataset**: Hỗ trợ Math500, MATH, GSM8K hoặc dataset tùy chỉnh

## Output

Kết quả sẽ được lưu trong file JSON với format:

```json
{
  "model": "GSAI-ML/LLaDA-8B-Base",
  "total_samples": 500,
  "correct": 385,
  "accuracy": 77.0,
  "use_cot": true,
  "results": [
    {
      "index": 0,
      "problem": "...",
      "ground_truth": "42",
      "generated_solution": "...",
      "predicted_answer": "42",
      "is_correct": true
    }
  ]
}
```

## Cấu trúc Script

### inference_llada_math500_diffusion.py ⭐ (KHUYẾN NGHỊ)
Script chính với đầy đủ diffusion sampling cho LLaDA:
- `generate_llada()`: Function diffusion sampling (từ LLaDA official code)
- `add_gumbel_noise()`: Thêm Gumbel noise cho categorical sampling
- `get_num_transfer_tokens()`: Tính số token cần transfer ở mỗi step
- `LLaDAMath500Evaluator`: Class chính để evaluation
  - `load_math500_dataset()`: Load dataset
  - `create_prompt()`: Tạo prompt (hỗ trợ Base và Instruct)
  - `generate_answer()`: Generate với diffusion sampling
  - `extract_answer()`: Trích xuất đáp án cuối cùng
  - `check_answer_correct()`: Kiểm tra bằng math_verify
  - `evaluate_dataset()`: Chạy evaluation trên toàn bộ dataset

### inference_llada_math500.py
Script cũ với autoregressive generation (không phù hợp cho LLaDA)

### test_llada_quick.py
Test nhanh LLaDA model với 1 câu hỏi:
```bash
python test_llada_quick.py
```

### test_math_verify.py
Test hoạt động của math_verify:
```bash
python test_math_verify.py
```

## LLaDA Diffusion Parameters

### Các tham số quan trọng:

- **steps**: Số bước diffusion sampling (default: 128)
  - Càng nhiều → chất lượng cao hơn nhưng chậm hơn
  - Khuyến nghị: 64-256

- **gen_length**: Độ dài tối đa sinh ra (default: 256)
  - Phải chia hết cho block_length

- **block_length**: Block size cho semi-autoregressive (default: 32)
  - Nhỏ hơn → autoregressive hơn, chậm hơn
  - Lớn hơn → parallel hơn, nhanh hơn
  - Khuyến nghị: 16-64

- **temperature**: Sampling temperature (default: 0.0)
  - 0 = greedy (deterministic)
  - > 0 = stochastic sampling

- **cfg_scale**: Classifier-free guidance (default: 0.0)
  - Tăng để tăng chất lượng (nhưng giảm diversity)
  - Chỉ có ý nghĩa với Instruct model

- **remasking**: Strategy cho remasking
  - `low_confidence`: Remask tokens có confidence thấp (khuyến nghị)
  - `random`: Random remasking

## Lưu ý

- **GPU Memory**: Model 8B yêu cầu ~16GB VRAM với bfloat16
- **Speed**: Diffusion sampling chậm hơn autoregressive (đổi lại chất lượng cao hơn)
- **Model Type**:
  - Base model: Dùng prompt đơn giản
  - Instruct model: Tự động dùng chat template
- **Dataset**: Math500 có thể thay bằng MATH hoặc GSM8K nếu không có sẵn
- **Left Padding**: LLaDA yêu cầu left padding (script tự động set)
