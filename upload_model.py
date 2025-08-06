import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def push_local_model_to_hub(local_model_dir, repo_id, private=True):
    """
    Tải một mô hình và tokenizer đã được lưu cục bộ, sau đó đẩy chúng lên
    Hugging Face Hub.

    Args:
        local_model_dir (str): Đường dẫn đến thư mục chứa mô hình đã lưu
                               (ví dụ: kết quả từ trainer.save_model()).
        repo_id (str): ID của repository trên Hub (ví dụ: 'your-username/my-model-name').
        private (bool): Đặt là True để tạo một repo riêng tư.
    """
    print(f"Bắt đầu đẩy mô hình từ thư mục: '{local_model_dir}'")
    if not os.path.exists(local_model_dir):
        print(f"Lỗi: Thư mục '{local_model_dir}' không tồn tại.")
        return

    print("Đang tải mô hình và tokenizer từ đĩa...")
    try:
        # Sử dụng đúng class cho tác vụ của model: AutoModelForQuestionAnswering, AutoModelForCausalLM, v.v.
        model = AutoModelForQuestionAnswering.from_pretrained(local_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        print("Tải thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình/tokenizer: {e}")
        return

    print(f"\nĐang đẩy mô hình lên Hub với repo ID: '{repo_id}'")
    try:
        model.push_to_hub(
            repo_id=repo_id,
            private=private,
            commit_message="Upload model weights"
        )
        tokenizer.push_to_hub(
            repo_id=repo_id,
            private=private,
            commit_message="Upload tokenizer"
        )
        
        print("\nHoàn tất! Mô hình và tokenizer của bạn đã có mặt trên Hugging Face Hub.")
        print(f"Truy cập tại: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\nĐã xảy ra lỗi trong quá trình đẩy lên Hub: {e}")
        print("Gợi ý: Hãy chắc chắn bạn đã chạy 'huggingface-cli login' và có quyền ghi vào repo.")


# --- CÁCH SỬ DỤNG ---
if __name__ == "__main__":
    LOCAL_MODEL_DIR = "./your-local-dir" 
    REPO_ID = "your-username/model-name"

    print("Vui lòng đảm bảo bạn đã chạy 'huggingface-cli login' trong terminal trước khi tiếp tục.")
    
    push_local_model_to_hub(
        local_model_dir=LOCAL_MODEL_DIR,
        repo_id=REPO_ID,
        private=True
    )