import os
import torch
from loguru import logger
from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer


def push_local_model_to_hub(local_model_dir, repo_id, private=True):
    """
    Tải một mô hình và tokenizer đã được lưu cục bộ, sau đó đẩy chúng lên Hugging Face Hub.

    Args:
        local_model_dir (str): Đường dẫn đến thư mục chứa mô hình đã lưu
                               (ví dụ: kết quả từ trainer.save_model()).
        repo_id (str): ID của repository trên Hub (ví dụ: 'your-username/my-model-name').
        private (bool): Đặt là True để tạo một repo riêng tư.
    """
    logger.info(f"Bắt đầu đẩy mô hình từ thư mục: '{local_model_dir}'")
    if not os.path.exists(local_model_dir):
        logger.error(f"Lỗi: Thư mục '{local_model_dir}' không tồn tại.")
        return

    logger.info("Đang tải mô hình và tokenizer từ đĩa...")
    try:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModel.from_pretrained(local_model_dir, attn_implementation="flash_attention_2", torch_dtype=compute_dtype)
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        logger.info("Tải thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình/tokenizer: {e}")
        return

    logger.info(f"\nĐang đẩy mô hình lên Hub với repo ID: '{repo_id}'")
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
        logger.error(f"\nĐã xảy ra lỗi trong quá trình đẩy lên Hub: {e}")
        print("Gợi ý: Hãy chắc chắn bạn đã chạy 'huggingface-cli login' và có quyền ghi vào repo.")


def push_onnx_folder_to_hub(local_dir: str, repo_id: str, private: bool = True):
    """
    Đẩy toàn bộ thư mục chứa model ONNX lên Hugging Face Hub.
    """
    if not os.path.exists(local_dir):
        logger.error(f"Lỗi: Thư mục '{local_dir}' không tồn tại.")
        return

    api = HfApi()

    logger.info(f"Đang kiểm tra/tạo repository: '{repo_id}'...")
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        logger.info("Repository đã sẵn sàng.")
    except Exception as e:
        logger.error(f"Lỗi khi tạo repo: {e}")
        return

    logger.info(f"Đang tải thư mục '{local_dir}' lên Hub...")
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message="Upload ONNX model optimized for FastEmbed"
        )
        
        print("\nHoàn tất! Model ONNX của bạn đã có mặt trên Hugging Face Hub.")
        print(f"Truy cập tại: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        logger.error(f"\nĐã xảy ra lỗi trong quá trình đẩy lên Hub: {e}")
        print("Gợi ý: Hãy chắc chắn bạn đã chạy 'huggingface-cli login' bằng token có quyền WRITE.")


if __name__ == "__main__":
    LOCAL_MODEL_DIR = "./your-local-dir" 
    REPO_ID = "your-username/model-name"

    logger.warning("Vui lòng đảm bảo bạn đã chạy 'huggingface-cli login' trong terminal trước khi tiếp tục.")

    # pytorch model
    # push_local_model_to_hub(
    #     local_model_dir=LOCAL_MODEL_DIR,
    #     repo_id=REPO_ID,
    #     private=True
    # )

    # onnx model
    push_onnx_folder_to_hub(
        local_dir=LOCAL_MODEL_DIR,
        repo_id=REPO_ID,
        private=False
    )