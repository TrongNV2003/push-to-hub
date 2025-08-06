import os
from huggingface_hub import HfApi
from datasets import load_dataset

def push_local_dataset_to_hub(local_data_dir, repo_id, private=True):
    """
    Tải một dataset từ các file cục bộ, sau đó đẩy nó lên Hugging Face Hub
    theo định dạng Parquet chuẩn.

    Args:
        local_data_dir (str): Đường dẫn đến thư mục chứa các file train.jsonl, validation.jsonl, v.v.
        repo_id (str): ID của repository trên Hub (ví dụ: 'your-username/my-dataset-name').
        private (bool): Đặt là True để tạo một repo riêng tư, False để công khai.
    """
    print(f"Bắt đầu xử lý dataset từ thư mục: '{local_data_dir}'")

    data_files = {
        "train": os.path.join(local_data_dir, "train.jsonl"),
        "validation": os.path.join(local_data_dir, "validation.jsonl"),
        "test": os.path.join(local_data_dir, "test.jsonl")
    }
    
    for split, path in data_files.items():
        if not os.path.exists(path):
            print(f"Cảnh báo: Không tìm thấy file cho split '{split}' tại đường dẫn: {path}")
            
    if not data_files:
        print("Lỗi: Không tìm thấy file dữ liệu nào để xử lý. Vui lòng kiểm tra lại đường dẫn.")
        return

    print("Đang tải các file cục bộ vào đối tượng DatasetDict...")
    local_dataset = load_dataset("json", data_files=data_files)
    
    print("Tải thành công! Cấu trúc dataset:")
    print(local_dataset)

    print(f"\nĐang đẩy dataset lên Hub với repo ID: '{repo_id}'")
    try:
        local_dataset.push_to_hub(
            repo_id=repo_id,
            private=private
        )
        print("\nHoàn tất! Dataset của bạn đã có mặt trên Hugging Face Hub.")
        print(f"Truy cập tại: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi trong quá trình đẩy lên Hub: {e}")
        print("Gợi ý: Hãy chắc chắn bạn đã chạy 'huggingface-cli login' và có quyền ghi vào repo.")


if __name__ == "__main__":
    LOCAL_DATASET_DIR = "your-local-dir" 
    REPO_ID = "your-username/dataset-name"

    # --- Execute ---
    """
    khi đẩy lên sẽ tự động lưu dataset thành Parquet:
    data/
    ├── train-00000-of-00001.parquet
    ├── validation-00000-of-00001.parquet
    └── test-00000-of-00001.parquet
    """
    
    print("Vui lòng đảm bảo bạn đã chạy 'huggingface-cli login' trong terminal trước khi tiếp tục.")
    
    push_local_dataset_to_hub(
        local_data_dir=LOCAL_DATASET_DIR,
        repo_id=REPO_ID,
        private=True
    )