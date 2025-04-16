from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from src.helper import load_img
from src.core_goc import process_inpaint
from src.ocr_utils import create_mask_from_boxes
import cv2
import numpy as np
import tempfile
import os

ocr = PaddleOCR(use_angle_cls=True, lang='ch', det_db_box_thresh=0.3)

def smooth_mask(mask: np.ndarray, kernel_size: int = 5, dilate_iter: int = 2) -> np.ndarray:
    """
    Làm mượt và mở rộng mask bằng dilation, closing và Gaussian blur.
    - kernel_size: kích thước nhân cho closing và blur
    - dilate_iter: số lần giãn mask (dilate)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 🔸 Mở rộng mask bằng Dilation
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # 🔸 Làm mượt mask (Closing = Dilation + Erosion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 🔸 Làm mờ để mềm vùng biên (tùy chọn)
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    # 🔸 Chuẩn hóa lại
    return (mask > 20).astype(np.uint8) * 255


def auto_remove_chinese_text(image_path: str) -> np.ndarray:
    img_np = load_img(open(image_path, "rb").read())
    ocr_result = ocr.ocr(image_path, cls=True)
    print("🔍 OCR Result:", ocr_result)

    # Tạo mask từ kết quả OCR
    mask_np = create_mask_from_boxes(img_np.shape, ocr_result)

    # ✅ Làm mượt mask
    mask_np = smooth_mask(mask_np)
    mask_np = 255 - mask_np  # đảo ngược mask để thử


    # ✅ Lưu mask ra file để kiểm tra
    #Image.fromarray(mask_np).save("mask_debug.png")

    # Inpaint ảnh
    inpainted = process_inpaint(img_np, mask_np)
    return inpainted



def process_image(image: Image.Image) -> Image.Image:
    """
    Hàm này được API gọi. Nhận ảnh PIL.Image và trả về ảnh đã inpaint.
    """
    # 🔸 Chuyển ảnh thành file tạm (vì hàm hiện tại dùng path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # 🔸 Gọi pipeline xử lý ảnh từ đường dẫn
    result_np = auto_remove_chinese_text(temp_path)

    # 🔸 Xóa file tạm
    os.remove(temp_path)

    # 🔸 Convert kết quả từ np.ndarray → PIL.Image để trả về
    result_img = Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB))

    return result_img
