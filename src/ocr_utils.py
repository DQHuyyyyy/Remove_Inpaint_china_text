import numpy as np
import cv2

def is_chinese(text):
    # Regex kiểm tra:
    # - Ký tự Trung Quốc: \u4e00-\u9fff
    # - Bopomofo (ký tự phát âm tiếng Trung): \u3100–\u312F
    # - Radicals Trung Quốc: \u2E80-\u2EFF
    # - Ký hiệu pinyin có dấu: āáǎàēéěè...
    # - Ký tự đặc biệt hoặc dấu câu Trung: ，。！？【】、《》；：“”（）
    special_chinese_symbols = r"[，。！？【】、《》；：“”（）~]"

    return (
        re.search(r'[\u4e00-\u9fff]', text) or
        re.search(r'[\u3100-\u312F]', text) or
        re.search(r'[\u2E80-\u2EFF]', text) or
        re.search(r'[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]', text) or
        re.search(special_chinese_symbols, text)
    )
def create_mask_from_boxes(image_shape, ocr_result, expand_px=5):
    """
    Tạo mask trắng trên nền đen từ kết quả OCR, có mở rộng box.
    - expand_px: số pixel mở rộng mỗi chiều
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    height, width = image_shape[:2]

    for line in ocr_result:
        for word_info in line:
            box = np.array(word_info[0]).astype(np.int32)
            x_min = np.clip(np.min(box[:, 0]) - expand_px, 0, width - 1)
            x_max = np.clip(np.max(box[:, 0]) + expand_px, 0, width - 1)
            y_min = np.clip(np.min(box[:, 1]) - expand_px, 0, height - 1)
            y_max = np.clip(np.max(box[:, 1]) + expand_px, 0, height - 1)

            # Fill vùng mở rộng thay vì chỉ box gốc
            mask[y_min:y_max, x_min:x_max] = 255

    return mask

