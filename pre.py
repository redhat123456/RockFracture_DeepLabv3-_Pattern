import cv2
import numpy as np
import os
import json

# -----------------------------
# 1. 配置路径
# -----------------------------
INPUT_ORIGIN_DIR = "dataset/origin/"          
INPUT_JSON_DIR = "dataset/json/"      
OUTPUT_IMAGE_DIR = "dataset/images_split/" 
OUTPUT_MASK_DIR = "dataset/masks_split/"  
OUTPUT_MASK1_DIR = "dataset/mask/"

os.makedirs(OUTPUT_MASK1_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# -----------------------------
# 2. 核心处理函数
# -----------------------------

def preprocess_image(img_path):
    """图像增强：CLAHE + 双边滤波 + 高通滤波"""
    try:
        data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None: return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        highpass = cv2.filter2D(bilateral, -1, kernel)
        final = cv2.addWeighted(bilateral, 1.0, highpass, 0.7, 0)
        return final
    except Exception as e:
        print(f"读取图片出错: {e}")
        return None

def json_to_mask(json_path, img_shape):
    """将数字标签或crack标签统一转为黑色Mask"""
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    h, w = data.get('imageHeight', img_shape[0]), data.get('imageWidth', img_shape[1])
    # 初始化白色背景
    mask = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    found_valid = False
    for shape in data['shapes']:
        label = str(shape.get("label", "")).lower().strip()
        points = np.array(shape["points"], dtype=np.int32)
        
        # 核心判定：如果是数字，或者是包含crack的字符串，则涂黑
        if label.isdigit() or "crack" in label:
            cv2.fillPoly(mask, [points], (0, 0, 0))
            found_valid = True
        else:
            print(f"  -> 跳过非相关标签: {label}")

    return mask if found_valid else None

def split_and_rotate(img, num_parts=10):
    """循环移位拼接数据增强"""
    h, w = img.shape[:2]
    part_w = w // num_parts
    actual_w = num_parts * part_w
    img_clipped = img[:, :actual_w]
    
    parts = [img_clipped[:, i*part_w:(i+1)*part_w] for i in range(num_parts)]
    
    results = []
    for i in range(num_parts):
        new_order = parts[i:] + parts[:i]
        results.append(np.hstack(new_order))
    return results

# -----------------------------
# 3. 主程序
# -----------------------------
image_files = sorted([f for f in os.listdir(INPUT_ORIGIN_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))])

print(f"🚀 启动增强与同步切分任务，目标图片数: {len(image_files)}")

for filename in image_files:
    name_base = os.path.splitext(filename)[0]
    img_path = os.path.join(INPUT_ORIGIN_DIR, filename)
    json_path = os.path.join(INPUT_JSON_DIR, name_base + ".json")

    # 1. 图像增强
    enhanced_img = preprocess_image(img_path)
    if enhanced_img is None: continue

    # 2. 生成掩码 (包含数字兼容逻辑)
    mask = json_to_mask(json_path, enhanced_img.shape)
    if mask is None:
        print(f"⚠️ 跳过: {filename} (未发现有效数字或crack标签)")
        continue

    # 3. 保存原始图像的掩码到 mask/ 文件夹
    original_mask_path = os.path.join(OUTPUT_MASK1_DIR, name_base + "_mask.png")
    cv2.imwrite(original_mask_path, mask)
    print(f"📌 已保存原始掩码: {name_base}_mask.png")

    # 4. 同步执行循环移位增强
    aug_images = split_and_rotate(enhanced_img, num_parts=10)
    aug_masks = split_and_rotate(mask, num_parts=10)

    # 5. 保存增强后的图像和掩码
    for i, (ri, rm) in enumerate(zip(aug_images, aug_masks), 1):
        save_base = f"{name_base}_aug{i:02d}"
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, save_base + ".jpg"), ri)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, save_base + ".png"), rm)

    print(f"✅ 完成同步切分: {filename}")

print("\n✨ 处理全部完成！")
print(f"   - 原始掩码已保存到: {OUTPUT_MASK1_DIR}")
print(f"   - 增强掩码已保存到: {OUTPUT_MASK_DIR}")