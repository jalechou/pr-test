import cv2
import numpy as np
import joblib
import os

def extract_features_consistency(img):
    """
    特征提取函数：必须与训练时的 _extract_features 逻辑完全一致
    """
    height, width = img.shape[:2]
    f_list = []
    
    # 1. BGR 颜色特征
    img_float = img.astype(np.float32) / 255.0
    f_list.append(img_float)
    
    # 2. HSV 颜色特征
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    f_list.append(hsv)
    
    # 3. 坐标特征 (0 到 1 归一化)
    # 注意：即便新图尺寸不同，0-1的相对坐标也能保证位置逻辑的一致性
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    f_list.append(xv[..., None].astype(np.float32))
    f_list.append(yv[..., None].astype(np.float32))
    
    # 4. 纹理特征
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    f_list.append(cv2.GaussianBlur(gray, (7, 7), 0)[..., None])
    
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    f_list.append(mag[..., None])
    
    # 合并并展平
    full_features = np.concatenate(f_list, axis=2)
    return full_features.reshape(-1, full_features.shape[-1])

def predict_new_image(image_path, model_path, output_dir="results"):
    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    model = joblib.load(model_path)
    print(f"成功加载模型: {model_path}")

    # 2. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    h, w = img.shape[:2]

    # 3. 提取特征
    print(f"正在处理: {os.path.basename(image_path)} ...")
    features = extract_features_consistency(img)

    # 4. 模型预测
    # preds 结果是 0 或 1 的数组
    preds = model.predict(features)
    
    # 5. 生成掩膜 (Mask)
    mask = preds.reshape(h, w).astype(np.uint8) * 255
    
    # 可选：对掩膜进行简单的后处理（去噪）
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 开运算去小白点

    # 6. 可视化效果 (将原图和黄色遮罩叠加)
    overlay = img.copy()
    color_mask = np.zeros_like(img)
    color_mask[mask == 255] = [0, 255, 255] # 黄色
    result_vis = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

    # 7. 保存结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(image_path)
    mask_save_path = os.path.join(output_dir, "mask_" + base_name)
    vis_save_path = os.path.join(output_dir, "vis_" + base_name)
    
    cv2.imwrite(mask_save_path, mask)
    cv2.imwrite(vis_save_path, result_vis)
    print(f"结果已保存至: {output_dir}")
    
    # 展示一下
    cv2.imshow("Automated Prediction", result_vis)
    cv2.waitKey(0)

# --- 使用示例 ---
if __name__ == "__main__":
    # 指定你保存的模型文件
    MODEL_FILE = "lgbm_segmentor_model.joblib"
    
    # 指定你想要预测的新图片路径
    # 它可以是另一张图片，也可以是同一张图片的不同版本
    NEW_IMG_PATH = "P-temp-09062023130944-0000_00077.png" 
    
    if os.path.exists(NEW_IMG_PATH):
        predict_new_image(NEW_IMG_PATH, MODEL_FILE)
    else:
        print(f"请将需要预测的图片放入路径: {NEW_IMG_PATH}")