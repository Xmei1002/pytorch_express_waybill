from config import root_dir, train_data_file, test_data_file
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from models.east_model import EAST  # 您的EAST模型定义
from utils.dataset import MyDataset  # 您的数据集类
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.metrics import precision_recall_fscore_support
import cv2
from shapely.geometry import Polygon as ShapelyPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EAST().to(device)
model.load_state_dict(torch.load('east_model.pth', map_location=device))
model.eval()

# 准备测试数据
transform = transforms.Compose([
    transforms.Resize((1080, 1080)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = MyDataset(root_dir, test_data_file, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
def iou(box1, box2):
    """
    计算两个边界框的IoU。
    box1, box2: 形状为[4, 2]的边界框，每个边界框由四个角点的(x, y)坐标组成。
    """
    box1 = box1.reshape(4, 2)
    box2 = box2.reshape(4, 2)

    poly1 = ShapelyPolygon(box1).convex_hull
    poly2 = ShapelyPolygon(box2).convex_hull

    if not poly1.intersects(poly2):  # 如果两个多边形不相交
        return 0.0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - intersection_area
    iou = intersection_area / union_area
    return iou

# 解码模型输出，转换为边界框
def decode_predictions(score_map, geo_map, score_map_thresh=1, nms_thresh=0.1):
    detections = []
    score_map = score_map.squeeze(0).cpu().detach().numpy()  # 先detach再转换为numpy数组
    geo_map = geo_map.squeeze(0).cpu().detach().numpy()

    # 确保score_map是二维的
    score_map = np.squeeze(score_map)

    # 遍历得分图
    for y in range(score_map.shape[0]):
        for x in range(score_map.shape[1]):
            if score_map[y, x] < score_map_thresh:  # 比较单个值
                continue

            geo_data = geo_map[:, y, x]
            top, right, bottom, left, angle = geo_data
            if angle < -np.pi / 4:
                angle = angle + np.pi / 2

            x0, y0 = x, y
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            h, w = top + bottom, right + left
            offset = [[-cos_a * sin_a * h + w * cos_a, sin_a * cos_a * h + w * sin_a],
                      [cos_a * sin_a * h + w * cos_a, -sin_a * cos_a * h + w * sin_a],
                      [-cos_a * sin_a * h - w * cos_a, sin_a * cos_a * h - w * sin_a],
                      [cos_a * sin_a * h - w * cos_a, -sin_a * cos_a * h - w * sin_a]]
            box = [[x0 + offset[0][0], y0 + offset[0][1]],
                   [x0 + offset[1][0], y0 + offset[1][1]],
                   [x0 + offset[2][0], y0 + offset[2][1]],
                   [x0 + offset[3][0], y0 + offset[3][1]]]
            detections.append(box)

    boxes = np.array(detections).reshape((-1, 8))
    boxes = non_max_suppression(boxes, nms_thresh)
    boxes = boxes.reshape((-1, 4, 2))
    # print('boxes:',boxes)
    return boxes


def non_max_suppression(boxes, iou_threshold):
    """
    非极大值抑制。
    boxes: 边界框数组，每个边界框由8个值（4个角点的x, y坐标）组成。
    iou_threshold: IoU阈值，用于决定是否去除重叠的边界框。
    """
    assert boxes.ndim == 2 and boxes.shape[1] == 8, "boxes应该是一个形状为[N, 8]的二维数组"

    # 计算每个边界框的面积
    areas = np.zeros((boxes.shape[0],))
    for i, box in enumerate(boxes):
        poly = ShapelyPolygon(box.reshape(4, 2))
        areas[i] = poly.area

    # 对边界框按面积进行排序
    sorted_indices = np.argsort(areas)

    keep = []
    while len(sorted_indices) > 0:
        # 选择面积最大的边界框并保留
        index = sorted_indices[-1]
        keep.append(index)
        if len(sorted_indices) == 1:
            break

        sorted_indices = sorted_indices[:-1]
        rest_boxes = boxes[sorted_indices]

        ious = np.array([iou(boxes[index], rest_box) for rest_box in rest_boxes])
        sorted_indices = sorted_indices[ious < iou_threshold]

    return boxes[keep]

# 性能评估
def evaluate_model(test_loader, model, iou_threshold=0.5):
    model.eval()
    iou_scores = []

    with torch.no_grad():
        for images, _, ground_truth_boxes in test_loader:
            images = images.to(device)
            predicted_score_maps, predicted_geo_maps = model(images)
            # 解码模型输出
            predicted_boxes = decode_predictions(predicted_score_maps, predicted_geo_maps)
            
            # 对于每个图像，计算预测边界框和真实边界框之间的 IoU
            image_ious = []
            for pred_box in predicted_boxes:
                best_iou = 0
                for gt_box in ground_truth_boxes:
                    # 对真实边界框进行格式转换
                    gt_box_np = np.array([[pt.item() for pt in point] for point in gt_box]).reshape(-1, 8)
                    iou_value = iou(pred_box, gt_box_np)  # 调用之前定义的iou函数
                    if iou_value > best_iou:
                        best_iou = iou_value
                if best_iou > iou_threshold:
                    image_ious.append(best_iou)

            if image_ious:
                average_iou = sum(image_ious) / len(image_ious)
                iou_scores.append(average_iou)

    # 计算所有图像的平均 IoU
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    return mean_iou

# 可视化结果
def visualize_results(image, predicted_boxes, ground_truth_boxes):
    plt.imshow(image.permute(1, 2, 0))
    ax = plt.gca()

    # 绘制预测的边界框
    for box in predicted_boxes:
        poly = Polygon(box, edgecolor='r', fill=False, linewidth=1)
        ax.add_patch(poly)

    # 绘制真实的边界框
    # for box in ground_truth_boxes:
    #     # 将Tensor转换为NumPy数组
    #     box = np.array([[tensor.item() for tensor in inner_list] for inner_list in box]).reshape(4, 2)
    #     poly = Polygon(box, edgecolor='g', fill=False, linewidth=1)
    #     ax.add_patch(poly)
    print('预测：',predicted_boxes)
    print('真实：',ground_truth_boxes)
    plt.show()

# 进行评估
mean_iou = evaluate_model(test_loader, model)
print(f"Mean IoU: {mean_iou}")

# 可视化示例图像的结果
for images, _, ground_truth_boxes in test_loader:
    images = images.to(device)
    predicted_score_maps, predicted_geo_maps = model(images)
    print('~~~~',predicted_score_maps)
    print('>>>>',predicted_geo_maps)
    # 解码模型输出
    predicted_boxes = decode_predictions(predicted_score_maps, predicted_geo_maps)
    # 可视化结果
    visualize_results(images.cpu().squeeze(0), predicted_boxes, ground_truth_boxes)

    break  # 只可视化第一个样本
