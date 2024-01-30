# models/detection_model.py
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_detection_model(num_classes):
    # 加载预训练的 Faster R-CNN 模型
    model = models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替换模型的头部以适应新的类别数
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 打印模型的一些信息
    # print("Model loaded with Faster R-CNN ResNet50 FPN")
    # print(f"Number of input features for ROI classifier: {in_features}")
    # print(f"Number of classes (including background): {num_classes}")

    return model

# 测试函数
if __name__ == "__main__":
    num_classes = 4
    model = get_detection_model(num_classes)
    print(model)