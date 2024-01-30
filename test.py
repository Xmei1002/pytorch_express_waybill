import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# 创建图像增强转换
transform = transforms.Compose([
    transforms.Resize((1080, 1080)),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.5, contrast=2), 
])

# 加载图像
image_path = r'dataset\train_imgs\type_23\JDAZ09310142227-1-1-.jpg'  # 替换为您的图像路径
original_image = Image.open(image_path).convert('RGB')

# 应用图像增强
enhanced_image = transform(original_image)

# 显示图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
# 将增强后的图像从Tensor转换回PIL Image以便显示
enhanced_image = transforms.ToPILImage()(enhanced_image)
plt.imshow(enhanced_image)
plt.title('Enhanced Image')
plt.axis('off')

plt.show()
