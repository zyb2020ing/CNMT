import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from model.swin import *
from dataset.dataset_semi import *
from model.utils import Evaluator

# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
num_classes = 2
batch_size = 1
image_size = (512, 512)  # 原始图像大小
patch_size = 128    
overlap = 64
stride = patch_size - overlap
save_dir = '/home/amax/zybin/Code/Vessel-Seg-main/result5/drive/'
base_dir = '/home/amax/zybin/Code/Vessel-Seg-main/data/drive/test/'
dataset = 'drive'

# 创建验证集 DataLoader
db_val = testBaseDataSets(base_dir, 'test.txt', image_size, dataset=dataset, transform=transforms.Compose([RandomGenerator()]))
valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=0)

# 定义模型
model_name = 'student_'  # 模型前缀名称
model = MT_Model_CNN(is_student=True,num_classes= num_classes)

# 将每张图像分割为多个 patch
def extract_ordered_overlap(image, patch_size, stride):
    image_h, image_w = image.shape[2], image.shape[3]
    N_patches_img = ((image_h - patch_size) // stride + 1) * ((image_w - patch_size) // stride + 1)
    patches = torch.empty((N_patches_img, 3, patch_size, patch_size)).to(image.device)

    iter_tot = 0
    for h in range((image_h - patch_size) // stride + 1):
        for w in range((image_w - patch_size) // stride + 1):
            patch = image[0, :, h * stride:(h * stride) + patch_size, w * stride:(w * stride) + patch_size]
            patches[iter_tot] = patch
            iter_tot += 1
    return patches

# 重新组合 patch 为完整图像
def recompone_overlap(preds, img_h, img_w, stride):
    patch_h, patch_w = preds.shape[1], preds.shape[2]
    full_prob = torch.zeros((img_h, img_w)).cuda()
    full_sum = torch.zeros((img_h, img_w)).cuda()
    k = 0

    for h in range((img_h - patch_h) // stride + 1):
        for w in range((img_w - patch_w) // stride + 1):
            full_prob[h * stride:(h * stride) + patch_h, w * stride:(w * stride) + patch_w] += preds[k]
            full_sum[h * stride:(h * stride) + patch_h, w * stride:(w * stride) + patch_w] += 1
            k += 1

    return full_prob / full_sum

# 遍历模型权重文件并进行测试
for k in range(30, 110, 2):
    print(f'/home/amax/zybin/Code/Vessel-Seg-main/runs5/{dataset}/{model_name}{k}.pth')
    model.load_state_dict(torch.load(f'/home/amax/zybin/Code/Vessel-Seg-main/runs5/{dataset}/{model_name}{k}.pth'))
    model.cuda()
    model.eval()
    evaluator = Evaluator()
    j = 0

    with torch.no_grad():
        for sampled_batch in valloader:
            images, labels = sampled_batch['image'], sampled_batch['label']
            images, labels = images.cuda(), labels.cuda()

            # 提取 patch 并进行预测
            patches = extract_ordered_overlap(images, patch_size, stride)
            predictions = model(patches.cuda())
            # import ipdb; ipdb.set_trace()
            predictions = torch.softmax(predictions, dim=1)
            pred = predictions[:, 1, :, :]  # 取第二个通道

            # 重新组合为完整图像
            pred_recomposed = recompone_overlap(pred, image_size[0], image_size[1], stride)

            # 更新评估器并保存结果
            evaluator.update(pred_recomposed, labels[0, :, :].float())

            for i in range(batch_size):
                # pred_recomposed_np = ((pred_recomposed).cpu().numpy() * 255).astype(np.uint8)
                pred_recomposed_np = pred_recomposed.cpu().numpy()   #dtype('float32')
                # 保存图片
                cv2.imwrite(f'{save_dir}Pred_MT_CNN{k}.jpg', pred_recomposed_np[:,:]*255)
                j+=1

    evaluator.show()  # 输出评估结果