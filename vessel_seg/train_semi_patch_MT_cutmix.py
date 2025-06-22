import argparse
import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.config import get_config
from model.dice import *
from model.swin import *
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset.dataset_semi import *
# from dataset.dataset import *
from model.utils import *
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")

# set environment parser
Parser = argparse.ArgumentParser()
Parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size")
Parser.add_argument("-s", "--save_path", default="runs_Kvasir-SEG0.1", type=str, help="save path")
Parser.add_argument("-w", "--num_workers", default=16, type=int, help="number of workers")
############
Parser.add_argument(
    '--cfg', type=str, default="model/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
Parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
Parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
Parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
Parser.add_argument('--resume', help='resume from checkpoint')
Parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
Parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
Parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
Parser.add_argument('--tag', help='tag of experiment')
Parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
Parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

args = Parser.parse_args()
config = get_config(args)

def sigmoid_rampup(current, rampup_length):
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
def get_current_consistency_weight(consistency,epoch,consistency_rampup):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def update_ema(model, ema_model, best_pred_idx, alpha, global_step):
    # 动态调整 alpha
    alpha = min(1 - 1 / (global_step + 1), alpha)

    # 根据 best_pred_idx 确定更新策略
    if best_pred_idx == 0:   
        for (teacher_param, student_param) in zip(ema_model.encoder2.parameters(), model.encoder1.parameters()):
           teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)
        for (teacher_param, student_param) in zip(ema_model.decoder2.parameters(), model.decoder1.parameters()):
            teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)
    elif best_pred_idx == 1:
        # 如果教师模型的E2D1效果较好，用学生网络 D1 更新教师网络的 D2
        for (teacher_param, student_param) in zip(ema_model.decoder2.parameters(), model.decoder1.parameters()):
            teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)
    elif best_pred_idx == 2:
        # 如果教师模型的E1D2效果较好，用学生网络 E1 更新教师网络的 E2
        for (teacher_param, student_param) in zip(ema_model.encoder2.parameters(), model.encoder1.parameters()):
            teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)
def rand_bbox(size, lam):
    """
    Generate random bounding box.
    
    Args:
        size (tuple): The size of the input image tensor (B, C, H, W).
        lam (float): Lambda value from the beta distribution.

    Returns:
        Tuple[int, int, int, int]: Coordinates of the bounding box (bbx1, bby1, bbx2, bby2).
    """
    W = size[2]  # Image width
    H = size[3]  # Image height
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniformly sample the center of the bounding box
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Calculate the bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(images_labeled, labels_labeled, images_unlabeled, labels_unlabeled):
    lam = np.random.beta(1.0, 1.0)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images_labeled.size(), lam)
    mixed_images = images_labeled.clone()
    mixed_labels = labels_labeled.clone()

    mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images_unlabeled[:, :, bbx1:bbx2, bby1:bby2]
    mixed_labels[:, bbx1:bbx2, bby1:bby2] = labels_unlabeled[:, bbx1:bbx2, bby1:bby2]

    return mixed_images, mixed_labels

def train():
    base_lr = 0.0001
    num_classes = 2
    image_size = (512,512)
    patch_size = 128
    overlap = 64
    labeled_bs = int(args.batch_size/2)
    labeled_slice = 10 *((image_size[0]-overlap)//(patch_size-overlap) * (image_size[1]-overlap)//(patch_size-overlap))
    # labeled_slice = 200
    base_dir = './data/drive/'
    dataset = 'drive'    
    max_epoch = 110
    lambda_u = 0.1

    model_arch = 'cnn'
    def create_model(ema=False,is_student=True): 
        models_dict = {'cnn': MT_Model_CNN(num_classes=num_classes,is_student=is_student)}
        model = models_dict[model_arch]
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model(is_student=True)
    model.cuda()

        # 在这里添加：统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters   : {trainable_params/1e6:.2f}M")

    ema_model = create_model(is_student=False,ema=True)
    ema_model.cuda()



    db_train = DriveDataset_patch_augment(base_dir+'train/', 'train.txt',image_size,patch_size, overlap, dataset, transform=transforms.Compose([RandomGenerator1()]))
    db_valid = DriveDataset_patch_augment(base_dir+'test/', 'test.txt',image_size,patch_size, overlap, dataset, transform=transforms.Compose([RandomGenerator1()]))
    # db_train = vessel_BaseDataSets(base_dir+'train/', 'train.txt',image_size, dataset, transform=transforms.Compose([RandomGenerator()]))
    # db_valid = vessel_BaseDataSets(base_dir+'test/', 'test.txt',image_size, dataset, transform=transforms.Compose([RandomGenerator()]))
    # 计算图片数量和总patch数量
    total_images = len(db_train.sample_list)  # 训练图片总数
    total_patches = len(db_train)  # 总patch数

    # 打印图片和patch信息
    print("Total patches: {}, labeled patches: {}".format(total_patches, labeled_slice))  # 这里用图片数量乘以每张图的patch数

    labeled_idxs = list(range(0, labeled_slice )) 
    unlabeled_idxs = list(range(labeled_slice, total_patches))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)
    
    train_loader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=0, pin_memory=True)
    valid_loader = DataLoader(db_valid, batch_size=args.batch_size, shuffle=False,num_workers=0)

    print('train len:', len(train_loader))

    optimizer = optim.Adam(model.parameters(), betas=(0.9,0.99), lr=base_lr, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_iterations =  max_epoch * len(train_loader)

    for epoch_num in range(max_epoch):
        train_acc = 0
        train_loss = 0
        test_acc = 0

        print('Epoch: {} / {} '.format(epoch_num, max_epoch))
        for i_batch, sampled_batch in enumerate(train_loader):
            # 提取数据并分配给教师和学生模型
            weak_images, strong_images, weak_labels, strong_labels = (
                sampled_batch['weak_image'],
                sampled_batch['weak_image'],
                sampled_batch['weak_mask'],
                sampled_batch['weak_mask']
            )

            weak_images, strong_images, weak_labels, strong_labels = (
                weak_images.cuda(),
                strong_images.cuda(),
                weak_labels.cuda(),
                strong_labels.cuda()
            )

            model.train()
            for param in model.parameters():
                param.requires_grad = True
            for param in ema_model.parameters():
                param.requires_grad = False   

            outputs = model(strong_images)#学生模型处理 强增强数据
            outputs_soft = torch.softmax(outputs, dim=1)#概率图
            #学生模型有监督损失
            supervised_loss = ce_loss(outputs[:labeled_bs],strong_labels[:labeled_bs].long())#对有标注的数据，计算交叉熵损失，作为监督损失

            ema_output = ema_model(weak_images)#教师模型处理强增强数据，教师模型输出[a,b,c]
            ema_output_soft1 = torch.softmax(ema_output[0], dim=1)
            ema_output_soft2 = torch.softmax(ema_output[1], dim=1)
            ema_output_soft3 = torch.softmax(ema_output[2], dim=1)
            # 计算教师模型每个有标签数据的预测与真实标签的交叉熵损失
            loss1 = ce_loss(ema_output[0][:labeled_bs], weak_labels[:labeled_bs].long())  # 第一预测的损失
            loss2 = ce_loss(ema_output[1][:labeled_bs], weak_labels[:labeled_bs].long())  # 第二预测的损失
            loss3 = ce_loss(ema_output[2][:labeled_bs], weak_labels[:labeled_bs].long())  # 第三预测的损失
            
            # 计算所有预测的平均交叉熵损失
            mean_loss1 = loss1.mean()
            mean_loss2 = loss2.mean()
            mean_loss3 = loss3.mean()

            # 将损失值存储在一个列表中
            losses = torch.tensor([mean_loss1, mean_loss2, mean_loss3])

            # 选择最小损失对应的预测
            best_pred_idx = torch.argmin(losses)

            # 根据选择的预测索引，从 ema_output 中选取最接近的预测作为伪标签
            if best_pred_idx == 0:
                ema_output_soft = ema_output_soft1[labeled_bs:]

            elif best_pred_idx == 1:
                ema_output_soft = ema_output_soft2[labeled_bs:]

            else:
                ema_output_soft = ema_output_soft3[labeled_bs:]

            pseudo_label = torch.max(ema_output_soft, dim=1)[1]  # 获取伪标签

            if iter_num < 200:
                consistency_loss = 0.0
                unsupervised_loss = 0.0      
            else:
                consistency_loss = torch.mean((outputs_soft[labeled_bs:]-ema_output_soft)**2)#学生模型一致性损失
                unsupervised_loss = ce_loss(outputs[labeled_bs:], pseudo_label.long())#学生模型无监督损失
            
            # mixed_images, mixed_labels = cutmix_data(
            #         strong_images[:labeled_bs],
            #         strong_labels[:labeled_bs],
            #         strong_images[labeled_bs:],
            #         pseudo_label
            #     )
            # mixed_outputs = model(mixed_images)
            # import ipdb; ipdb.set_trace()
            # mixed_loss = ce_loss(mixed_outputs, mixed_labels)
            consistency_weight = get_current_consistency_weight(lambda_u,epoch_num,max_epoch)
            # 学生模型最终损失
            loss = supervised_loss + consistency_weight*(consistency_loss + unsupervised_loss )
            #计算准确率
            prediction = torch.max(outputs[:labeled_bs], dim=1)[1]#从模型的输出 outputs 中选择预测概率最大的类别（即预测的类别标签）
            #计算预测结果与真实标签的相似度
            train_correct = (prediction == weak_labels[:labeled_bs]).float().mean().cpu().numpy()
            # import ipdb; ipdb.set_trace()
            train_acc = train_acc + train_correct#累计当前批次的准确率到总的训练准确率 train_acc 中

            train_loss = train_loss + loss.detach().cpu().numpy()#累计当前批次的损失值到总的训练损失 train_loss 中
            # 更新模型参数和权重参数
            optimizer.zero_grad()#学生模型梯度清零
            loss.backward()
            optimizer.step()#更新学生模型参数
            lr = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
#########通过学生模型参数EMA更新教师模型参数          
            update_ema(model, ema_model, best_pred_idx, alpha=0.99, global_step=iter_num)

            iter_num += 1
        
###test
        model.eval()
        for i_batch, sampled_batch in enumerate(valid_loader):
            images, labels = sampled_batch['weak_image'], sampled_batch['weak_mask']
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(images)
                outputs_soft = torch.softmax(outputs, dim=1)
                prediction = torch.max(outputs,1)[1]
                test_correct = (prediction == labels).float().mean().cpu().numpy()
                test_acc = test_acc + test_correct
        print('train_loss: ',train_loss/(labeled_slice/labeled_bs),' train_acc: ',train_acc/(labeled_slice/labeled_bs),'test_acc: ',test_acc/len(valid_loader)) 
        model.train()
        if epoch_num>=30 and epoch_num % 2 == 0:
            torch.save(model.state_dict(), './runs5/'+str(dataset)+'/student_'+str(epoch_num)+'.pth') 


train()
