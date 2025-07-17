import torch
from yolov1 import YoloV1
from yolodset import YoloV1Dataset
from config import YoloV1Loss, plot_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from yolotrain import train_model

model = YoloV1()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dir = "/home/ashiksufaid/summer proj 2/Data_122824/Glioma_MDC_2025_training"
train_dataset = YoloV1Dataset(train_dir, label_map={"Mitosis":1,"Non-mitosis":0}, S=8, B=2, C=2, img_size=(512,512),transforms=None)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle= True)

val_dir = "/home/ashiksufaid/summer proj 2/Data_122824/Glioma_MDC_2025_tester"
val_dataset = YoloV1Dataset(val_dir, label_map={"Mitosis":1,"Non-mitosis":0}, S=8, B=2, C=2, img_size=(512,512),transforms=None)
val_loader = DataLoader(val_dataset, batch_size=4)

criterion = YoloV1Loss(S=8, B=2, C=2, lambda_coord = 5.0, lambda_noobj = 0.5)
optimizer = Adam(model.parameters())
epochs = 100
#os.makedirs("/kaggle/working", exist_ok=True)
save_path = "/home/ashiksufaid/sum proj 2/resnet/Mitosis-Detection/checkpoints/yolov1_weights"
train_loss, val_loss, val_maps = train_model(model, epochs, train_loader, val_loader, criterion, optimizer, device, save_path)

plot_loss(train_loss, val_loss, val_maps, epochs, save_path="/home/ashiksufaid/sum proj 2/resnet/Mitosis-Detection/yolov1_plot")
