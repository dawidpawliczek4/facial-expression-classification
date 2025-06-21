import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

lr =0.001
bs =64


dr= 0.3
ls =0.0
max_epochs = 180

patience=20
TAM_IMG = 48



to_gray=transforms.Grayscale(1)
resize =transforms.Resize((TAM_IMG,TAM_IMG))
to_tensor = transforms.ToTensor()
norm =transforms.Normalize([0.5],[0.5])

tf_train= transforms.Compose([to_gray,transforms.RandAugment(),transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(TAM_IMG, scale=(0.8, 1.0)),to_tensor,norm])



tf_eval = transforms.Compose([to_gray,resize,to_tensor,norm])



def tta_prediction(model, loader, device, n_augment=5):
    model.eval()
    all_preds =[]
    all_targets= []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device).float()
            targets = targets.to(device)

            outputs =[]
            for _ in range(n_augment):
                noise= torch.randn_like(imgs) * 0.01
                noisy_imgs = imgs+ noise
                logits = model(noisy_imgs.float())
                probs =torch.softmax(logits,dim=1)
                outputs.append(probs)

            mean_probs= torch.stack(outputs).mean(dim=0)
            preds =mean_probs.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())
    return accuracy_score(all_targets, all_preds)



class EmotionCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super().__init__()
        self.drop_rate =dropout_rate
        self.num_classes= num_classes

        self.conv_blocks =nn.ModuleList([nn.Identity() for _ in range(6)])
        self.norm_blocks= nn.ModuleList([nn.Identity() for _ in range(6)])
        self.pool_layers = nn.ModuleList([nn.Identity() for _ in range(3)])


        self.flatten= nn.Flatten()
        self.linear1 =None
        self.linear2= None

    def forward(self, x):
        if isinstance(self.conv_blocks[0], nn.Identity):
            

            self.conv_blocks[0] =nn.Conv2d(1, 64, 3, padding=1).to(x.device)
            self.norm_blocks[0]= nn.BatchNorm2d(64).to(x.device)

            self.conv_blocks[1] = nn.Conv2d(64,64, 3,padding=1).to(x.device)
            self.norm_blocks[1] =nn.BatchNorm2d(64).to(x.device)

            self.conv_blocks[2]= nn.Conv2d(64, 128, 3, padding=1).to(x.device)
            self.norm_blocks[2] =nn.BatchNorm2d(128).to(x.device)

            self.conv_blocks[3]= nn.Conv2d(128,128, 3, padding=1).to(x.device)
            self.norm_blocks[3]= nn.BatchNorm2d(128).to(x.device)

            self.conv_blocks[4] =nn.Conv2d(128, 256,3, padding=1).to(x.device)
            self.norm_blocks[4]= nn.BatchNorm2d(256).to(x.device)

            self.conv_blocks[5] =nn.Conv2d(256,256, 3, padding=1).to(x.device)
            self.norm_blocks[5]= nn.BatchNorm2d(256).to(x.device)


            self.pool_layers[0] =nn.MaxPool2d(2).to(x.device)
            self.pool_layers[1]= nn.MaxPool2d(2).to(x.device)
            self.pool_layers[2] =nn.MaxPool2d(2).to(x.device)

            self.linear1 =nn.Linear(256 *6 * 6, 512).to(x.device)
            self.linear2 = nn.Linear(512, self.num_classes).to(x.device)

        for i in range(0, 6,2):
            x= torch.relu(self.norm_blocks[i](self.conv_blocks[i](x)))
            x =torch.relu(self.norm_blocks[i+1](self.conv_blocks[i+1](x)))
            x =self.pool_layers[i//2](x)
            x = nn.functional.dropout(x,self.drop_rate, self.training)

        x = self.flatten(x)
        x= torch.relu(self.linear1(x))
        x = nn.functional.dropout(x, self.drop_rate,self.training)
        out = self.linear2(x)
        return out



def train_once(data_path,output_path, device):

    print(f"using gpu? {device}")
    
    run_name =f"refinado2_lr{lr}_bs{bs}_do{dr}_ls{ls}"


    full_save_path = os.path.join(output_path,run_name)

    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)



    ds_train= datasets.ImageFolder(os.path.join(data_path,"train"), transform=tf_train)
    ds_val=datasets.ImageFolder(os.path.join(data_path, "val"), transform=tf_eval)
    ds_test= datasets.ImageFolder(os.path.join(data_path,"test"), transform=tf_eval)

    dl_train= DataLoader(ds_train, batch_size=bs,shuffle=True,num_workers=2)
    dl_val=DataLoader(ds_val, batch_size=bs,shuffle=False, num_workers=2)
    dl_test= DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=2)

    model =EmotionCNN(7, dr).to(device)
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 1,TAM_IMG, TAM_IMG).to(device)
        model.eval()
        model(dummy_input)
        model.train()


    loss_fn= nn.CrossEntropyLoss(label_smoothing=ls)
    optimizer=optim.Adam(model.parameters(),lr=lr)

    scheduler= optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,steps_per_epoch=len(dl_train), epochs=max_epochs)

    best_val =0.0
    no_improve = 0

    for ep in range(max_epochs):
        model.train()
        for imgs,targets in dl_train:
            imgs = imgs.to(device).float()
            targets =targets.to(device)

            optimizer.zero_grad()
            logits= model(imgs)
            loss =loss_fn(logits,targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        val_preds= []
        val_labels =[]
        with torch.no_grad():
            for vimgs, vlabels in dl_val:
                vimgs = vimgs.to(device).float()
                vlabels = vlabels.to(device)
                vout = model(vimgs)
                val_preds.extend(vout.argmax(1).cpu().tolist())
                val_labels.extend(vlabels.cpu().tolist())

        acc = accuracy_score(val_labels,val_preds)
        print(f"[{run_name}] epoc {ep+1} -  has valAcc: {acc*100:.2f}%")

        if acc > best_val:
            best_val =acc
            no_improve= 0
            torch.save(model.state_dict(), os.path.join(full_save_path,"mejor_modelo.pth"))
        else:
            no_improve =no_improve+ 1
            if no_improve >= patience:
                break

        if (ep + 1) % 40 ==0:
            torch.save(model.state_dict(),os.path.join(full_save_path, f"checkpoint_ep{ep+1}.pth"))

    model.load_state_dict(torch.load(os.path.join(full_save_path, "mejor_modelo.pth")))
    acc_tta = tta_prediction(model,dl_test,device)

    vram_used=torch.cuda.memory_allocated() /1024**2
    vram_peak= torch.cuda.max_memory_allocated() / 1024**2

    with open(os.path.join(full_save_path, "resultados.txt"), "w") as f:
        f.write(f"best val acc: {best_val:.4f}\n")
        f.write(f"TTA test acc:{acc_tta:.4f}\n")
        f.write(f"current VRAM: {vram_used:.2f} MB\n")
        f.write(f"p`eak VRAM:{vram_peak:.2f} MB\n")


if __name__ == "__main__":

    

    base_path =os.path.dirname(os.path.abspath(__file__))
    data_path= os.path.join(base_path,"EntrenarCNN", "data")
    save_path =os.path.join(base_path, "EntrenarCNN","refinados2")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dispo =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_once(data_path, save_path, dispo)
