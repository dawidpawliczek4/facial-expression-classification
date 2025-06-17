# Version 1.5
# UPGRADES From last version

#fake generatos was doing 96x96 images instead 48x48
#a lot of changes to make compatible with my own computer
import os ,torch
import torch.nn  as nn
import torch.optim  as optim
from torchvision  import datasets , transforms
from torchvision.utils import save_image
from torch.utils.data  import DataLoader


import time




ruta =os.path.dirname(os.path.abspath(__file__)) 

ruta_guardado = os.path.join(ruta ,"generadas_cgan")

if not os.path.exists(ruta_guardado  ):
 os.makedirs(ruta_guardado )



dispo = torch.device( "cuda" if torch.cuda.is_available( ) else "cpu")




TAM_IMG = 48
z_dim =128
n_etiquetas =7
bs= 64
epocas = 200


to_gray=transforms.Grayscale(1)
resize =transforms.Resize((TAM_IMG ,TAM_IMG ))
to_tensor = transforms.ToTensor()
norm = transforms.Normalize([0.5] ,[0.5])

data_tf = transforms.Compose([to_gray ,resize ,to_tensor ,norm])


ruta_data =os.path.join(ruta,"train")






class Generador(nn.Module):
 def __init__(self):
  nn.Module.__init__(self)
  self.etq= nn.Embedding(n_etiquetas, z_dim)
  self.etq_extra = nn.Linear(z_dim, z_dim)
  self.proyector = nn.Linear(z_dim * 2, 512 * 3 * 3)
  self.bn1 = nn.BatchNorm2d(256)
  self.bn2 = nn.BatchNorm2d(128)
  self.bn3 = nn.BatchNorm2d(64)
  self.conv_layers = [None, None, None, None]

  self.dropout = nn.Dropout(0.3)
  

 def forward(self,z,y):
  if z.shape[1] !=z_dim:
    print("Algo raro pasa con z_dim",z.shape)

  y_emb = self.etq(y)
  y2 =torch.relu(self.etq_extra(y_emb))
  x = torch.cat([z, y2 * 2.0], dim=1)
  x = self.proyector(x).view(-1,512,3, 3)


  if self.conv_layers[0] is None:
      self.conv_layers[0]=nn.ConvTranspose2d(512, 256, 4, 2, 1).to(x.device)  # 3x3 → 6x6
      self.conv_layers[1] =nn.ConvTranspose2d(256, 128, 4, 2, 1).to(x.device)  # 6x6 → 12x12
      self.conv_layers[2]= nn.ConvTranspose2d(128, 64, 4, 2, 1).to(x.device)   # 12x12 → 24x24
      self.conv_layers[3] =nn.ConvTranspose2d(64, 1, 4, 2, 1).to(x.device)     # 24x24 → 48x48

  x =torch.relu(self.bn1(self.conv_layers[0](x)))
  x= torch.relu(self.bn2(self.conv_layers[1](x)))
  x = torch.relu(self.bn3(self.conv_layers[2](x)))


  x = torch.tanh(self.conv_layers[3](x))
  return x


class Discriminador(nn.Module):
 def __init__(self):
  nn.Module.__init__(self)
  self.etq= nn.Embedding(n_etiquetas, TAM_IMG * TAM_IMG)
  self.conv1 = nn.utils.spectral_norm(nn.Conv2d(2, 256,4,2,1))
  self.ln1 = nn.LayerNorm([256, 24, 24])
  self.act1 = nn.LeakyReLU(0.2)
  self.conv2 = nn.utils.spectral_norm(nn.Conv2d(256,512, 4, 2,1))
  self.ln2 = nn.LayerNorm([512, 12, 12])
  self.act2= nn.LeakyReLU(0.2)
  self.conv3 = nn.utils.spectral_norm(nn.Conv2d(512,1024, 4,2, 1))
  self.ln3 = nn.LayerNorm([1024, 6, 6])
  self.act3=nn.LeakyReLU(0.2)
  self.final = nn.utils.spectral_norm(nn.Conv2d(1024, 1,6))

 def forward(self, x, etiqueta):
  if x.shape[2:] != (TAM_IMG, TAM_IMG):
   x =nn.functional.interpolate(x, size=(TAM_IMG, TAM_IMG), mode='bilinear')

  ymap=self.etq(etiqueta).view(-1, 1, TAM_IMG, TAM_IMG)
  inp =torch.cat([x, ymap],dim=1)

  x= self.act1(self.ln1(self.conv1(inp)))
  x= self.act2(self.ln2(self.conv2(x)))
  x =self.act3(self.ln3(self.conv3(x)))

  x =self.final(x)

  return x.view(-1)







def compute_grad_penalty(D, reales,falsas, clases):
  bsz = reales.size(0)
  alpha = torch.rand(bsz, 1, 1,1, device=reales.device,dtype=reales.dtype)
  mezcladas= alpha * reales + (1 -alpha) *falsas

  

  decision = D(mezcladas,clases)

  grad_out = torch.ones_like(decision)

  grad = torch.autograd.grad(
      outputs=decision,
      inputs=mezcladas,
      grad_outputs=grad_out,
      create_graph=True,
      retain_graph=False,
      only_inputs=True
  )[0]

  grad =grad.view(bsz, -1)

  norma= grad.norm(2, dim=1)


  penalizacion =((norma- 1) **2).mean()
  return penalizacion






ruta_modelos =os.path.join(ruta, "modelos")
os.makedirs(ruta_modelos, exist_ok=True)




#completelly remodelate to work on my pc

if __name__ == "__main__":
    print("Models has been iniciated")

    G =Generador().to(dispo)
    D= Discriminador().to(dispo)

    opt_G =optim.Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))


    opt_D= optim.RMSprop(D.parameters(),lr=0.0001)


    criterio =nn.BCEWithLogitsLoss()

    trainset = datasets.ImageFolder(ruta_data,transform=data_tf )
    loader= DataLoader(trainset,batch_size=bs,shuffle=True, num_workers=0,pin_memory=True)

    print(f"training is going to start total epoch: {epocas}")

    for ep in range(epocas):
        t0 = time.time()
        print(f" empezando época {ep+1}")
        print(f"el total  de batches por epoca: {len(loader)}")
        for idx, (reales, clases) in enumerate(loader):
            if ep == 0 and idx ==0:
                print("el dispositivo de reales:",reales.device)
                reales= reales.to(dispo)
                clases =clases.to(dispo)
                print("Dispositivo de fakez_d:", G(torch.randn(1, z_dim,device=dispo), clases[:1]).device)
            else:
                reales= reales.to(dispo)
                clases =clases.to(dispo)

                
            tam = reales.size(0)

            unos=torch.full((tam,), 0.9, device=dispo)

            ceros = torch.zeros((tam,),device=dispo)

            z_d= torch.randn(tam, z_dim, device=dispo)

            with torch.no_grad():
                fakez_d= G(z_d, clases)

            noise_strength =0.05 * (1 -ep /epocas)

            reales_gp=reales.detach().clone() +torch.randn_like(reales,device=dispo) * noise_strength
            falsas_gp = fakez_d.detach().clone() + torch.randn_like(fakez_d)* noise_strength

            reales_gp.requires_grad_(True)
            falsas_gp.requires_grad_(True)

            gp =compute_grad_penalty(D, reales_gp,falsas_gp, clases)

            reales_noisy = reales + torch.randn_like(reales) * noise_strength


            fakez_noisy = fakez_d +torch.randn_like(fakez_d) * noise_strength

            out_real =D(reales_noisy, clases)
            out_fake = D(fakez_noisy,clases)

            lossD = criterio(out_real,unos) + criterio(out_fake, ceros) + 10.0*gp.detach()

            opt_D.zero_grad()
            lossD.backward()
            opt_D.step()

            del fakez_d, gp, out_real,out_fake, reales_noisy,fakez_noisy

            if idx % 1== 0:
                z_g = torch.randn(tam, z_dim,device=dispo)
                fakez_g =G(z_g, clases)

                out_fake_2 =D(fakez_g, clases)
                unos_g =torch.full((tam,),0.9,device=dispo)
                lossG= criterio(out_fake_2,unos_g)

                opt_G.zero_grad()
                lossG.backward()
                opt_G.step()

                del fakez_g,out_fake_2
                torch.cuda.empty_cache()

            if idx % 50 == 0:
                print(f"Epoc {ep+1} | batch {idx} | LD {lossD.item():.5f} | LG {lossG.item():.5f}")

        if (ep + 1) % 5 == 0:
            t1 =time.time()
            print(f"time for the epoch {ep+1}: {t1 - t0:.2f} seconds")#this is not workig

            with torch.no_grad():
                test_z =torch.randn(n_etiquetas, z_dim, device=dispo)
                test_y= torch.arange(n_etiquetas, device=dispo)
                muestras =G(test_z,test_y)
                save_image(muestras, f"{ruta_guardado}/ep_{ep+1}.png",nrow=7, normalize=True)
                del muestras
                print(f"saved files from epoch {ep+1}")

            if (ep + 1) % 10 == 0:
                torch.save(G.state_dict(), os.path.join(ruta_modelos, f"gen_ep{ep+1}.pth"))
                torch.save(D.state_dict(), os.path.join(ruta_modelos, f"disc_ep{ep+1}.pth"))
                print(f"Checkpoint {ep+1}")

    torch.save(G.state_dict(),os.path.join(ruta, "modelos","generador_final_cgan_human.pth"))
    torch.save(D.state_dict(), os.path.join(ruta, "modelos","discriminador_final_cgan_human.pth"))

    print("all done, not crash so")
