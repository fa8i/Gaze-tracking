import torch

print(torch.cuda.is_available())

import os
import pandas as pd
import h5py
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import models
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
print(pl.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

torch.set_float32_matmul_precision('medium')

vgg16_weights = models.VGG16_Weights.IMAGENET1K_V1


class GazeDataset(Dataset):
    def __init__(self, root_dir, h5_file_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Cargar labels desde el archivo .h5
        with h5py.File(h5_file_path, 'r') as hdf_file:
            file_name_base = hdf_file['file_name_base'][:]
            gaze_pitch = hdf_file['gaze_pitch'][:]
            gaze_yaw = hdf_file['gaze_yaw'][:]

        # Convertir a DataFrame para facilitar el manejo
        self.labels_df = pd.DataFrame({
            'file_name_base': [name.decode('utf-8') for name in file_name_base],
            'gaze_pitch': gaze_pitch,
            'gaze_yaw': gaze_yaw
        })

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Obtener el nombre base del archivo, que incluye paciente y día
        base_name = self.labels_df.iloc[idx]['file_name_base']  # Ejemplo: 'p00/day01/0005'

        # Construir las rutas a las imágenes
        images = []
        img_suffixes = ['-full_face.png', '-right_eye.png', '-left_eye.png']

        for img_suffix in img_suffixes:
            img_name = base_name + img_suffix  # Ejemplo: 'p00/day01/0005-full_face.png'
            img_path = os.path.join(self.root_dir, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            else:
                raise FileNotFoundError(f'No se encontró la imagen: {img_path}')

        # Obtener los labels
        gaze_pitch = self.labels_df.iloc[idx]['gaze_pitch']
        gaze_yaw = self.labels_df.iloc[idx]['gaze_yaw']
        labels = torch.tensor([gaze_pitch, gaze_yaw], dtype=torch.float32)

                # Obtener person_idx a partir de 'file_name_base'
        base_name = self.labels_df.iloc[idx]['file_name_base']  # Ejemplo: 'p00/day01/0005'
        person_folder = base_name.split('/')[0]  # 'p00'
        person_idx = int(person_folder[1:])  # Convertir 'p00' a 0

        # Convertir imágenes a tensores
        full_face = images[0]
        right_eye = images[1]
        left_eye = images[2]

        return (person_idx, full_face, right_eye, left_eye), labels


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer

    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(  # Excitation (similar to attention)
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Model(LightningModule):
    """
    Base model from https://github.com/pperle/gaze-tracking
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.subject_biases = nn.Parameter(torch.zeros(15 * 2, 2))  # pitch and yaw offset for the original and mirrored participant

        self.cnn_face = nn.Sequential(
            models.vgg16(weights=vgg16_weights).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.cnn_eye = nn.Sequential(
            models.vgg16(weights=vgg16_weights).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.cnn_eye2fc = nn.Sequential(
            SELayer(256),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            SELayer(256),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            SELayer(128),
        )

        self.fc_eye = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 6 * 128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc_eyes_face = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )


        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()

    def training_step(self, batch, batch_idx):
        (person_idx, full_face, right_eye, left_eye), labels = batch
        outputs = self.forward(person_idx, full_face, right_eye, left_eye)
        loss = nn.MSELoss()(outputs, labels)

        # Update and log training metrics
        self.train_mse.update(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mse', self.train_mse, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # Reset training metrics
        self.train_mse.reset()

    def validation_step(self, batch, batch_idx):
        (person_idx, full_face, right_eye, left_eye), labels = batch
        outputs = self.forward(person_idx, full_face, right_eye, left_eye)
        loss = nn.MSELoss()(outputs, labels)

        # Update validation metrics
        self.val_mse.update(outputs, labels)
        self.val_mae.update(outputs, labels)

        # Log validation loss per epoch
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self):
        # Log validation metrics at the end of the epoch
        self.log('val_mse', self.val_mse.compute(), prog_bar=True)
        self.log('val_mae', self.val_mae.compute(), prog_bar=True)

        # Reset validation metrics
        self.val_mse.reset()
        self.val_mae.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, person_idx: torch.Tensor, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
        out_cnn_face = self.cnn_face(full_face)
        out_fc_face = self.fc_face(out_cnn_face)

        out_cnn_right_eye = self.cnn_eye(right_eye)
        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

        return t_hat + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# Definir transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Crear instancias del Dataset y DataLoader
train_dataset = GazeDataset(
    root_dir='/home/fabian/Escritorio/MPIIFaceGaze_preprocessed',  # Reemplaza con tu ruta
    h5_file_path='/home/fabian/Escritorio/MPIIFaceGaze_preprocessed/data.h5',
    transform=transform
)

# Obtener los índices de los datos
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Crear Subsets para entrenamiento y validación
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

# Crear DataLoaders para entrenamiento y validación
train_loader = DataLoader(
    train_subset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Instanciar el modelo
model = Model()

# Logger de TensorBoard
logger = TensorBoardLogger('logs/', name='gaze_model')

# Callback para guardar el mejor modelo
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='gaze-model-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min',
)

# Configurar el entrenador
trainer = Trainer(
    max_epochs=10,
    accelerator='gpu',
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=10,
)

# Entrenar el modelo
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Cargar el mejor modelo guardado
best_model = Model.load_from_checkpoint(checkpoint_callback.best_model_path)

# Evaluar en el conjunto de validación
trainer.validate(best_model, val_loader)


