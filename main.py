import uvicorn
from fastapi import FastAPI
import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import io



test = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Test(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 7 * 7, 256),
        nn.ReLU(),
        nn.Linear(256, 6)
    )

  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

class_name = ['cardboard',
              'glass',
              'metal',
              'paper',
              'plastic',
              'trash'
              ]

model = Test()
model.load_state_dict(torch.load('test1.pth', map_location=device))

model.to(device)

st.title('EXAM')
st.text('загрузи изображение')
file = st.file_uploader('загрузи фото ', type=['jpg', 'jpeg', 'svg'])
if not file:
    st.info('Загрузи правилный формат')

if st.button('Запустить'):

    image_data = file.read()
    img = Image.open(io.BytesIO(image_data))

    with torch.no_grad():





if __name__ == '__main__':
    uvicorn.run(test, host='127.0.0.0', port=8000)