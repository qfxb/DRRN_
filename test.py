import os
import torch
import scipy.io as scio
from STED.DRRN import DRRN
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ===================== 加载模型 =====================
model = DRRN()
model.to(device)

weights_path = r"C:\Users\PC\Desktop\DRRN\model_train\sanshe.tar"

checkpoint = torch.load(weights_path, weights_only=False, map_location=device)

model.load_state_dict(checkpoint['state_dict'])
model.eval()

# ===================== 数据文件夹 =====================
dir1_base = r'C:\Users\PC\Desktop\DRRN\data\Dounut_test'
dir2_base = r'C:\Users\PC\Desktop\DRRN\data\Gaussian_test'
dir3_base = r'C:\Users\PC\Desktop\DRRN\data\Fusion_test'

savedata_root = r'C:\Users\PC\Desktop\DRRN\model_train\test'

os.makedirs(savedata_root, exist_ok=True)


def load_and_preprocess_image(image_path):
    image_numpy = cv2.imread(image_path,flags=cv2.IMREAD_GRAYSCALE)

    image_tensor = image_numpy[None, None, :, :]
    image_tensor = torch.FloatTensor(image_tensor)
    image_tensor = image_tensor.to(device)

    return image_tensor, image_numpy


# ===================== 指定读取的编号 =====================
data_id = 1608

dounut_name = f'Dounut_{data_id}.tif'
gaussian_name = f'Gaussian_{data_id}.tif'
fusion_name = f'Fusion_{data_id}.tif'

dounut_path = os.path.join(dir1_base,dounut_name)
gaussian_path = os.path.join(dir2_base,gaussian_name)
fusion_path = os.path.join(dir3_base,fusion_name)

print('读取Dounut：', dounut_path)
print('读取Gaussian：', gaussian_path)
print('读取Fusion：', fusion_path)

# ===================== 读取两张图 =====================
input1_image, dounut_numpy = load_and_preprocess_image(dounut_path)
input2_image, gaussian_numpy = load_and_preprocess_image(gaussian_path)
fusion_tensor, fusion_numpy = load_and_preprocess_image(fusion_path)

# ===================== 按照你原代码的方式测试 =====================
with torch.no_grad():
    predictions = model(input1_image, input1_image)

print('output_shape：', predictions.shape)

predictions = predictions.cpu().numpy()

n = predictions.shape[2]
m = predictions.shape[3]

predictions = predictions.reshape(n, m)

# ===================== 保存三张图 =====================
save_name = f'fusion_{data_id}.mat'
save_path = os.path.join(savedata_root,save_name)

scio.savemat(
    save_path,
    {
        'Output': predictions,
        'Gaussian': gaussian_numpy,
        'Dounut': dounut_numpy,
        'Fusion': fusion_numpy
    }
)

print('保存完成：', save_path)