import torch

print(torch.cuda.is_available())  # Output: True jika GPU tersedia
print(torch.cuda.device_count())  # Jumlah GPU yang terdeteksi
print(torch.cuda.get_device_name(0))  # Nama GPU yang digunakan
