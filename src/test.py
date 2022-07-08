from sahi.model import Yolov5DetectionModel

from azure.storage.blob import BlobServiceClient

STORAGEACCOUNTURL = "https://testws2273653885.blob.core.windows.net"
STORAGEACCOUNTKEY = "2TQq7sVVNaGld4i2jmMzuXbuxh5ILOcF42Vu9CjZRAuiWDsAQB/awXFTDOplc6fZCiuL9TMiOhpj+AStwQmJkA=="
CONTAINERNAME = "azureml"
BLOBNAME = "best299.pt"

blob_service_client_instance = BlobServiceClient(
    account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)

blob_client_instance = blob_service_client_instance.get_blob_client(
    CONTAINERNAME, BLOBNAME, snapshot=None)

blob_data = blob_client_instance.download_blob()
image_path = "./test.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

detection_model = Yolov5DetectionModel(model_path="best299.pt",confidence_threshold=0.8,device="cpu")
result = get_sliced_prediction(image,detection_model,slice_height = 416,slice_width = 416, overlap_height_ratio = 0.6, overlap_width_ratio = 0.6)
result.export_visuals(export_dir="demo_data_src/")
