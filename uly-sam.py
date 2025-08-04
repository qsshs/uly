from ultralytics import FastSAM

# Load a model
model_XL = FastSAM("FastSAM-x.pt")

results = model_XL.predict(source=r"C:\Users\QY\Desktop\mine\天窗\蘑菇扣\OK\13-31-55.png", conf = 0.5, imgsz=(2048, 2048))

results[0].show()