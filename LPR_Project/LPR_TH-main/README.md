# LPR_TH

ระบบตรวจจับทะเบียนรถด้วย YOLO และ Flask

## การติดตั้ง

1. ติดตั้ง **Python 3.10** และติดตั้ง **Anaconda** https://www.python.org/downloads/
2. สร้างสภาพแวดล้อมด้วย Conda

```bash
conda create --name ProjectLPR python=3.10
```
```bash
conda create --name ProjectLPR_Train python=3.10
```
```bash
conda activate ProjectLPR
```

3. ติดตั้งไลบรารีที่จำเป็น

```bash
conda install -c conda-forge ultralytics=8.3.3
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-contrib-python shapely
pip install flask opencv-python-headless ultralytics numpy
```

4. ไม่ต้องติดตั้งฐานข้อมูลแยก เนื่องจากโปรเจ็กต์ใช้ **SQLite** ซึ่งมาพร้อมกับ Python
   เมื่อรัน `python LPR_TH/app.py` หรือสคริปต์ในโฟลเดอร์ `scripts` จะมีการสร้างไฟล์
   `vehicle.db` ให้อัตโนมัติ


5. ตรวจสอบเวอร์ชันไลบรารีที่ติดตั้ง

```bash
pip list
```

6. ดูสเปกคอมพิวเตอร์ที่แนะนำได้ที่ [YOLO hardware requirements](https://www.proxpc.com/blogs/system-hardware-requirements-for-yolo-in-2025?utm_source=chatgpt.com)

## การใช้งาน

1. กำหนด path วิดีโอด้วยตัวแปรสภาพแวดล้อม `VIDEO_PATH` หรือส่งพารามิเตอร์ `--video` ขณะรัน `app.py`
2. เริ่มต้นฐานข้อมูลและรันแอปพลิเคชัน (เมื่อรันครั้งแรกจะสร้างไฟล์ `vehicle.db` ให้อัตโนมัติ)

```bash
VIDEO_PATH=/path/to/video.mp4 python LPR_TH/app.py
# หรือ
python LPR_TH/app.py --video /path/to/video.mp4
```

### เติมข้อมูลตัวอย่าง

เพื่อสร้างข้อมูลทะเบียนรถตัวอย่างลงใน `vehicle.db` ให้รัน

```bash
python scripts/populate_mock_data.py
```

### จุดเชื่อมต่อ API

- `POST /register_vehicle` – ลงทะเบียนป้ายทะเบียนใหม่ ต้องระบุ `plate` และสามารถระบุ `province` หรือ `driver_name` ได้
- `GET /vehicles` – รายการป้ายทะเบียนที่ลงทะเบียนแล้ว
- `GET /detections` – ประวัติการตรวจจับจากฐานข้อมูล

## การทดสอบ

ติดตั้ง `pytest` แล้วรันคำสั่ง

```bash
pytest
```
