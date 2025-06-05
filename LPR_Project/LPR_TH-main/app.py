# นำเข้าไลบรารีที่จำเป็น
from flask import Flask, Response, jsonify, render_template, request
import cv2
from ultralytics import YOLO
from function.helper import get_thai_character, data_province, split_license_plate_and_province
from function.database import (
    init_db,
    register_vehicle,
    list_vehicles,
    is_registered,
    log_detection,
    list_detections,
)
import threading 
import time 
import numpy as np 
from datetime import datetime 
import base64 

# โหลดโมเดล YOLO
vehicle_model = YOLO("model/license_plate.pt") 
plate_model = YOLO("model/data_plate.pt")

class LicensePlateDetector:  # คลาสสำหรับจัดการการตรวจจับป้ายทะเบียน
    def __init__(self, db_conn=None):
        self.db_conn = db_conn
        self.is_running = False
        self.video_source = None
        self.roi_y_start = 0.6 
        self.roi_y_end = 0.75 
        self.last_detection_time = 0
        self.detection_cooldown = 1.0 
        self.detection_log = [] 
        self.frame = None 
        self.lock = threading.Lock() 

    def get_vehicle_snapshot(self, frame, x1, y1, x2, y2): # เมธอดสำหรับจับภาพยานพาหนะจาก bounding box ที่ได้
        height, width = frame.shape[:2] 
        expand_px = 20  
        
        # คำนวณพิกัดใหม่ของ bounding box ที่ขยายแล้ว โดยตรวจสอบไม่ให้เกินขอบเขตของเฟรม
        x1 = max(0, x1 - expand_px)
        y1 = max(0, y1 - expand_px) 
        x2 = min(width, x2 + expand_px)
        y2 = min(height, y2 + expand_px) 
        
        # ครอบตัด (crop) บริเวณยานพาหนะจากเฟรมเดิม
        vehicle_img = frame[y1:y2, x1:x2]
        
        # แปลงรูปภาพที่ครอบตัดแล้วเป็น Base64 string
        _, buffer = cv2.imencode('.jpg', vehicle_img) 
        img_base64 = base64.b64encode(buffer).decode('utf-8') 
        return img_base64 

    def is_same_plate(self, plate1, plate2): # เมธอดสำหรับเปรียบเทียบป้ายทะเบียนสองป้ายว่าเป็นของรถคันเดียวกันหรือไม่
        """
        เปรียบเทียบป้ายทะเบียนสองป้ายเพื่อตัดสินว่าเป็นรถคันเดียวกันหรือไม่
        โดยพิจารณาการตรงกันบางส่วน และความแตกต่างระหว่างอักขระไทยกับตัวเลข
        """
        if not plate1 or not plate2: # ถ้าป้ายใดป้ายหนึ่งเป็นค่าว่าง
            return False # ถือว่าไม่เหมือนกัน
            
        # ฟังก์ชันย่อยสำหรับดึงเฉพาะตัวเลขออกจากสตริงป้ายทะเบียน
        def numbers_only(plate):
            return ''.join(c for c in plate if c.isdigit()) # สร้างสตริงใหม่ที่มีเฉพาะตัวเลข
            
        nums1 = numbers_only(plate1) # ดึงตัวเลขจากป้ายที่ 1
        nums2 = numbers_only(plate2) # ดึงตัวเลขจากป้ายที่ 2
        
        # ถ้าส่วนตัวเลขของป้ายหนึ่งเป็นส่วนหนึ่งของอีกป้าย (หรือกลับกัน)
        if nums1 in nums2 or nums2 in nums1:
            # ถ้าความยาวของส่วนตัวเลขเท่ากัน ถือว่าเป็นป้ายเดียวกัน
            if len(nums1) == len(nums2):
                return True
            # ถ้าความยาวไม่เท่ากัน (เช่น ป้ายหนึ่งอ่านได้สมบูรณ์กว่า) ก็ยังถือว่าอาจเป็นป้ายเดียวกันได้
            return True
            
        return False

    def get_time_diff_seconds(self, time1, time2): # เมธอดสำหรับคำนวณผลต่างของเวลาเป็นวินาที
        """คำนวณผลต่างของเวลาเป็นวินาทีระหว่างเวลาสองค่าในรูปแบบ HH:MM:SS"""
        t1 = datetime.strptime(time1, "%H:%M:%S") 
        t2 = datetime.strptime(time2, "%H:%M:%S") 
        return abs((t2 - t1).total_seconds()) 

    def get_better_plate(self, plate1, plate2): # เมธอดสำหรับเลือกป้ายทะเบียนที่ "ดีกว่า" หรือ "สมบูรณ์กว่า"
        """
        ตัดสินว่าหมายเลขทะเบียนใดสมบูรณ์/ดีกว่า
        โดยทั่วไป หมายเลขทะเบียนที่มีอักขระไทยจะดีกว่าหมายเลขทะเบียนที่มีแต่ตัวเลข
        """
        if not plate1: # ถ้าป้ายที่ 1 ไม่มีข้อมูล
            return plate2 # คืนค่าป้ายที่ 2
        if not plate2: # ถ้าป้ายที่ 2 ไม่มีข้อมูล
            return plate1 # คืนค่าป้ายที่ 1
            
        # ตรวจสอบว่าป้ายใดมีอักขระที่ไม่ใช่ตัวเลข (สันนิษฐานว่าเป็นอักขระไทย)
        has_thai1 = any(c for c in plate1 if not c.isdigit())
        has_thai2 = any(c for c in plate2 if not c.isdigit())
        
        if has_thai1 and not has_thai2: # ถ้าป้ายที่ 1 มีอักขระไทย และป้ายที่ 2 ไม่มี
            return plate1 # คืนค่าป้ายที่ 1
        if has_thai2 and not has_thai1: # ถ้าป้ายที่ 2 มีอักขระไทย และป้ายที่ 1 ไม่มี
            return plate2 # คืนค่าป้ายที่ 2
            
        # ถ้าทั้งสองป้ายมี (หรือไม่มี) อักขระไทยเหมือนกัน ให้เลือกป้ายที่ยาวกว่า (สมบูรณ์กว่า)
        return plate1 if len(plate1) >= len(plate2) else plate2

    def process_frame(self, frame): # เมธอดหลักในการประมวลผลแต่ละเฟรมของวิดีโอ
        height, width = frame.shape[:2] 
        # คำนวณพิกัด y ของ ROI จากสัดส่วนที่กำหนดไว้
        roi_y1 = int(height * self.roi_y_start)
        roi_y2 = int(height * self.roi_y_end)

        # วาดสี่เหลี่ยมแสดงขอบเขต ROI บนเฟรม
        cv2.rectangle(frame, (0, roi_y1), (width, roi_y2), (0, 255, 255), 2) # สีเหลือง (BGR), ความหนา 2
        # เพิ่มข้อความ "Detection Zone" เหนือ ROI
        cv2.putText(frame, "Detection Zone", (10, roi_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        current_time = time.time() # เวลาปัจจุบัน (timestamp)

        if current_time - self.last_detection_time < self.detection_cooldown:
            return frame, "", "" # คืนค่าเฟรมเดิมโดยไม่มีการตรวจจับใหม่

        # ทำการตรวจจับยานพาหนะด้วย `vehicle_model`
        # `conf=0.4` คือ confidence threshold สำหรับการตรวจจับ (ค่าความมั่นใจขั้นต่ำ 0.4)
        vehicle_results = vehicle_model(frame, conf=0.4)
        detected_classes = []
        vehicle_bbox = None 

        # วนลูปผลลัพธ์การตรวจจับยานพาหนะ (แต่ละ `result` คือ 1 ภาพ)
        for result in vehicle_results:

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                
                # คำนวณจุดกึ่งกลางของ bounding box ในแกน Y
                box_center_y = (y1 + y2) // 2

                if roi_y1 <= box_center_y <= roi_y2:
                    # วาดสี่เหลี่ยมรอบยานพาหนะที่ตรวจพบใน ROI
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                    vehicle_bbox = (x1, y1, x2, y2) 

                    # ครอบตัดเฉพาะส่วนของยานพาหนะ (car_roi) เพื่อนำไปตรวจหาตัวอักษรบนป้ายทะเบียน
                    car_roi = frame[y1:y2, x1:x2]

                    plate_results = plate_model(car_roi, conf=0.3)
                    plates = []
                    # วนลูปผลลัพธ์การตรวจจับตัวอักษร/ตัวเลข (แต่ละ `plate` คือ 1 ภาพที่ครอบตัด)
                    for plate in plate_results:
 
                        for plate_box in plate.boxes:
                            # ดึงพิกัดของ bounding box ของตัวอักษร (px1, py1, px2, py2)
                            px1_char, py1_char, px2_char, py2_char = map(int, plate_box.xyxy[0])

                            
                            # แก้ไข: ใช้ชื่อตัวแปรใหม่สำหรับพิกัดบนเฟรมเต็ม
                            abs_px1, abs_py1 = px1_char + x1, py1_char + y1
                            abs_px2, abs_py2 = px2_char + x1, py2_char + y1

                            plates.append((abs_px1, plate_box.cls, (abs_px1, abs_py1, abs_px2, abs_py2)))

                    # เรียงลำดับตัวอักษร/ตัวเลขที่ตรวจพบบนป้ายจากซ้ายไปขวาตามพิกัด x เริ่มต้น
                    plates.sort(key=lambda x: x[0])

                    # วนลูปตัวอักษร/ตัวเลขที่เรียงลำดับแล้ว
                    for plate_data in plates:
                        px1_sorted, cls, (x1_plate, y1_plate, x2_plate, y2_plate) = plate_data # ดึงข้อมูลออกมา

                        cv2.rectangle(frame, (x1_plate, y1_plate), (x2_plate, y2_plate), (255, 255, 0), 2) # สีฟ้าอ่อน, ความหนา 2
 
                        clsname = plate_model.names[int(cls)]
                        detected_classes.append(clsname)

        if detected_classes: # ถ้ามีการตรวจพบตัวอักษร/ตัวเลขบนป้ายทะเบียน
            self.last_detection_time = current_time # อัปเดตเวลาล่าสุดที่มีการตรวจจับ

            provinces = [cls for cls in detected_classes if cls in data_province]
            others = [cls for cls in detected_classes if cls not in data_province]
            detected_classes = others + provinces

            # รวมตัวอักษร/ตัวเลขที่ตรวจพบเป็นสตริงเดียว

            combined_text = "".join([get_thai_character(item) for item in detected_classes])

            license_plate, province = split_license_plate_and_province(combined_text)
            
            if license_plate or province: # ถ้าได้หมายเลขทะเบียนหรือจังหวัด
                current_time_str = datetime.now().strftime("%H:%M:%S") # เวลาปัจจุบันในรูปแบบ HH:MM:SS
                
                snapshot = None # ตัวแปรสำหรับเก็บภาพ snapshot ของยานพาหนะ
                if vehicle_bbox: # ถ้ามี bounding box ของยานพาหนะ
                    snapshot = self.get_vehicle_snapshot(frame, *vehicle_bbox) # สร้าง snapshot
                
                existing_entry = None # ตัวแปรสำหรับเก็บ entry ที่มีอยู่แล้วใน log

                for entry in self.detection_log:
                    if self.is_same_plate(entry["plate"], license_plate): # เปรียบเทียบกับป้ายใน log
                        # คำนวณผลต่างของเวลากับ entry ที่พบใน log
                        time_diff = self.get_time_diff_seconds(entry["time"], current_time_str)
                        # ถ้าผลต่างของเวลาน้อยกว่า 5 วินาที ถือว่าเป็นการตรวจจับต่อเนื่องของรถคันเดิม
                        if time_diff < 5:
                            existing_entry = entry # กำหนดให้เป็น entry ที่มีอยู่
                            better_plate = self.get_better_plate(entry["plate"], license_plate)
                            entry["plate"] = better_plate

                            break # หยุดค้นหาเมื่อเจอ entry ที่ตรงกัน
                
                if existing_entry:  # ถ้ามีการอัปเดต entry ที่มีอยู่
                    existing_entry["time"] = current_time_str  # อัปเดตเวลาของ entry
                    if snapshot:  # ถ้ามีการสร้าง snapshot ใหม่
                        existing_entry["snapshot"] = snapshot  # อัปเดต snapshot

                    if self.db_conn:
                        reg = is_registered(self.db_conn, existing_entry["plate"])
                        existing_entry["registered"] = reg
                        log_detection(
                            self.db_conn,
                            existing_entry["plate"],
                            existing_entry.get("province"),
                            existing_entry.get("snapshot"),
                            reg,
                        )

                    self.detection_log.remove(existing_entry)
                    self.detection_log.append(existing_entry)
                else: # ถ้าเป็นการตรวจจับใหม่ (ไม่มีใน log หรือนานเกิน 5 วินาที)

                    registered = False
                    if self.db_conn:
                        registered = is_registered(self.db_conn, license_plate)
                        # Get driver name from database
                        cursor = self.db_conn.cursor()
                        cursor.execute('SELECT driver_name FROM registered_vehicles WHERE plate=?', (license_plate,))
                        driver_result = cursor.fetchone()
                        driver_name = driver_result[0] if driver_result else None

                        log_detection(
                            self.db_conn,
                            license_plate,
                            province,
                            snapshot,
                            registered is not None,
                            is_employee=bool(registered)
                        )

                    self.detection_log.append({
                        "time": current_time_str,
                        "plate": license_plate,
                        "province": province,
                        "registered": registered is not None,
                        "is_employee": bool(registered),  # เพิ่มสถานะพนักงาน
                        "driver_name": driver_name if registered else None,  # เพิ่มชื่อคนขับ
                        "snapshot": snapshot,
                    })
                    # จำกัดจำนวนการตรวจจับล่าสุดใน log ไว้ที่ 10 รายการ
                    if len(self.detection_log) > 10:
                        self.detection_log.pop(0) 
                
                return frame, license_plate, province # คืนค่าเฟรมที่ประมวลผล, หมายเลขทะเบียน, และจังหวัด

        return frame, "", "" # คืนค่าเฟรมเดิม และสตริงว่างหากไม่มีการตรวจจับ

    def start_video(self, video_path): # เมธอดสำหรับเริ่มการประมวลผลวิดีโอ
        self.video_source = cv2.VideoCapture(video_path) # เปิดไฟล์วิดีโอหรือกล้องด้วย OpenCV
        self.is_running = True 
        
        threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_video(self): # เมธอดสำหรับหยุดการประมวลผลวิดีโอ
        self.is_running = False 
        if self.video_source: 
            self.video_source.release() 

    def update_frame(self): 
        while self.is_running: 
            ret, frame = self.video_source.read() # อ่านเฟรมจาก video_source
            if ret: # ถ้าอ่านเฟรมสำเร็จ (ret คือ boolean)
                processed_frame, license_plate, province = self.process_frame(frame) # ประมวลผลเฟรม
                with self.lock: 
                    self.frame = processed_frame # อัปเดตเฟรมที่ประมวลผลล่าสุด
            else: 

                self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05) # หน่วงเวลาเล็กน้อย (ประมาณ 20 FPS) เพื่อลดการใช้ CPU

    def get_frame(self): 
        if self.frame is not None: 
            with self.lock: 
                return self.frame.copy() 
        return None 

# สร้าง Flask app และ instance ของ LicensePlateDetector
app = Flask(__name__)
db_conn = init_db()
detector = LicensePlateDetector(db_conn)

def generate_frames(): 
    while True: # วนลูปไม่รู้จบเพื่อส่งเฟรมอย่างต่อเนื่อง
        frame = detector.get_frame() # ดึงเฟรมล่าสุดจาก detector
        if frame is not None: # ถ้าได้เฟรมมา
            ret, buffer = cv2.imencode('.jpg', frame) # เข้ารหัสเฟรมเป็น JPG format
            if ret: # ถ้าเข้ารหัสสำเร็จ
                frame_bytes = buffer.tobytes() # แปลง buffer เป็น bytes
                # ส่งเฟรมในรูปแบบ MJPEG stream
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # หน่วงเวลาเล็กน้อย (ประมาณ 33 FPS)

@app.route('/') 
def index():
    return render_template('index.html') # แสดงผลไฟล์ index.html

@app.route('/video') 
def video():
    # คืนค่า Response ที่เป็น MJPEG stream โดยเรียกใช้ generate_frames
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log')
def get_log():
    return jsonify(detector.detection_log) # คืนค่า detection_log ในรูปแบบ JSON


@app.route('/register_vehicle', methods=['POST'])
def add_vehicle():
    data = request.get_json() or {}
    plate = data.get('plate')
    province = data.get('province')
    driver_name = data.get('driver_name')
    if not plate:
        return jsonify({'error': 'plate required'}), 400
    register_vehicle(db_conn, plate, province, driver_name)
    return jsonify({'status': 'registered'})


@app.route('/vehicles')
def get_vehicles():
    vehicles = [
        {
            'plate': v[0],
            'province': v[1],
            'driver_name': v[2],
        }
        for v in list_vehicles(db_conn)
    ]
    return jsonify(vehicles)


@app.route('/detections')
def get_detections():
    items = list_detections(db_conn)
    detections = [
        {
            'timestamp': ts,
            'plate': p,
            'province': prov,
            'registered': bool(reg),
        }
        for ts, p, prov, reg in items
    ]
    return jsonify(detections)

@app.route('/clear_cache', methods=['POST']) 
def clear_cache():
    detector.detection_log = []  # ล้างรายการ log การตรวจจับ
    detector.last_detection_time = 0  # รีเซ็ตเวลาการตรวจจับล่าสุด (เพื่อให้ cooldown ไม่มีผลทันที)
    return jsonify({"status": "success", "message": "Cache cleared successfully"}) # คืนค่าสถานะความสำเร็จ

if __name__ == '__main__': 
    video_path = r"C:\Users\suranan\Desktop\LPR_Project\LPR_TH-main\video\Untitled video - Made with Clipchamp.mp4" # กำหนด path ของไฟล์วิดีโอ
    detector.start_video(video_path)  # เริ่มการประมวลผลวิดีโอด้วยไฟล์ที่กำหนด
    # รัน Flask development server
    # host='0.0.0.0' ทำให้สามารถเข้าถึงได้จากทุก IP address ในเครือข่าย
    # port=5000 กำหนด port ที่จะรัน server
    # debug=False ปิด debug mode สำหรับ production (หรือถ้าไม่ต้องการ debug)
    app.run(host='0.0.0.0', port=5000, debug=False)