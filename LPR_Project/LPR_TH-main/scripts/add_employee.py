import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from function.database import init_db, register_vehicle

# เชื่อมต่อกับฐานข้อมูล
conn = init_db()

# ลงทะเบียนรถพนักงาน
plate = "3ฌฌ8939"
province = "กรุงเทพมหานคร"
driver_name = "นายสุรนันท์ ปานกลาง"
is_employee = True  # เพิ่มสถานะพนักงาน

register_vehicle(conn, plate, province, driver_name, is_employee=True)

print(f"เพิ่มข้อมูลพนักงาน {driver_name} ทะเบียน {plate} เรียบร้อยแล้ว") 