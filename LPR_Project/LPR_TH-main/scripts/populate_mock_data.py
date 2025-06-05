import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from function.database import init_db, register_vehicle

# Connect to database (creates if not exists)
conn = init_db()

thai_letters = [
    'ก','ข','ค','ง','จ','ฉ','ช','ซ','ญ','ด','ต','ถ','ท','ธ','น','บ','ป','ผ','พ','ฟ','ม','ย','ร','ล','ว','ศ','ส','ห','อ','ฮ'
]

provinces = ['กรุงเทพมหานคร', 'ชลบุรี', 'เชียงใหม่', 'นครราชสีมา', 'ภูเก็ต']

for i in range(100):
    prefix = random.choice(thai_letters) + random.choice(thai_letters)
    number = 1000 + i
    plate = f"{prefix}{number}"
    driver_name = f"คนขับ{i+1}"
    province = random.choice(provinces)
    register_vehicle(conn, plate, province, driver_name)

print("Inserted 100 mock vehicle records.")
