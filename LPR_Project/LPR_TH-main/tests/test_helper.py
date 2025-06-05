import sys
from pathlib import Path

# Add project root to path so modules can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

from function.helper import split_license_plate_and_province, get_thai_character, data_province


def test_split_license_plate_and_province_basic():
    text = "กข1234กรุงเทพมหานคร"
    plate, province = split_license_plate_and_province(text)
    assert plate == "กข1234"
    assert province == "กรุงเทพมหานคร"


def test_split_license_plate_and_province_no_numbers():
    text = "ไม่มีตัวเลขที่นี่"
    plate, province = split_license_plate_and_province(text)
    assert plate is None
    assert province is None


def test_get_thai_character():
    assert get_thai_character("A01") == "ก"


def test_get_thai_character_province():
    assert get_thai_character("BKK") == "กรุงเทพมหานคร"


def test_data_province_lookup():
    assert data_province["BKK"] == "กรุงเทพมหานคร"
