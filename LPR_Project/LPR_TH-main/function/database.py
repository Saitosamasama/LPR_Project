import sqlite3
from datetime import datetime

DB_PATH = 'vehicle.db'

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS registered_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            province TEXT,
            driver_name TEXT,
            is_employee BOOLEAN DEFAULT FALSE
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            plate TEXT,
            province TEXT,
            snapshot TEXT,
            registered INTEGER,
            is_employee BOOLEAN DEFAULT FALSE
        )
    ''')
    conn.commit()
    return conn

def register_vehicle(conn, plate, province=None, driver_name=None, is_employee=False):
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO registered_vehicles (plate, province, driver_name, is_employee) VALUES (?, ?, ?, ?)',
        (plate, province, driver_name, is_employee)
    )
    conn.commit()


def list_vehicles(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT plate, province, driver_name, is_employee FROM registered_vehicles')
    return cursor.fetchall()


def is_registered(conn, plate):
    cursor = conn.cursor()
    cursor.execute('SELECT is_employee FROM registered_vehicles WHERE plate=?', (plate,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return None


def log_detection(conn, plate, province, snapshot, registered, is_employee=False):
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO detections (timestamp, plate, province, snapshot, registered, is_employee) VALUES (?, ?, ?, ?, ?, ?)',
        (datetime.now().isoformat(), plate, province, snapshot, int(registered), is_employee)
    )
    conn.commit()


def list_detections(conn, limit=100):
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, plate, province, registered, is_employee FROM detections ORDER BY id DESC LIMIT ?', (limit,))
    return cursor.fetchall()
