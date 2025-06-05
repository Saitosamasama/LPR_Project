import sys
from pathlib import Path

import pytest

flask_mod = pytest.importorskip("flask")
Flask = flask_mod.Flask
jsonify = flask_mod.jsonify

# Make sure project modules can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

from function.database import (
    init_db,
    register_vehicle,
    list_vehicles,
    is_registered,
    log_detection,
    list_detections,
)


@pytest.fixture
def conn():
    conn = init_db(':memory:')
    yield conn
    conn.close()


def test_init_db_creates_tables(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert 'registered_vehicles' in tables
    assert 'detections' in tables


def test_register_and_list_vehicles(conn):
    register_vehicle(conn, 'ABC123', 'Bangkok', 'John', True)
    vehicles = list_vehicles(conn)
    assert vehicles == [('ABC123', 'Bangkok', 'John', 1)]


def test_is_registered(conn):
    register_vehicle(conn, 'XYZ999', 'Bangkok', 'Jane', False)
    assert is_registered(conn, 'XYZ999') == 0
    assert is_registered(conn, 'NOTEXIST') is None


def test_log_and_list_detections(conn):
    log_detection(conn, 'XYZ999', 'Bangkok', 'snap.jpg', True, False)
    detections = list_detections(conn)
    assert len(detections) == 1
    ts, plate, prov, registered, is_emp = detections[0]
    assert plate == 'XYZ999'
    assert prov == 'Bangkok'
    assert registered == 1
    assert is_emp == 0


def create_test_app(conn):
    app = Flask(__name__)

    @app.route('/vehicles')
    def vehicles():
        data = list_vehicles(conn)
        return jsonify([
            {
                'plate': d[0],
                'province': d[1],
                'driver_name': d[2],
                'is_employee': bool(d[3]),
            }
            for d in data
        ])

    @app.route('/detections')
    def detections():
        data = list_detections(conn)
        return jsonify([
            {
                'timestamp': ts,
                'plate': plate,
                'province': prov,
                'registered': bool(reg),
                'is_employee': bool(emp),
            }
            for ts, plate, prov, reg, emp in data
        ])

    return app


def test_vehicle_and_detection_routes(conn):
    register_vehicle(conn, 'ROUTE1', 'Bangkok', 'Tester', False)
    log_detection(conn, 'ROUTE1', 'Bangkok', 'snap.jpg', True, False)
    app = create_test_app(conn)
    client = app.test_client()

    rv = client.get('/vehicles')
    assert rv.status_code == 200
    assert rv.get_json() == [
        {
            'plate': 'ROUTE1',
            'province': 'Bangkok',
            'driver_name': 'Tester',
            'is_employee': False,
        }
    ]

    rd = client.get('/detections')
    assert rd.status_code == 200
    data = rd.get_json()
    assert len(data) == 1
    assert data[0]['plate'] == 'ROUTE1'
    assert data[0]['province'] == 'Bangkok'
    assert data[0]['registered'] is True


