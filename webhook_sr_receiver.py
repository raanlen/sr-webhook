#!/usr/bin/env python3
"""
🌐 WEBHOOK S/R RECEIVER
Recibe alertas de TradingView y guarda los niveles S/R en JSON + SQLite
Desplegable en EasyPanel/Docker
"""

from flask import Flask, request, jsonify
import json
import os
import sqlite3
from datetime import datetime
import logging

# Importar ML Signal Generator
try:
    from ml_signal_generator import process_sr_event, format_signal_telegram
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False
    print("⚠️ ML Signal Generator no disponible")

# ========== CONFIGURACIÓN ==========
# Variables de entorno para Docker/EasyPanel, con defaults para desarrollo local
PORT = int(os.environ.get('PORT', 5000))
SR_DATA_DIR = os.environ.get('SR_DATA_DIR', '/app/data/sr_reales')
DB_PATH = os.environ.get('DB_PATH', '/app/data/db/forex_bot.db')

# Si está en desarrollo local (no Docker), usar rutas relativas
if not os.path.exists('/app'):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SR_DATA_DIR = os.path.join(BASE_DIR, 'INTEGRACION', 'data', 'sr_reales')
    DB_PATH = os.path.join(BASE_DIR, 'NTS', 'data', 'db', 'forex_bot.db')

# Crear directorios si no existen
os.makedirs(SR_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ========== DATABASE ==========
def get_db_connection():
    """Obtiene conexión a SQLite"""
    conn = sqlite3.connect(DB_PATH)
    return conn

def save_to_db(symbol: str, level_type: str, price: float, quality: int = 1,
               confirmed: bool = True, source: str = 'PIVOT', tf_minutes: int = 15):
    """Guarda nivel S/R en SQLite"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Crear tabla si no existe
        cursor.execute('''CREATE TABLE IF NOT EXISTS sr_levels (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            level_type TEXT NOT NULL,
            price REAL NOT NULL,
            quality INTEGER DEFAULT 1,
            confirmed BOOLEAN DEFAULT 1,
            source TEXT DEFAULT 'PIVOT',
            tf_minutes INTEGER DEFAULT 15,
            detected_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            touched_count INTEGER DEFAULT 0,
            broken_at TIMESTAMP,
            UNIQUE(symbol, level_type, price)
        )''')
        
        # Insertar o actualizar
        cursor.execute('''
            INSERT INTO sr_levels (symbol, level_type, price, quality, confirmed, source, tf_minutes, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, level_type, price) DO UPDATE SET
                quality = excluded.quality,
                confirmed = excluded.confirmed,
                is_active = 1
        ''', (symbol, level_type, price, quality, confirmed, source, tf_minutes, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error save_to_db: {e}")
        return False

# ========== ENDPOINTS ==========

@app.route('/webhook/sr', methods=['POST'])
def receive_sr_alert():
    """
    Recibe alertas S/R de TradingView
    
    Formato esperado:
    {
        "current_price": 4595.40,
        "event": "SR_TABLE_UPDATE" | "SR_LOCAL" | "SR_POSSIBLE",
        "symbol": "XAUUSD",
        "tf_minutes": 15,
        "resistances": [{"n": "R1", "p": 4620.50, "q": 5, "time": "..."}],
        "supports": [{"n": "S1", "p": 4580.00, "q": 4, "time": "..."}]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        event = data.get('event', 'UNKNOWN')
        symbol = data.get('symbol', '').replace('/', '').upper()
        tf_minutes = data.get('tf_minutes', 15)
        current_price = data.get('current_price', 0)
        
        if not symbol:
            new_level = data.get('new_level', {})
            if isinstance(new_level, dict):
                symbol = new_level.get('symbol', 'UNKNOWN')
            else:
                symbol = 'UNKNOWN'
        
        logger.info(f"📥 Alerta recibida: {event} para {symbol}")
        
        # Procesar según tipo de evento
        if event in ['SR_TABLE_UPDATE', 'SR_CONFIRMED']:
            # Evento con tabla completa de S/R
            save_sr_levels(symbol, data, tf_minutes, current_price)
            return jsonify({
                "status": "ok",
                "event": event,
                "symbol": symbol,
                "current_price": current_price,
                "resistances": len(data.get('resistances', [])),
                "supports": len(data.get('supports', []))
            }), 200
            
        elif event == 'SR_LOCAL':
            # Evento de detección temprana
            update_local_sr(symbol, data, tf_minutes, current_price)
            
            # 🤖 TRIGGER ML SIGNAL GENERATOR
            ml_signal = None
            if ML_ENABLED and current_price > 0:
                try:
                    ml_signal = process_sr_event(symbol, current_price, data)
                    if ml_signal and ml_signal.get('signal'):
                        logger.info(f"🎯 SEÑAL ML GENERADA: {ml_signal['signal']} {symbol}")
                except Exception as e:
                    logger.error(f"Error ML: {e}")
            
            return jsonify({
                "status": "ok",
                "event": event,
                "symbol": symbol,
                "current_price": current_price,
                "type": data.get('type', 'UNKNOWN'),
                "price": data.get('price', 0),
                "ml_signal": ml_signal.get('signal') if ml_signal else None
            }), 200
        
        elif event == 'SR_POSSIBLE':
            # Evento de posible S/R (más temprano)
            save_possible_sr(symbol, data, tf_minutes, current_price)
            return jsonify({
                "status": "ok",
                "event": event,
                "symbol": symbol,
                "current_price": current_price,
                "possible": data.get('possible', {})
            }), 200
            
        else:
            logger.warning(f"⚠️ Evento desconocido: {event}")
            save_raw_event(symbol, data)
            return jsonify({
                "status": "ok",
                "event": event,
                "warning": "Unknown event type, saved as raw"
            }), 200
            
    except Exception as e:
        logger.error(f"❌ Error procesando webhook: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/webhook/test', methods=['GET', 'POST'])
def test_endpoint():
    """Endpoint de prueba"""
    return jsonify({
        "status": "ok",
        "message": "Webhook server running",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/sr/<symbol>', methods=['GET'])
def get_sr_levels_endpoint(symbol):
    """Obtiene los niveles S/R guardados para un símbolo"""
    try:
        symbol = symbol.upper()
        filepath = os.path.join(SR_DATA_DIR, f'{symbol}_sr.json')
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return jsonify(data), 200
        else:
            return jsonify({
                "error": f"No S/R data for {symbol}",
                "available_symbols": get_available_symbols()
            }), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/sr', methods=['GET'])
def list_all_sr():
    """Lista todos los símbolos con datos S/R"""
    try:
        symbols = get_available_symbols()
        return jsonify({
            "available_symbols": symbols,
            "count": len(symbols)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========== FUNCIONES AUXILIARES ==========

def save_sr_levels(symbol: str, data: dict, tf_minutes: int = 15, current_price: float = 0):
    """Guarda los niveles S/R en archivo JSON Y en SQLite"""
    filepath = os.path.join(SR_DATA_DIR, f'{symbol}_sr.json')
    
    # Añadir metadata
    data['last_updated'] = datetime.now().isoformat()
    data['symbol'] = symbol
    if current_price > 0:
        data['current_price'] = current_price
        data['current_price_time'] = datetime.now().isoformat()
        # Guardar historial para ML
        save_price_history(symbol, current_price, data.get('event', 'SR_TABLE_UPDATE'), data)
    
    # Normalizar resistances y supports para aceptar ambos formatos:
    # Formato 1: [3250.5, 3260.0, 3275.0]  (array de números)
    # Formato 2: [{"n": "R1", "p": 3250.5, "q": 5}]  (array de dicts)
    quality_global = data.get('q', 1)  # Calidad global si viene en el mensaje
    
    def normalize_levels(levels, prefix):
        normalized = []
        for i, level in enumerate(levels):
            if isinstance(level, (int, float)):
                # Es un número simple
                normalized.append({
                    "n": f"{prefix}{i+1}",
                    "p": float(level),
                    "q": quality_global,
                    "time": datetime.now().strftime('%d %b %H:%M')
                })
            elif isinstance(level, dict):
                # Ya es un diccionario
                normalized.append(level)
        return normalized
    
    data['resistances'] = normalize_levels(data.get('resistances', []), 'R')
    data['supports'] = normalize_levels(data.get('supports', []), 'S')
    
    # Guardar en JSON (INTEGRACION)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Guardar en SQLite (NTS)
    db_saved = 0
    for r in data.get('resistances', []):
        if save_to_db(symbol, 'RES', r.get('p', 0), r.get('q', 1), True, 'PIVOT', tf_minutes):
            db_saved += 1
    
    for s in data.get('supports', []):
        if save_to_db(symbol, 'SUP', s.get('p', 0), s.get('q', 1), True, 'PIVOT', tf_minutes):
            db_saved += 1
    
    r_count = len(data.get('resistances', []))
    s_count = len(data.get('supports', []))
    logger.info(f"✅ Guardados {r_count}R + {s_count}S para {symbol} (JSON + {db_saved} en DB)")


def update_local_sr(symbol: str, data: dict, tf_minutes: int = 15, current_price: float = 0):
    """Actualiza S/R local (añade a existentes si no es duplicado)"""
    filepath = os.path.join(SR_DATA_DIR, f'{symbol}_sr.json')
    
    # Cargar existentes o crear nuevo
    existing = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing = json.load(f)
    
    sr_type = data.get('type', 'RES')
    price = data.get('price', 0)
    
    # Determinar lista a actualizar
    list_key = 'resistances' if sr_type == 'RES' else 'supports'
    
    if list_key not in existing:
        existing[list_key] = []
    
    # Verificar si ya existe (tolerancia)
    tolerance = 0.5 if symbol == 'XAUUSD' else 0.0005  # Ajustar según par
    is_duplicate = any(
        abs(level.get('p', 0) - price) < tolerance 
        for level in existing[list_key]
    )
    
    if not is_duplicate:
        # Añadir nuevo nivel local
        new_level = {
            "n": f"{'R' if sr_type == 'RES' else 'S'}L",  # L = Local
            "p": price,
            "time": data.get('time', datetime.now().strftime('%d %b %H:%M')),
            "q": 1,  # Calidad baja (no confirmado)
            "confirmed": False,
            "current_price": current_price if current_price > 0 else None
        }
        existing[list_key].insert(0, new_level)
        
        # Limitar a 15 niveles máximo
        existing[list_key] = existing[list_key][:15]
        
        existing['last_updated'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=2)
        
        logger.info(f"📍 Añadido SR_LOCAL: {sr_type} @ {price} para {symbol} (precio actual: {current_price})")
        # Guardar historial para ML
        if current_price > 0:
            save_price_history(symbol, current_price, 'SR_LOCAL', data)
    else:
        logger.info(f"⏭️ Nivel duplicado ignorado: {sr_type} @ {price}")


def save_possible_sr(symbol: str, data: dict, tf_minutes: int = 15, current_price: float = 0):
    """Guarda evento SR_POSSIBLE para análisis temprano"""
    raw_dir = os.path.join(SR_DATA_DIR, 'possible')
    os.makedirs(raw_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(raw_dir, f'{symbol}_{timestamp}.json')
    
    # Enriquecer con current_price
    data['current_price'] = current_price
    data['timestamp'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Guardar historial para ML
    if current_price > 0:
        save_price_history(symbol, current_price, 'SR_POSSIBLE', data)
    
    possible = data.get('possible', {})
    logger.info(f"🔮 SR_POSSIBLE: {possible.get('type', '?')} @ {possible.get('p', 0)} para {symbol} (precio: {current_price})")


def save_price_history(symbol: str, current_price: float, event: str, data: dict):
    """Guarda historial de precios con contexto S/R para entrenamiento ML"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Crear tabla si no existe
        cursor.execute('''CREATE TABLE IF NOT EXISTS sr_price_history (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            current_price REAL NOT NULL,
            event_type TEXT NOT NULL,
            sr_type TEXT,
            sr_price REAL,
            distance_to_sr REAL,
            resistances TEXT,
            supports TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Extraer datos del evento (con verificación de tipos)
        sr_type = ''
        sr_price = 0
        
        # Verificar tipo de data.get('type')
        if data.get('type'):
            sr_type = data.get('type', '')
        
        # Verificar si possible es un dict antes de acceder
        possible = data.get('possible', {})
        if isinstance(possible, dict) and not sr_type:
            sr_type = possible.get('type', '')
            sr_price = possible.get('p', 0)
        
        # Verificar si new_level es un dict
        new_level = data.get('new_level', {})
        if isinstance(new_level, dict) and not sr_type:
            sr_type = new_level.get('type', '')
            sr_price = new_level.get('price', 0) or sr_price
        
        # Fallback a price si existe
        if not sr_price and data.get('price'):
            sr_price = data.get('price', 0)
            
        distance = abs(current_price - sr_price) if sr_price and current_price else None
        
        # Serializar resistencias y soportes
        resistances = json.dumps(data.get('resistances', []))
        supports = json.dumps(data.get('supports', []))
        
        cursor.execute('''
            INSERT INTO sr_price_history 
            (symbol, current_price, event_type, sr_type, sr_price, distance_to_sr, resistances, supports)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, current_price, event, sr_type, sr_price, distance, resistances, supports))
        
        conn.commit()
        conn.close()
        logger.info(f"📊 Historial ML guardado: {symbol} @ {current_price} ({event})")
    except Exception as e:
        logger.error(f"Error guardando historial ML: {e}")


def save_raw_event(symbol: str, data: dict):
    """Guarda evento raw para debugging"""
    raw_dir = os.path.join(SR_DATA_DIR, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(raw_dir, f'{symbol}_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)



    
    logger.info(f"📁 Evento raw guardado: {filepath}")


def get_available_symbols() -> list:
    """Retorna lista de símbolos con datos S/R"""
    symbols = []
    if os.path.exists(SR_DATA_DIR):
        for filename in os.listdir(SR_DATA_DIR):
            if filename.endswith('_sr.json'):
                symbols.append(filename.replace('_sr.json', ''))
    return sorted(symbols)


# ========== MAIN ==========

if __name__ == '__main__':
    logger.info(f"""
╔════════════════════════════════════════════════════╗
║  🌐 WEBHOOK S/R RECEIVER                          ║
║  Puerto: {PORT}                                       ║
║  S/R Dir: {SR_DATA_DIR[:35]}...
║  DB: {DB_PATH[:40]}...
║                                                    ║
║  Endpoints:                                        ║
║  POST /webhook/sr    → Recibir alertas            ║
║  GET  /webhook/test  → Probar conexión            ║
║  GET  /sr/<symbol>   → Ver S/R de símbolo         ║
║  GET  /sr            → Listar símbolos            ║
╚════════════════════════════════════════════════════╝
    """)
    
    # En producción usar gunicorn: gunicorn -b 0.0.0.0:5000 webhook_sr_receiver:app
    # En desarrollo usar Flask:
    is_production = os.environ.get('PRODUCTION', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=PORT, debug=not is_production)




