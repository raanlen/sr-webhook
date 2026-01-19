#!/usr/bin/env python3
"""
🤖 ML SIGNAL GENERATOR - INTEGRADO CON AGENTE_PRECIOS_REALES
Genera señales de trading combinando:
1. Niveles S/R REALES de TradingView (via webhook)
2. Noticias de Alpha Vantage + Marketaux
3. Análisis Técnico de agente_precios_reales.py
4. Cálculo de probabilidad multi-factor

Autor: Sistema de Trading Automatizado
Integración: agente_precios_reales.py + webhook_sr_receiver.py
"""

import os
import json
import sqlite3
import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# Configuración de logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========== CONFIGURACIÓN ==========
MARKETAUX_API_KEY = os.environ.get('MARKETAUX_API_KEY', '5xu4J5I6QWn0CbMcQd3HdcJh03MtujCN2zvFGROk')
DB_PATH = os.environ.get('DB_PATH', '/app/data/db/forex_bot.db')
SR_DATA_DIR = os.environ.get('SR_DATA_DIR', '/app/data/sr_reales')

# Si está en desarrollo local
if not os.path.exists('/app'):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(BASE_DIR, 'NTS', 'data', 'db', 'forex_bot.db')
    SR_DATA_DIR = os.path.join(BASE_DIR, 'INTEGRACION', 'data', 'sr_reales')

# Configuración de pares
FOREX_PAIRS = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 
               'USDCAD', 'NZDUSD', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY']

# Multiplicadores de pips por par
PIP_MULTIPLIERS = {
    'XAUUSD': 0.1,  # $0.10 = 1 pip en oro
    'USDJPY': 100,
    'EURJPY': 100,
    'GBPJPY': 100,
    'DEFAULT': 10000
}

# Umbrales de trading
PROBABILITY_THRESHOLD = 0.85  # 85% mínimo para señal
DISTANCE_PIPS_MIN = 5  # Mínimo 5 pips de S/R para entrar
DISTANCE_PIPS_MAX = 50  # Máximo 50 pips para considerar "cerca"


class NewsAnalyzer:
    """Analiza noticias de Marketaux para un par específico"""
    
    @staticmethod
    def fetch_news_for_pair(pair: str, hours: int = 4) -> List[Dict]:
        """Obtiene noticias recientes para un par específico"""
        try:
            published_after = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime('%Y-%m-%dT%H:%M:%S')
            params = {
                'api_token': MARKETAUX_API_KEY,
                'entity_types': 'currency',
                'filter_entities': 'true',
                'must_have_entities': 'true',
                'language': 'en',
                'limit': 20,
                'published_after': published_after.split('T')[0]  # Solo fecha
            }
            
            response = requests.get('https://api.marketaux.com/v1/news/all', params=params, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"⚠️ Marketaux API error: {response.status_code}")
                return []
            
            data = response.json()
            relevant_news = []
            
            for news in data.get('data', []):
                # Verificar si la noticia es relevante para el par
                for entity in news.get('entities', []):
                    if entity.get('type') == 'currency':
                        symbol = entity.get('symbol', '')
                        # Normalizar par
                        if symbol == pair or NewsAnalyzer._normalize_pair(symbol) == pair:
                            relevant_news.append({
                                'title': news.get('title', ''),
                                'sentiment_score': entity.get('sentiment_score', 0),
                                'published_at': news.get('published_at', ''),
                                'source': news.get('source', 'Unknown')
                            })
                            break
            
            logger.info(f"📰 Encontradas {len(relevant_news)} noticias para {pair}")
            return relevant_news
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    @staticmethod
    def _normalize_pair(symbol: str) -> Optional[str]:
        """Normaliza pares invertidos"""
        if symbol in FOREX_PAIRS:
            return symbol
        if len(symbol) == 6:
            inverted = symbol[3:] + symbol[:3]
            if inverted in FOREX_PAIRS:
                return inverted
        return None
    
    @staticmethod
    def analyze_sentiment(news_list: List[Dict]) -> Dict:
        """Analiza el sentimiento agregado de las noticias"""
        if not news_list:
            return {
                'count': 0,
                'avg_sentiment': 0,
                'direction': 'NEUTRAL',
                'confidence': 0
            }
        
        sentiments = [n.get('sentiment_score', 0) for n in news_list]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Determinar dirección
        if avg_sentiment > 0.15:
            direction = 'BULLISH'
        elif avg_sentiment < -0.15:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        # Calcular confianza basada en consistencia
        same_direction = sum(1 for s in sentiments if (s > 0) == (avg_sentiment > 0))
        consistency = same_direction / len(sentiments) if sentiments else 0
        
        return {
            'count': len(news_list),
            'avg_sentiment': avg_sentiment,
            'direction': direction,
            'confidence': abs(avg_sentiment) * consistency
        }


class SRAnalyzer:
    """Analiza niveles de Soporte y Resistencia"""
    
    @staticmethod
    def load_sr_levels(symbol: str) -> Dict:
        """Carga niveles S/R desde archivo JSON"""
        filepath = os.path.join(SR_DATA_DIR, f'{symbol}_sr.json')
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {'resistances': [], 'supports': []}
    
    @staticmethod
    def find_nearest_sr(symbol: str, current_price: float) -> Dict:
        """Encuentra el S/R más cercano al precio actual"""
        data = SRAnalyzer.load_sr_levels(symbol)
        
        pip_mult = PIP_MULTIPLIERS.get(symbol, PIP_MULTIPLIERS['DEFAULT'])
        
        nearest_resistance = None
        nearest_support = None
        dist_to_r = float('inf')
        dist_to_s = float('inf')
        
        # Buscar resistencia más cercana (arriba del precio)
        for r in data.get('resistances', []):
            price = r.get('p', 0)
            if price > current_price:
                dist = (price - current_price) * pip_mult
                if dist < dist_to_r:
                    dist_to_r = dist
                    nearest_resistance = {
                        'price': price,
                        'quality': r.get('q', 1),
                        'name': r.get('n', 'R'),
                        'distance_pips': dist
                    }
        
        # Buscar soporte más cercano (debajo del precio)
        for s in data.get('supports', []):
            price = s.get('p', 0)
            if price < current_price:
                dist = (current_price - price) * pip_mult
                if dist < dist_to_s:
                    dist_to_s = dist
                    nearest_support = {
                        'price': price,
                        'quality': s.get('q', 1),
                        'name': s.get('n', 'S'),
                        'distance_pips': dist
                    }
        
        return {
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'price_position': SRAnalyzer._determine_position(dist_to_r, dist_to_s)
        }
    
    @staticmethod
    def _determine_position(dist_to_r: float, dist_to_s: float) -> str:
        """Determina la posición del precio respecto a S/R"""
        if dist_to_r == float('inf') and dist_to_s == float('inf'):
            return 'NO_LEVELS'
        elif dist_to_r == float('inf'):
            return 'ABOVE_RESISTANCES'
        elif dist_to_s == float('inf'):
            return 'BELOW_SUPPORTS'
        elif dist_to_r < DISTANCE_PIPS_MIN:
            return 'AT_RESISTANCE'
        elif dist_to_s < DISTANCE_PIPS_MIN:
            return 'AT_SUPPORT'
        elif dist_to_r < dist_to_s:
            return 'NEAR_RESISTANCE'
        else:
            return 'NEAR_SUPPORT'


class SignalGenerator:
    """Genera señales de trading combinando S/R + Noticias"""
    
    def __init__(self):
        self.db_path = DB_PATH
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._setup_db()
    
    def _setup_db(self):
        """Crea tabla de señales ML si no existe"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS ml_signals (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                nearest_sr_type TEXT,
                nearest_sr_price REAL,
                distance_to_sr REAL,
                news_count INTEGER,
                news_sentiment TEXT,
                news_confidence REAL,
                probability REAL,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                executed_at TIMESTAMP,
                result TEXT DEFAULT 'PENDING'
            )''')
            conn.commit()
            conn.close()
            logger.info("✅ Tabla ml_signals lista")
        except Exception as e:
            logger.error(f"Error setup DB: {e}")
    
    def analyze_opportunity(self, symbol: str, current_price: float, event_data: dict = None) -> Dict:
        """
        Analiza una oportunidad de trading al recibir evento S/R
        
        Args:
            symbol: Par de divisas (ej: XAUUSD)
            current_price: Precio actual del mercado
            event_data: Datos del evento S/R recibido
            
        Returns:
            Dict con señal generada o None si no hay oportunidad
        """
        logger.info(f"🔍 Analizando oportunidad para {symbol} @ {current_price}")
        
        # 1. Analizar posición respecto a S/R
        sr_analysis = SRAnalyzer.find_nearest_sr(symbol, current_price)
        
        # 2. Obtener noticias recientes
        news = NewsAnalyzer.fetch_news_for_pair(symbol, hours=4)
        news_analysis = NewsAnalyzer.analyze_sentiment(news)
        
        # 3. Calcular probabilidad de éxito
        probability, signal_type, reason = self._calculate_probability(
            sr_analysis, news_analysis, current_price, event_data
        )
        
        # 4. Generar señal si probabilidad es suficiente
        if probability >= PROBABILITY_THRESHOLD:
            signal = self._generate_signal(
                symbol, current_price, signal_type, probability,
                sr_analysis, news_analysis, reason
            )
            
            # Guardar en base de datos
            self._save_signal(signal)
            
            logger.info(f"🎯 SEÑAL GENERADA: {signal_type} {symbol} @ {current_price} (prob: {probability*100:.1f}%)")
            return signal
        else:
            logger.info(f"⏳ Sin señal: probabilidad {probability*100:.1f}% < {PROBABILITY_THRESHOLD*100:.0f}%")
            return {
                'signal': None,
                'probability': probability,
                'reason': reason,
                'sr_analysis': sr_analysis,
                'news_analysis': news_analysis
            }
    
    def _calculate_probability(self, sr_analysis: Dict, news_analysis: Dict, 
                               current_price: float, event_data: dict = None) -> Tuple[float, str, str]:
        """
        Calcula probabilidad de éxito del trade
        
        Factores:
        - Proximidad a S/R (30%)
        - Calidad del S/R (20%)
        - Sentimiento de noticias (30%)
        - Consistencia de noticias (20%)
        """
        base_probability = 0.5  # 50% base
        signal_type = None
        reasons = []
        
        position = sr_analysis.get('price_position', 'NO_LEVELS')
        nearest_r = sr_analysis.get('nearest_resistance')
        nearest_s = sr_analysis.get('nearest_support')
        
        # ========== Factor 1: Proximidad a S/R (30%) ==========
        proximity_score = 0
        
        if position == 'AT_SUPPORT':
            # Precio en soporte = oportunidad BUY
            signal_type = 'BUY'
            proximity_score = 0.30
            reasons.append(f"Precio en soporte {nearest_s['name']}")
            
        elif position == 'AT_RESISTANCE':
            # Precio en resistencia = oportunidad SELL
            signal_type = 'SELL'
            proximity_score = 0.30
            reasons.append(f"Precio en resistencia {nearest_r['name']}")
            
        elif position == 'NEAR_SUPPORT' and nearest_s:
            dist = nearest_s.get('distance_pips', 100)
            if dist < DISTANCE_PIPS_MAX:
                signal_type = 'BUY'
                proximity_score = 0.25 * (1 - dist / DISTANCE_PIPS_MAX)
                reasons.append(f"Cerca de soporte ({dist:.1f} pips)")
                
        elif position == 'NEAR_RESISTANCE' and nearest_r:
            dist = nearest_r.get('distance_pips', 100)
            if dist < DISTANCE_PIPS_MAX:
                signal_type = 'SELL'
                proximity_score = 0.25 * (1 - dist / DISTANCE_PIPS_MAX)
                reasons.append(f"Cerca de resistencia ({dist:.1f} pips)")
        
        # ========== Factor 2: Calidad del S/R (20%) ==========
        quality_score = 0
        if signal_type == 'BUY' and nearest_s:
            quality = nearest_s.get('quality', 1)
            quality_score = min(0.20, quality * 0.04)  # Hasta 5 estrellas
            if quality >= 3:
                reasons.append(f"Soporte fuerte (Q={quality})")
                
        elif signal_type == 'SELL' and nearest_r:
            quality = nearest_r.get('quality', 1)
            quality_score = min(0.20, quality * 0.04)
            if quality >= 3:
                reasons.append(f"Resistencia fuerte (Q={quality})")
        
        # ========== Factor 3: Sentimiento de noticias (30%) ==========
        news_score = 0
        news_direction = news_analysis.get('direction', 'NEUTRAL')
        news_confidence = news_analysis.get('confidence', 0)
        
        # Verificar alineación con señal
        if signal_type == 'BUY' and news_direction == 'BULLISH':
            news_score = 0.30 * min(1.0, news_confidence * 2)
            reasons.append(f"Noticias BULLISH ({news_analysis['count']} notas)")
            
        elif signal_type == 'SELL' and news_direction == 'BEARISH':
            news_score = 0.30 * min(1.0, news_confidence * 2)
            reasons.append(f"Noticias BEARISH ({news_analysis['count']} notas)")
            
        elif news_direction == 'NEUTRAL':
            # Neutral no suma ni resta mucho
            news_score = 0.10
            
        else:
            # Noticias en contra = penalización
            news_score = -0.15
            reasons.append(f"⚠️ Noticias en contra ({news_direction})")
        
        # ========== Factor 4: Cantidad de noticias (20%) ==========
        count_score = 0
        news_count = news_analysis.get('count', 0)
        if news_count >= 3:
            count_score = 0.20
        elif news_count >= 1:
            count_score = 0.10
        else:
            count_score = 0.05  # Sin noticias = menos confianza
        
        # ========== Cálculo final ==========
        probability = base_probability + proximity_score + quality_score + news_score + count_score
        probability = max(0.0, min(1.0, probability))  # Clamp entre 0 y 1
        
        reason = " + ".join(reasons) if reasons else "Sin factores claros"
        
        # Si no hay señal clara, retornar None
        if signal_type is None:
            return (probability, 'HOLD', "Sin oportunidad clara")
        
        return (probability, signal_type, reason)
    
    def _generate_signal(self, symbol: str, current_price: float, signal_type: str,
                         probability: float, sr_analysis: Dict, news_analysis: Dict,
                         reason: str) -> Dict:
        """Genera señal completa con SL/TP"""
        
        pip_mult = PIP_MULTIPLIERS.get(symbol, PIP_MULTIPLIERS['DEFAULT'])
        pip_value = 1 / pip_mult
        
        # Calcular SL/TP basados en S/R
        nearest_r = sr_analysis.get('nearest_resistance')
        nearest_s = sr_analysis.get('nearest_support')
        
        if signal_type == 'BUY':
            # SL debajo del soporte, TP en resistencia
            sl_pips = 15 if not nearest_s else max(10, nearest_s['distance_pips'] + 5)
            tp_pips = 30 if not nearest_r else min(100, nearest_r['distance_pips'] * 0.8)
            
            stop_loss = current_price - (sl_pips * pip_value)
            take_profit = current_price + (tp_pips * pip_value)
            sr_used = nearest_s
            sr_type = 'SUPPORT'
            
        else:  # SELL
            # SL arriba de resistencia, TP en soporte
            sl_pips = 15 if not nearest_r else max(10, nearest_r['distance_pips'] + 5)
            tp_pips = 30 if not nearest_s else min(100, nearest_s['distance_pips'] * 0.8)
            
            stop_loss = current_price + (sl_pips * pip_value)
            take_profit = current_price - (tp_pips * pip_value)
            sr_used = nearest_r
            sr_type = 'RESISTANCE'
        
        return {
            'signal': signal_type,
            'symbol': symbol,
            'entry_price': current_price,
            'stop_loss': round(stop_loss, 5 if 'JPY' not in symbol else 3),
            'take_profit': round(take_profit, 5 if 'JPY' not in symbol else 3),
            'probability': probability,
            'reason': reason,
            'sr_type': sr_type,
            'sr_price': sr_used['price'] if sr_used else None,
            'distance_to_sr': sr_used['distance_pips'] if sr_used else None,
            'news_count': news_analysis.get('count', 0),
            'news_sentiment': news_analysis.get('direction', 'NEUTRAL'),
            'news_confidence': news_analysis.get('confidence', 0),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
    
    def _save_signal(self, signal: Dict):
        """Guarda señal en base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO ml_signals 
                (symbol, signal_type, entry_price, stop_loss, take_profit,
                 nearest_sr_type, nearest_sr_price, distance_to_sr,
                 news_count, news_sentiment, news_confidence, probability, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (signal['symbol'], signal['signal'], signal['entry_price'],
                 signal['stop_loss'], signal['take_profit'], signal['sr_type'],
                 signal.get('sr_price'), signal.get('distance_to_sr'),
                 signal['news_count'], signal['news_sentiment'],
                 signal['news_confidence'], signal['probability'], signal['reason']))
            conn.commit()
            conn.close()
            logger.info(f"💾 Señal guardada en DB")
        except Exception as e:
            logger.error(f"Error guardando señal: {e}")
    
    def get_pending_signals(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Obtiene señales pendientes"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM ml_signals WHERE result = 'PENDING'"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            
            return results
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return []


def format_signal_telegram(signal: Dict) -> str:
    """Formatea señal para enviar a Telegram"""
    if signal.get('signal') is None:
        return None
    
    emoji = '🟢 COMPRA' if signal['signal'] == 'BUY' else '🔴 VENTA'
    probability_pct = signal['probability'] * 100
    
    # Formato de precios según par
    symbol = signal['symbol']
    if symbol == 'XAUUSD':
        entry_fmt = f"${signal['entry_price']:.2f}"
        sl_fmt = f"${signal['stop_loss']:.2f}"
        tp_fmt = f"${signal['take_profit']:.2f}"
    elif 'JPY' in symbol:
        entry_fmt = f"{signal['entry_price']:.3f}"
        sl_fmt = f"{signal['stop_loss']:.3f}"
        tp_fmt = f"{signal['take_profit']:.3f}"
    else:
        entry_fmt = f"{signal['entry_price']:.5f}"
        sl_fmt = f"{signal['stop_loss']:.5f}"
        tp_fmt = f"{signal['take_profit']:.5f}"
    
    msg = f"""
🎯 *SEÑAL ML - {symbol}*
{emoji}

📍 *Entrada:* {entry_fmt}
🛑 *Stop Loss:* {sl_fmt}
🎯 *Take Profit:* {tp_fmt}

📊 *Probabilidad:* {probability_pct:.1f}%
📰 *Noticias:* {signal['news_count']} ({signal['news_sentiment']})

💡 *Razón:* {signal['reason']}

⏰ {signal.get('created_at', 'Ahora')[:16]}
"""
    return msg.strip()


# ========== FUNCIONES PARA INTEGRAR CON WEBHOOK ==========

def process_sr_event(symbol: str, current_price: float, event_data: dict) -> Optional[Dict]:
    """
    Función principal para llamar desde webhook_sr_receiver.py
    
    Args:
        symbol: Par de divisas
        current_price: Precio actual
        event_data: Datos del evento S/R
        
    Returns:
        Señal generada o None
    """
    generator = SignalGenerator()
    result = generator.analyze_opportunity(symbol, current_price, event_data)
    
    if result.get('signal'):
        return result
    return None


if __name__ == '__main__':
    # Prueba rápida
    print("🧪 Probando ML Signal Generator...")
    
    generator = SignalGenerator()
    
    # Simular análisis de XAUUSD
    result = generator.analyze_opportunity('XAUUSD', 2720.50, {
        'event': 'SR_LOCAL',
        'type': 'SUP',
        'price': 2718.00
    })
    
    print(f"\n📊 Resultado:")
    print(json.dumps(result, indent=2, default=str))
    
    if result.get('signal'):
        msg = format_signal_telegram(result)
        print(f"\n📱 Mensaje Telegram:\n{msg}")
