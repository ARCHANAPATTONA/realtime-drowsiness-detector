"""
alert_system.py
---------------
Handles:
  - Alarm sound playback via pygame
  - Location retrieval (IP-based or mock)
  - SMS emergency alert via Twilio (or mock mode)
"""

import json
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Location service
# -----------------------------------------------------------------------

class LocationService:
    """Retrieves driver location via IP geolocation or falls back to mock."""

    def __init__(self, config: dict):
        self.use_ip  = config["location"]["use_ip_geolocation"]
        self.mock_lat  = config["location"]["mock_latitude"]
        self.mock_lon  = config["location"]["mock_longitude"]
        self.mock_city = config["location"]["mock_city"]

    def get_location(self) -> Tuple[float, float, str]:
        """
        Returns (latitude, longitude, city_name).
        Tries IP geolocation first; falls back to mock on failure.
        """
        if self.use_ip:
            try:
                import urllib.request
                with urllib.request.urlopen(
                    "http://ip-api.com/json/?fields=lat,lon,city,regionName", timeout=3
                ) as resp:
                    data = json.loads(resp.read())
                    lat  = data.get("lat", self.mock_lat)
                    lon  = data.get("lon", self.mock_lon)
                    city = f"{data.get('city','?')}, {data.get('regionName','?')}"
                    logger.info(f"[Location] IP geolocation: {city} ({lat}, {lon})")
                    return lat, lon, city
            except Exception as e:
                logger.warning(f"[Location] IP lookup failed ({e}), using mock.")

        return self.mock_lat, self.mock_lon, self.mock_city


# -----------------------------------------------------------------------
# Alarm
# -----------------------------------------------------------------------

class AlarmController:
    """Controls alarm sound using pygame mixer."""

    def __init__(self, config: dict):
        self.sound_path = Path(config["alert"]["alarm_sound_path"])
        self.volume     = config["alert"]["alarm_volume"]
        self._playing   = False
        self._alarm_obj = None
        self._init_pygame()

    def _init_pygame(self) -> None:
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._pygame = pygame

            if self.sound_path.exists():
                self._alarm_obj = pygame.mixer.Sound(str(self.sound_path))
                self._alarm_obj.set_volume(self.volume)
                logger.info(f"[Alarm] Loaded: {self.sound_path}")
            else:
                logger.warning(
                    f"[Alarm] Sound file not found: {self.sound_path}. "
                    "Generating synthetic beep."
                )
                self._alarm_obj = self._create_beep()

        except ImportError:
            logger.error("[Alarm] pygame not installed. No audio alerts.")
            self._pygame = None

    def _create_beep(self):
        """Generate a simple 880 Hz beep using numpy if sound file missing."""
        try:
            import numpy as np
            import pygame
            sample_rate = 44100
            duration    = 0.6        # seconds
            freq        = 880        # Hz
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            # Stereo
            stereo = np.column_stack([wave, wave])
            sound = pygame.sndarray.make_sound(stereo)
            sound.set_volume(self.volume)
            return sound
        except Exception as e:
            logger.error(f"[Alarm] Could not create synthetic beep: {e}")
            return None

    def start(self) -> None:
        if self._playing or self._pygame is None or self._alarm_obj is None:
            return
        self._alarm_obj.play(loops=-1)
        self._playing = True
        logger.info("[Alarm] Started.")

    def stop(self) -> None:
        if not self._playing or self._pygame is None or self._alarm_obj is None:
            return
        self._alarm_obj.stop()
        self._playing = False
        logger.info("[Alarm] Stopped.")

    @property
    def is_playing(self) -> bool:
        return self._playing


# -----------------------------------------------------------------------
# SMS / Emergency
# -----------------------------------------------------------------------

class EmergencyAlert:
    """
    Sends emergency SMS via Twilio or logs a mock alert.
    Runs in a background thread to avoid blocking the main loop.
    """

    def __init__(self, config: dict):
        self.use_mock    = config["alert"]["use_mock_sms"]
        self.twilio_cfg  = config["twilio"]
        self.location_svc = LocationService(config)
        self._sent       = False
        self._lock       = threading.Lock()

    def send(self, driver_status: str = "UNRESPONSIVE/DROWSY") -> None:
        """
        Dispatch the emergency alert in a daemon thread.
        Guaranteed to run only once per detector reset.
        """
        with self._lock:
            if self._sent:
                return
            self._sent = True

        thread = threading.Thread(
            target=self._dispatch,
            args=(driver_status,),
            daemon=True,
        )
        thread.start()

    def reset(self) -> None:
        with self._lock:
            self._sent = False

    def _build_message(self, lat: float, lon: float, city: str, status: str) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        maps_url  = f"https://maps.google.com/?q={lat},{lon}"
        return (
            f"🚨 DRIVER EMERGENCY ALERT 🚨\n"
            f"Time    : {timestamp}\n"
            f"Status  : {status}\n"
            f"Location: {city}\n"
            f"GPS     : {lat:.6f}, {lon:.6f}\n"
            f"Map     : {maps_url}\n"
            f"ACTION REQUIRED: Please check on the driver immediately."
        )

    def _dispatch(self, status: str) -> None:
        lat, lon, city = self.location_svc.get_location()
        message = self._build_message(lat, lon, city, status)

        if self.use_mock:
            print("\n" + "="*60)
            print("  [MOCK SMS] Emergency alert would be sent:")
            print("="*60)
            print(message)
            print("="*60 + "\n")
            logger.info("[Emergency] Mock SMS dispatched.")
        else:
            self._send_twilio(message)

    def _send_twilio(self, message: str) -> None:
        try:
            from twilio.rest import Client
            client = Client(
                self.twilio_cfg["account_sid"],
                self.twilio_cfg["auth_token"]
            )
            msg = client.messages.create(
                body=message,
                from_=self.twilio_cfg["from_number"],
                to=self.twilio_cfg["to_number"],
            )
            logger.info(f"[Emergency] Twilio SMS sent. SID: {msg.sid}")
        except ImportError:
            logger.error("[Emergency] twilio package not installed.")
        except Exception as e:
            logger.error(f"[Emergency] Twilio send failed: {e}")


# -----------------------------------------------------------------------
# Convenience wrapper
# -----------------------------------------------------------------------

class AlertSystem:
    """Unified interface consumed by main.py."""

    def __init__(self, config: dict):
        self.alarm     = AlarmController(config)
        self.emergency = EmergencyAlert(config)

    def trigger_alarm(self) -> None:
        self.alarm.start()

    def stop_alarm(self) -> None:
        self.alarm.stop()

    def send_emergency(self, status: str = "UNRESPONSIVE/DROWSY") -> None:
        self.emergency.send(status)

    def reset(self) -> None:
        self.alarm.stop()
        self.emergency.reset()
