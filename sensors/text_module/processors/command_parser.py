from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ..schema.textseg import CommandIntent, IntentScore

@dataclass
class CommandParser:
    def parse(self, text: str) -> List[IntentScore]:
        t = (text or "").lower().strip()
        intents: List[IntentScore] = []

        def has_any(keys: List[str]) -> bool:
            return any(k in t for k in keys)

        if has_any(["الان", "now", "right now"]) and has_any(["چه خبره", "what is happening", "what's going on", "scene"]):
            intents.append(IntentScore(intent=CommandIntent.DESCRIBE_SCENE, confidence=0.85))

        if has_any(["آدم", "person", "people", "human"]) and has_any(["میبینی", "see", "detect", "there"]):
            intents.append(IntentScore(intent=CommandIntent.DETECT_PERSON, confidence=0.80))

        if has_any(["روشن", "نور", "bright", "brightness", "dark"]):
            intents.append(IntentScore(intent=CommandIntent.SCENE_BRIGHTNESS, confidence=0.72))

        if has_any(["شلوغ", "crowd", "busy", "crowded"]):
            intents.append(IntentScore(intent=CommandIntent.SCENE_CROWD, confidence=0.70))

        if has_any(["کجاست", "where", "location", "inside", "outside", "indoor", "outdoor", "کوه", "دریا", "mountain", "sea"]):
            intents.append(IntentScore(intent=CommandIntent.SCENE_LOCATION_TYPE, confidence=0.68))

        if has_any(["عکس", "snapshot", "screenshot", "photo"]) and has_any(["بگیر", "take", "capture", "save", "نگه دار"]):
            intents.append(IntentScore(intent=CommandIntent.TAKE_SNAPSHOT, confidence=0.88))

        if has_any(["ذخیره", "remember", "store", "save this", "یادت", "memory"]):
            intents.append(IntentScore(intent=CommandIntent.STORE_MEMORY, confidence=0.66))

        if has_any(["چی میشنوی", "what do you hear", "sound", "audio", "صدا"]):
            intents.append(IntentScore(intent=CommandIntent.WHAT_IS_HEARD, confidence=0.74))

        if has_any(["صداشون", "speaker", "talking person", "seen", "visible", "دیده"]):
            intents.append(IntentScore(intent=CommandIntent.SPEAKER_VISIBLE, confidence=0.62))

        if not intents:
            if "?" in t or has_any(["what", "why", "how", "where", "کی", "چی", "چرا", "چطور", "کجا"]):
                intents.append(IntentScore(intent=CommandIntent.QUESTION, confidence=0.55))
            else:
                intents.append(IntentScore(intent=CommandIntent.OTHER, confidence=0.40))
        return intents
