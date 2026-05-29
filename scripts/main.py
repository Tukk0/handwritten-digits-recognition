"""Telegram bot for MNIST digit recognition via Triton Inference Server."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests


def _get_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if token is None:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var first.")
    return token


def _get_triton_endpoint() -> str:
    return os.getenv(
        "TRITON_ENDPOINT",
        "http://triton:8000/v2/models/digit_recognition/infer",
    )


def _preprocess(photo_path: str) -> np.ndarray:
    img = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image from {photo_path}")
    img = cv2.resize(img, (28, 28))
    return img.astype(np.float32) / 255.0


def _run_bot() -> None:
    from telebot import TeleBot

    bot = TeleBot(_get_token())
    triton_endpoint = _get_triton_endpoint()

    @bot.message_handler(content_types=["photo"])
    def handle_photo(message):
        file_info = bot.get_file(message.photo[-1].file_id)
        raw = bot.download_file(file_info.file_path)
        tmp_path = Path(tempfile.mkdtemp()) / "digit.png"
        tmp_path.write_bytes(raw)
        img = _preprocess(str(tmp_path))
        payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 28, 28],
                    "datatype": "FP32",
                    "data": img.flatten().tolist(),
                }
            ]
        }
        response = requests.post(triton_endpoint, json=payload, timeout=10)
        data = response.json()
        probs = np.array(data["outputs"][0]["data"]).reshape(10)
        label = int(np.argmax(probs))
        confidence = float(probs[label])
        bot.send_message(
            message.chat.id,
            f"Digit: {label}  (confidence: {confidence:.0%})",
            parse_mode="HTML",
        )

    @bot.message_handler(content_types=["text"])
    def handle_text(message):
        text = message.text.strip().lower()
        if text == "/start":
            bot.send_message(
                message.chat.id,
                "Start the bot and send a photo of a handwritten digit!",
                parse_mode="HTML",
            )
        elif text == "/help":
            bot.send_message(
                message.chat.id,
                "/help — show this message",
                parse_mode="HTML",
            )
        else:
            bot.send_message(message.chat.id, "Send a photo of a handwritten digit.")

    print("Starting Telegram bot...")
    bot.infinity_polling()


if __name__ == "__main__":
    _run_bot()
