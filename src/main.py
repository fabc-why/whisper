"""Entry point for speech recognition and command publishing."""

from __future__ import annotations

import logging

from listening import Listener
from ros2_send import Ros2Sender
from semantic import interpret


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    listener = Listener()
    sender = Ros2Sender()
    sender.connect()
    try:
        for text, meaning in listener.listen(interpret):
            LOGGER.info("Recognized command %s from %r", meaning, text)
            sender.send(text, meaning)
    except KeyboardInterrupt:
        LOGGER.info("Stopping speech recognition")
    finally:
        sender.close()


if __name__ == "__main__":
    main()