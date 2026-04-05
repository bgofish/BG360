# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

import lichtfeld as lf
from .bg360_panel import BG360Panel

_classes = [BG360Panel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("BG360 loaded")


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("BG360 unloaded")
