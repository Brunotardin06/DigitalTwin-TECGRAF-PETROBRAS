"""Entrypoint da interface grafica (Flet).

    python app.py

Se o cliente desktop do Flet nao subir (ex.: falta libmpv.so.1 no sistema),
rode em modo web e abra http://localhost:8550 no navegador:

    FLET_VIEW=web python app.py
"""

import os

import flet as ft

from src.ui.app import main

if __name__ == "__main__":
    if os.environ.get("FLET_VIEW", "desktop").lower() == "web":
        ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=8550)
    else:
        ft.app(target=main)
