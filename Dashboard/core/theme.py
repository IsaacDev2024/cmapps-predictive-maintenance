"""Utilidades relacionadas con la detección del tema activo de Streamlit."""

from __future__ import annotations

import re
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components


_SCRIPT = """
<script>
const getThemeFromStyles = () => {
    try {
        const root = window.parent.document.documentElement;
        const styles = window.parent.getComputedStyle(root);
        const color = styles.getPropertyValue('--background-color');
        const matches = color.match(/\d+\.\d+|\d+/g);
        const rgb = matches ? matches.slice(0, 3).map(parseFloat) : [15,17,23];
        const luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
        return luminance > 0.5 ? 'light' : 'dark';
    } catch (error) {
        return null;
    }
};

const sendTheme = () => {
    const theme = getThemeFromStyles();
    const fallback = window.parent?.Streamlit?.getComponentValue?.() || null;
    const value = theme || fallback || 'dark';
    if (window.parent?.Streamlit) {
        window.parent.Streamlit.setComponentValue(value);
    }
};

const root = window.parent.document.documentElement;

if (!window.parent.__streamlitThemeObserver__) {
    const observer = new MutationObserver(() => {
        sendTheme();
    });
    observer.observe(root, { attributes: true, attributeFilter: ['class', 'data-theme'] });
    window.parent.__streamlitThemeObserver__ = observer;
}

sendTheme();
</script>
"""


def _infer_from_color(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    match_hex = re.fullmatch(r"#?([0-9a-fA-F]{6})", value.strip())
    if match_hex:
        hex_value = match_hex.group(1)
        rgb = tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))
    else:
        matches = re.findall(r"\d+", value)
        if len(matches) < 3:
            return None
        rgb = tuple(int(matches[i]) for i in range(3))

    luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
    return "light" if luminance > 0.5 else "dark"


def _detect_from_dom() -> Optional[str]:
    value = components.html(_SCRIPT, height=0)
    if isinstance(value, str) and value in {"light", "dark"}:
        return value
    return None


def get_current_theme() -> str:
    """Obtiene el tema activo ("light" | "dark") usando la configuración y el DOM."""

    if 'app_theme' in st.session_state:
        return st.session_state['app_theme']

    detected = _detect_from_dom()
    if detected:
        st.session_state['app_theme'] = detected
        return detected

    base_theme = st.get_option("theme.base")
    if base_theme in {"light", "dark"}:
        st.session_state['app_theme'] = base_theme
        return base_theme

    bg_color = st.get_option("theme.backgroundColor")
    inferred = _infer_from_color(bg_color)
    if inferred:
        st.session_state['app_theme'] = inferred
        return inferred

    text_color = st.get_option("theme.textColor")
    inferred = _infer_from_color(text_color)
    if inferred:
        st.session_state['app_theme'] = "dark" if inferred == "light" else "light"
        return st.session_state['app_theme']

    st.session_state['app_theme'] = "dark"
    return "dark"
