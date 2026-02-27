
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from wordtolatex.generator import _map_font_family, _get_font_setting

def test_map_font_family():
    # Mock Path.exists to test both scenarios
    with patch("pathlib.Path.exists", autospec=True) as mock_exists:
        # 1. Test Carlito Present
        def side_effect(self):
            p = str(self)
            if "Carlito" in p: return True
            if "termes" in p: return False
            return False
        mock_exists.side_effect = side_effect
        
        assert _map_font_family("Calibri") == "Carlito"
        assert _map_font_family("Arial") == "Arial"
        assert _map_font_family("Times New Roman") == "Times New Roman"

        # 2. Test Carlito Missing
        mock_exists.side_effect = None
        mock_exists.return_value = False
        assert _map_font_family("Calibri") == "Arial" # Fallback

        # 3. Test Termes Present
        def side_effect_termes(self):
            p = str(self)
            if "termes" in p: return True
            return False
        mock_exists.side_effect = side_effect_termes
        assert _map_font_family("Times New Roman") == "TeX Gyre Termes"

def test_get_font_setting():
    # Test Carlito path generation
    with patch("pathlib.Path.exists", autospec=True) as mock_exists:
        mock_exists.return_value = True
        
        # Test Carlito
        setting = _get_font_setting("Carlito")
        print(f"DEBUG: Carlito Setting: {setting}")
        assert "\\setmainfont{Carlito}" in setting
        assert "Path=" in setting
        # The generator uses explicit checks on logical names too, so ensure we mock correctly
        
        # Test Termes
        setting = _get_font_setting("TeX Gyre Termes")
        assert "\\setmainfont{texgyretermes}" in setting
        
        # Test System Fallback (simulate file missing for Carlito logic inside _get_font_setting?)
        # Actually _get_font_setting checks file existence again.
        
        mock_exists.return_value = False
        # If file missing, it should fallback
        setting = _get_font_setting("Carlito")
        assert "\\setmainfont{Arial}" in setting
