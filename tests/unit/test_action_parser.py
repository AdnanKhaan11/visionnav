"""Unit tests — action parser."""
import pytest
from visionnav.actions.parser import ActionParseError, parse_action
from visionnav.actions.schema import ActionType


def test_parse_click():
    out = '<action>{"type":"click","coordinates":[0.5,0.3],"description":"OK"}</action>'
    a   = parse_action(out)
    assert a.type == ActionType.CLICK
    assert a.coordinates == (0.5, 0.3)

def test_parse_type():
    a = parse_action('<action>{"type":"type","text":"hello"}</action>')
    assert a.type == ActionType.TYPE
    assert a.text == "hello"

def test_parse_done():
    a = parse_action('<action>{"type":"done","description":"fin"}</action>')
    assert a.type == ActionType.DONE

def test_missing_action_block():
    with pytest.raises(ActionParseError, match="No <action>"):
        parse_action("just text, no block")

def test_invalid_json():
    with pytest.raises(ActionParseError, match="Invalid JSON"):
        parse_action("<action>not json</action>")

def test_unknown_type():
    with pytest.raises(ActionParseError, match="Unknown action type"):
        parse_action('<action>{"type":"fly"}</action>')

def test_coords_out_of_range():
    with pytest.raises(ActionParseError, match="out of"):
        parse_action('<action>{"type":"click","coordinates":[1.5,0.5]}</action>')
