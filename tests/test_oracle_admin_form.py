"""Tests for the admin oracle form parser (_form_to_oracle), specifically the
day-of-week checkbox handling: the picker drives days_of_week, and 'all' or
'none' selected both collapse to None (every day) so an oracle can't be made
un-runnable by clearing it."""

from src.admin.blueprint import _form_to_oracle


class _Form:
    """Minimal Werkzeug-MultiDict stand-in: .get() for singles, .getlist() for
    the repeated `days` checkboxes."""

    def __init__(self, single=None, multi=None):
        self._single = single or {}
        self._multi = multi or {}

    def get(self, key, default=None):
        return self._single.get(key, default)

    def getlist(self, key):
        return list(self._multi.get(key, []))


def test_form_parses_sun_thu_days():
    form = _Form(
        single={"kind": "wsb_digest", "schedule_kind": "clock", "clock_time": "20:00"},
        multi={"days": ["6", "0", "1", "2", "3"]},
    )
    o = _form_to_oracle(form, context_id=1)
    assert o.days_of_week == frozenset({6, 0, 1, 2, 3})
    assert o.weekdays_only is False
    assert o.effective_days() == frozenset({6, 0, 1, 2, 3})


def test_form_all_seven_days_means_every_day():
    form = _Form(multi={"days": ["0", "1", "2", "3", "4", "5", "6"]})
    assert _form_to_oracle(form, context_id=1).days_of_week is None


def test_form_no_days_means_every_day():
    assert _form_to_oracle(_Form(), context_id=1).days_of_week is None


def test_form_ignores_garbage_day_values():
    form = _Form(multi={"days": ["0", "x", "9", "4", ""]})
    assert _form_to_oracle(form, context_id=1).days_of_week == frozenset({0, 4})
