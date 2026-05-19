#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for malformed-MusicXML / extreme-ppq edge cases.

Each of these used to crash with a cryptic stack trace deep in
partitura or mido:

  * empty <part-list/>  →  ValueError: need at least one array to concatenate
                           (numpy, inside get_ppq's np.concatenate)
  * <chord> on first note of a part
                        →  bare AssertionError with no message in _handle_note
  * very high ppq (lcm of many tuplet denominators)
                        →  struct.error inside mido at write time
"""

import tempfile
import unittest
import warnings
from pathlib import Path

import partitura as pt
import partitura.score as score
from partitura.io.exportmidi import get_ppq


_EMPTY_PARTLIST_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="3.0">
  <part-list />
</score-partwise>
"""


_CHORD_ON_FIRST_NOTE_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="3.0">
  <part-list>
    <score-part id="P1"><part-name>P</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time>
      </attributes>
      <note>
        <chord/>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""


class TestMidiExportRobustness(unittest.TestCase):
    def test_empty_partlist_raises_clear_value_error(self):
        with tempfile.TemporaryDirectory() as td:
            xml_path = Path(td) / "empty.musicxml"
            xml_path.write_text(_EMPTY_PARTLIST_XML)
            s = pt.load_musicxml(str(xml_path))
            self.assertEqual(len(s), 0)
            with self.assertRaises(ValueError) as cm:
                pt.save_score_midi(s, str(Path(td) / "out.mid"))
            # Caller should see a useful message, not a raw numpy error.
            self.assertIn("no parts", str(cm.exception).lower())

    def test_get_ppq_empty_parts_returns_safe_default(self):
        # save_score_midi raises before calling get_ppq on an empty score, so
        # this guards the helper's own no-parts branch directly: returns a
        # representable header value (480) without crashing in np.concatenate.
        self.assertEqual(get_ppq([]), 480)

    def test_ppq_overflow_is_capped_with_warning(self):
        """Synthesize a part whose quarter_duration exceeds the SMF 16-bit
        ticks-per-beat ceiling so get_ppq must cap and warn."""
        ppq_in = 40000  # > 32767 SMF max
        part = score.Part("P0", "p", quarter_duration=ppq_in)
        part.add(score.TimeSignature(4, 4), start=0)
        part.add(score.Note(step="C", octave=4, alter=None), start=0, end=ppq_in)
        score.add_measures(part)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.mid"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pt.save_score_midi(part, str(out))
            self.assertTrue(out.exists() and out.stat().st_size > 0)
            self.assertTrue(
                any("ppq" in str(item.message).lower() for item in w),
                f"expected a ppq-cap RuntimeWarning, got {[str(i.message) for i in w]}",
            )


class TestMusicXMLImportRobustness(unittest.TestCase):
    def test_chord_on_first_note_warns_and_continues(self):
        with tempfile.TemporaryDirectory() as td:
            xml_path = Path(td) / "chord_first.musicxml"
            xml_path.write_text(_CHORD_ON_FIRST_NOTE_XML)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                s = pt.load_musicxml(str(xml_path))
            self.assertEqual(len(s), 1)
            na = s[0].note_array()
            self.assertEqual(len(na), 2)
            # ensure we surfaced a warning (rather than the old bare assert)
            self.assertTrue(
                any("chord" in str(item.message).lower() for item in w),
                f"expected a chord-without-prev warning, got {[str(i.message) for i in w]}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
