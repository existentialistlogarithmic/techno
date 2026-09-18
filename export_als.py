#!/usr/bin/env python3
"""Build an Ableton Live Set (.als) of the arrangement.

    python3 export_als.py [out.als]

An .als is gzipped XML with a large, version-specific schema, so this does
not invent one: it loads a real Live 11 set as a template, clones one of its
MIDI tracks per part, and replaces the names and the notes. Every Id in the
document is then renumbered so the clones do not collide.
"""

import copy
import gzip
import sys
import xml.etree.ElementTree as ET

from export_arrangement import TRACKS, filename_for, build_events

TEMPLATE = "template/reference.als"
PPQ_BEATS = 4.0          # beats per bar


def load_template(path=TEMPLATE):
    return ET.fromstring(gzip.open(path, "rb").read())


def find_track_template(root):
    """A MidiTrack that already contains an arrangement clip."""
    for t in root.find("LiveSet/Tracks").findall("MidiTrack"):
        if t.find("DeviceChain/MainSequencer/ClipTimeable/ArrangerAutomation/"
                  "Events/MidiClip") is not None:
            return t
    raise SystemExit("no usable MidiTrack in the template")


def set_val(parent, tag, value):
    el = parent.find(tag)
    if el is not None:
        el.set("Value", str(value))


def make_clip(clip_tmpl, name, notes, length_beats):
    """notes: [(start_beat, dur_beat, key, vel), ...]"""
    clip = copy.deepcopy(clip_tmpl)
    clip.set("Time", "0")
    set_val(clip, "CurrentStart", 0)
    set_val(clip, "CurrentEnd", length_beats)
    set_val(clip, "Name", name)
    loop = clip.find("Loop")
    for tag, v in (("LoopStart", 0), ("LoopEnd", length_beats),
                   ("OutMarker", length_beats), ("HiddenLoopStart", 0),
                   ("HiddenLoopEnd", length_beats), ("StartRelative", 0)):
        set_val(loop, tag, v)
    set_val(loop, "LoopOn", "false")

    kts = clip.find("Notes/KeyTracks")
    proto = copy.deepcopy(kts.find("KeyTrack"))
    for kt in list(kts):
        kts.remove(kt)

    by_key = {}
    for start, dur, key, vel in notes:
        by_key.setdefault(int(key), []).append((start, dur, vel))

    nid = 1
    for i, key in enumerate(sorted(by_key)):
        kt = copy.deepcopy(proto)
        kt.set("Id", str(i))
        kt.find("MidiKey").set("Value", str(key))
        nn = kt.find("Notes")
        for ev in list(nn):
            nn.remove(ev)
        for start, dur, vel in sorted(by_key[key]):
            ET.SubElement(nn, "MidiNoteEvent", {
                "Time": f"{start:.6g}", "Duration": f"{max(dur, 0.0625):.6g}",
                "Velocity": str(max(1, min(127, int(vel)))),
                "VelocityDeviation": "0", "OffVelocity": "64",
                "Probability": "1", "IsEnabled": "true", "NoteId": str(nid)})
            nid += 1
        kts.append(kt)
    return clip


def renumber_ids(root, start=1000):
    """Cloned tracks carry cloned Ids. Live uses these as a global namespace,
    so every one is reassigned and NextPointeeId is pushed past them."""
    n = start
    for el in root.iter():
        if el.get("Id") is not None and el.tag not in ("KeyTrack",):
            el.set("Id", str(n))
            n += 1
    set_val(root.find("LiveSet"), "NextPointeeId", n + 1000)
    return n


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "CONCRETE_CATHEDRAL.als"
    print("reading the arrangement...")
    per_track, total_bars = build_events()
    length = total_bars * PPQ_BEATS

    print("loading the template Live Set...")
    root = load_template()
    tracks_el = root.find("LiveSet/Tracks")
    tmpl_track = copy.deepcopy(find_track_template(root))
    clip_tmpl = copy.deepcopy(tmpl_track.find(
        "DeviceChain/MainSequencer/ClipTimeable/ArrangerAutomation/Events/MidiClip"))
    returns = [copy.deepcopy(t) for t in tracks_el.findall("ReturnTrack")]

    for t in list(tracks_el):
        tracks_el.remove(t)

    made = 0
    for name, notes in per_track:
        if not notes:
            continue
        t = copy.deepcopy(tmpl_track)
        nm = t.find("Name")
        for tag in ("EffectiveName", "UserName"):
            set_val(nm, tag, name)
        set_val(nm, "MemorizedFirstClipName", name)
        # fresh track: no automation, no devices, no session clips
        env = t.find("AutomationEnvelopes/Envelopes")
        if env is not None:
            for e in list(env):
                env.remove(e)
        devs = t.find("DeviceChain/DeviceChain/Devices")
        if devs is not None:
            for d in list(devs):
                devs.remove(d)
        for slot in t.findall("DeviceChain/MainSequencer/ClipSlotList/ClipSlot"):
            inner = slot.find("ClipSlot")
            if inner is not None:
                for c in list(inner):
                    if c.tag.endswith("Clip"):
                        inner.remove(c)
        events = t.find("DeviceChain/MainSequencer/ClipTimeable/"
                        "ArrangerAutomation/Events")
        for c in list(events):
            events.remove(c)
        events.append(make_clip(clip_tmpl, name, notes, length))
        tracks_el.append(t)
        made += 1

    for r in returns:
        tracks_el.append(r)

    mt = root.find("LiveSet/MasterTrack")
    for tempo in mt.iter("Tempo"):
        set_val(tempo, "Manual", 150)
    set_val(mt.find("Name"), "EffectiveName", "Master")

    last = renumber_ids(root)
    xml = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
    with gzip.open(out, "wb") as f:
        f.write(xml)
    print(f"wrote {out}: {made} midi tracks, {len(returns)} returns, "
          f"{length:.0f} beats ({total_bars} bars), ids up to {last}")


if __name__ == "__main__":
    main()
