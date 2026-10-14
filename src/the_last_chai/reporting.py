"""Generate a print-friendly HTML report from the investigation's real state."""

from html import escape
from typing import Dict, List


def generate_report(state: Dict[str, object]) -> str:
    completed = state.get("completed", [])
    results = state.get("results", {})
    accusation = state.get("accusation", {})
    rows = "".join("<li>{}</li>".format(escape(str(item))) for item in completed)
    details = []
    for mission, result in results.items():
        details.append("<h3>{}</h3>".format(escape(str(mission))))
        for key, value in result.items():
            if key in ("image", "crop", "difference", "annotated") and str(value).startswith("data:image/"):
                details.append('<figure><img src="{}" alt="{}"><figcaption>{}</figcaption></figure>'.format(escape(value, quote=True), escape(str(key)), escape(str(key))))
            elif key == "video" and str(value).startswith("/"):
                details.append('<p>Annotated video output: <a href="{}">open video</a></p>'.format(escape(value, quote=True)))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                details.append("<p><b>{}:</b> {}</p>".format(escape(str(key)), escape(str(value))))
            elif key not in ("boxes", "crossing"):
                details.append("<p><b>{}:</b> {}</p>".format(escape(str(key)), escape(str(value))))
        if result.get("crossing") is not None:
            details.append("<p><b>Observed crossing:</b> {}</p>".format(escape(str(result["crossing"]))))
        if result.get("boxes"):
            details.append("<p><b>Detected candidate boxes:</b> {}</p>".format(escape(str(result["boxes"]))))
    details = "".join(details)
    conclusion = "Submitted: {}".format(escape(str(accusation))) if accusation else "No final accusation was submitted."
    return """<!doctype html><html><head><meta charset="utf-8"><title>The Last Chai | Case Report</title><style>body{font:16px Georgia,serif;max-width:850px;margin:48px auto;color:#222;line-height:1.6}h1,h2{font-family:Arial,sans-serif;border-bottom:1px solid #aaa;padding-bottom:8px}h3{font-family:Arial,sans-serif}figure{margin:16px 0}img{max-width:100%;max-height:460px;object-fit:contain;border:1px solid #aaa}figcaption{font:12px monospace;color:#555}.note{border-left:4px solid #a33;padding:8px 16px;background:#f5eeee}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media print{body{margin:16mm}.grid{break-inside:avoid}}</style></head><body><h1>THE LAST CHAI / CASE REPORT</h1><p><b>Case ID:</b> ARG-2342 &nbsp; <b>Simulation:</b> Fictional educational exercise</p><h2>Incident summary</h2><p>Fictional laboratory incident; prototype and mobile phone are listed missing. Findings below reflect only operations actually recorded in this browser investigation.</p><h2>Persons of interest</h2><ul><li>Aarav, research assistant; owns the red backpack according to the case file.</li><li>Kabir, laboratory technician; canonical fictional culprit.</li><li>Meera, research student; partial witness account.</li><li>Ramesh, security guard; account conflicts with the staged sequence.</li></ul><h2>Investigation state</h2><p>Completed missions:</p><ul>""" + rows + """</ul><h2>Observed operation results</h2>""" + (details or "<p>No processing outputs were recorded.</p>") + """<h2>Reconstructed scripted timeline</h2><ol><li>23:35: Multiple people are present in or near the lab.</li><li>23:41: Professor Rao is last seen alive in staged footage.</li><li>23:42: Scripted incident window.</li><li>23:43: Kabir removes the prototype using Aarav's backpack, per the fictional case record.</li><li>23:44: Prepared fictional log records corruption-script execution.</li></ol><h2>Final accusation</h2><p>""" + conclusion + """</p><h2>Limitations and method</h2><p>OpenCV enhancement processes recorded pixels; HSV segmentation locates red candidates; the tracker observes candidate movement. None of these algorithms identifies a person or determines ownership. The owner and carrier identities come from the fictional case file. The system log is a separate prepared story asset. Generic object detection requires optional model weights; scene comparison is not model inference. The staged timestamp does not independently prove a time of death.</p><p class="note">All characters, evidence and timestamps are fictional. Visual algorithms can be uncertain and should not be treated as forensic proof.</p><p>Browser print is available: use Print / Save as PDF.</p></body></html>"""