"""
RAG Document Preprocessor

Supports:
✔ Paragraphs
✔ Tables
✔ FAQ
✔ Metadata
✔ Heading Detection
✔ Skips noisy headings
✔ Preserves document order

Author:
Pradyumn
"""

import json
import re
import uuid

from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P


INPUT_DOC = "HCLTech Leave Policy.docx"

OUTPUT_JSON = "data/processed/knowledge_base.json"

DOCUMENT_NAME = "HCLTech Leave Policy"


# ------------------------------------------------
# Iterate document preserving order
# ------------------------------------------------

def iter_block_items(parent):

    if isinstance(parent, _Document):

        parent_elm = parent.element.body

    else:

        parent_elm = parent._tc

    for child in parent_elm.iterchildren():

        if isinstance(child, CT_P):

            yield Paragraph(child, parent)

        elif isinstance(child, CT_Tbl):

            yield Table(child, parent)


# ------------------------------------------------
# Clean text
# ------------------------------------------------

def clean(text):

    text = text.replace("\xa0", " ")

    text = text.replace("•", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------------------
# Heading detection
# ------------------------------------------------

def is_heading(para):

    style = para.style.name.lower()

    txt = clean(para.text)

    if style.startswith("heading"):

        return True

    if txt.isupper():

        return True

    return False


# ------------------------------------------------
# Skip useless headings
# ------------------------------------------------

SKIP_HEADINGS = {

    "revision history",

    "annexure",

    "contact us",

    "disclaimers",

    "consequences"

}


# ------------------------------------------------
# Main
# ------------------------------------------------

doc = Document(INPUT_DOC)

records = []

current_section = "General"

faq_mode = False

faq_question = None

record_id = 1

for block in iter_block_items(doc):

    # ----------------------------------------
    # Paragraph
    # ----------------------------------------

    if isinstance(block, Paragraph):

        txt = clean(block.text)

        if not txt:

            continue

        if is_heading(block):

            current_section = txt

            faq_mode = txt.upper() == "FAQ"

            continue

        if current_section.lower() in SKIP_HEADINGS:

            continue

        # ---------------- FAQ ----------------

        if faq_mode:

            if re.match(r"^\d+\.", txt):

                faq_question = re.sub(r"^\d+\.\s*", "", txt)

            else:

                if faq_question:

                    records.append({

                        "id": f"DOC_{record_id:04d}",

                        "source": DOCUMENT_NAME,

                        "section": "FAQ",

                        "type": "faq",

                        "question": faq_question,

                        "answer": txt

                    })

                    record_id += 1

                    faq_question = None

            continue

        # Skip tiny headings

        if len(txt.split()) <= 3:

            continue

        records.append({

            "id": f"DOC_{record_id:04d}",

            "source": DOCUMENT_NAME,

            "section": current_section,

            "type": "paragraph",

            "content": txt

        })

        record_id += 1

    # ----------------------------------------
    # Table
    # ----------------------------------------

    elif isinstance(block, Table):

        rows = []

        for row in block.rows:

            cells = [clean(c.text) for c in row.cells]

            rows.append(cells)

        if len(rows) < 2:

            continue

        headers = rows[0]

        for row in rows[1:]:

            sentence = []

            for h, v in zip(headers, row):

                if v:

                    sentence.append(f"{h}: {v}")

            records.append({

                "id": f"DOC_{record_id:04d}",

                "source": DOCUMENT_NAME,

                "section": current_section,

                "type": "table",

                "content": ". ".join(sentence)

            })

            record_id += 1


# ---------------------------------------------
# Save
# ---------------------------------------------

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

    json.dump(records, f, indent=4, ensure_ascii=False)

print(f"\nGenerated {len(records)} knowledge records.")