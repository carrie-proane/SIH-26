#!/usr/bin/env python3
"""Build the three SIH technical handbooks from their maintained Markdown sources."""

from __future__ import annotations

import hashlib
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf"

NAVY = HexColor("#101828")
INK = HexColor("#243044")
MUTED = HexColor("#5F6C80")
LINE_COLOR = HexColor("#D9E2EC")
PALE = HexColor("#F4F7FA")
CYAN = HexColor("#00A6B2")
PURPLE = HexColor("#7657E8")
ORANGE = HexColor("#F79009")
RED = HexColor("#D92D20")
GREEN = HexColor("#039855")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]
    mono_candidates = [
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/Library/Fonts/Courier New.ttf"),
    ]
    regular = next((path for path in candidates if path.is_file()), None)
    bold = next((path for path in bold_candidates if path.is_file()), None)
    mono = next((path for path in mono_candidates if path.is_file()), None)
    if regular and bold:
        pdfmetrics.registerFont(TTFont("TraceSans", str(regular)))
        pdfmetrics.registerFont(TTFont("TraceSansBold", str(bold)))
        pdfmetrics.registerFontFamily(
            "TraceSans", normal="TraceSans", bold="TraceSansBold", italic="TraceSans"
        )
        body = "TraceSans"
        body_bold = "TraceSansBold"
    else:
        body = "Helvetica"
        body_bold = "Helvetica-Bold"
    if mono:
        pdfmetrics.registerFont(TTFont("TraceMono", str(mono)))
        mono_name = "TraceMono"
    else:
        mono_name = "Courier"
    return body, body_bold, mono_name


BODY_FONT, BOLD_FONT, MONO_FONT = register_fonts()


def ascii_safe(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2192": "->",
        "\u2190": "<-",
        "\u00b7": "/",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00b0": " degrees",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.encode("ascii", "replace").decode("ascii")


def inline_markup(raw: str) -> str:
    text = escape(ascii_safe(raw.strip()))
    text = re.sub(r"`([^`]+)`", rf'<font name="{MONO_FONT}" color="#344054">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(
        r"(?<![\"'=])(https?://[^\s<]+)",
        r'<link href="\1" color="#007C89">\1</link>',
        text,
    )
    return text


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "DocBody",
            parent=styles["BodyText"],
            fontName=BODY_FONT,
            fontSize=9.1,
            leading=13.0,
            textColor=INK,
            spaceAfter=5.5,
            allowWidows=0,
            allowOrphans=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "DocH1",
            parent=styles["Heading1"],
            fontName=BOLD_FONT,
            fontSize=17,
            leading=20,
            textColor=NAVY,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=False,
        )
    )
    styles.add(
        ParagraphStyle(
            "DocH2",
            parent=styles["Heading2"],
            fontName=BOLD_FONT,
            fontSize=12.3,
            leading=15.5,
            textColor=PURPLE,
            spaceBefore=10,
            spaceAfter=4.5,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "DocH3",
            parent=styles["Heading3"],
            fontName=BOLD_FONT,
            fontSize=10.1,
            leading=13,
            textColor=CYAN,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "DocBullet",
            parent=styles["DocBody"],
            leftIndent=2,
            firstLineIndent=0,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            "DocCode",
            parent=styles["Code"],
            fontName=MONO_FONT,
            fontSize=7.2,
            leading=9.3,
            textColor=HexColor("#E6EDF3"),
            backColor=NAVY,
            borderColor=NAVY,
            borderWidth=0.5,
            borderPadding=8,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "TOCHeading",
            fontName=BOLD_FONT,
            fontSize=22,
            leading=26,
            textColor=NAVY,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "Callout",
            parent=styles["DocBody"],
            fontName=BOLD_FONT,
            fontSize=10,
            leading=14,
            textColor=NAVY,
            backColor=HexColor("#EAF8F9"),
            borderColor=CYAN,
            borderWidth=1,
            borderPadding=10,
            spaceBefore=6,
            spaceAfter=10,
        )
    )
    return styles


STYLES = build_styles()


@dataclass(frozen=True)
class DocumentSpec:
    source: Path
    output: Path
    title: str
    subtitle: str
    label: str
    accent: colors.Color
    statement: str
    diagram: str


class TechnicalDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, spec: DocumentSpec):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=18 * mm,
            title=spec.title,
            author="Trace3D / SIH26158 team",
            subject=spec.subtitle,
        )
        self.spec = spec
        body_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=body_frame, onPage=self.draw_cover_page),
                PageTemplate(id="body", frames=body_frame, onPage=self.draw_body_page),
            ]
        )

    def draw_cover_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, A4[1] - 13 * mm, A4[0], 13 * mm, fill=1, stroke=0)
        canvas.setFillColor(self.spec.accent)
        canvas.rect(0, 0, 7 * mm, A4[1], fill=1, stroke=0)
        canvas.restoreState()

    def draw_body_page(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE_COLOR)
        canvas.setLineWidth(0.5)
        canvas.line(self.leftMargin, A4[1] - 13 * mm, A4[0] - self.rightMargin, A4[1] - 13 * mm)
        canvas.setFont(BOLD_FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, A4[1] - 9.5 * mm, ascii_safe(self.spec.label.upper()))
        canvas.setFont(BODY_FONT, 7.5)
        canvas.drawRightString(A4[0] - self.rightMargin, A4[1] - 9.5 * mm, "SIH26158 / TRACE3D")
        canvas.setStrokeColor(LINE_COLOR)
        canvas.line(self.leftMargin, 11 * mm, A4[0] - self.rightMargin, 11 * mm)
        canvas.setFont(BODY_FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, 7 * mm, "Audited repository documentation - 03 Sep 2026")
        canvas.drawRightString(A4[0] - self.rightMargin, 7 * mm, f"Page {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        level_map = {"DocH1": 0, "DocH2": 1, "DocH3": 2}
        level = level_map.get(flowable.style.name)
        if level is None:
            return
        text = flowable.getPlainText()
        digest = hashlib.sha1(f"{flowable.style.name}:{text}".encode()).hexdigest()[:12]
        key = f"section-{digest}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        self.notify("TOCEntry", (level, text, self.page, key))


def title_page(spec: DocumentSpec):
    title_style = ParagraphStyle(
        "CoverTitle",
        fontName=BOLD_FONT,
        fontSize=31,
        leading=35,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        fontName=BODY_FONT,
        fontSize=13,
        leading=18,
        textColor=MUTED,
        alignment=TA_LEFT,
    )
    label_style = ParagraphStyle(
        "CoverLabel",
        fontName=BOLD_FONT,
        fontSize=9,
        leading=11,
        textColor=WHITE,
        backColor=spec.accent,
        borderPadding=(5, 9, 5, 9),
    )
    meta_style = ParagraphStyle(
        "CoverMeta",
        fontName=BODY_FONT,
        fontSize=9,
        leading=14,
        textColor=MUTED,
    )
    return [
        Spacer(1, 26 * mm),
        Table([[Paragraph(ascii_safe(spec.label.upper()), label_style)]], hAlign="LEFT"),
        Spacer(1, 12 * mm),
        Paragraph(ascii_safe(spec.title), title_style),
        Paragraph(ascii_safe(spec.subtitle), subtitle_style),
        Spacer(1, 16 * mm),
        build_diagram(spec.diagram, spec.accent),
        Spacer(1, 13 * mm),
        Paragraph(inline_markup(spec.statement), STYLES["Callout"]),
        Spacer(1, 8 * mm),
        Paragraph(
            "Repository: github.com/carrie-proane/SIH-26<br/>"
            "Audited branch: main / commit 3088287<br/>"
            "Prepared: 03 September 2026 / Asia-Kolkata",
            meta_style,
        ),
        NextPageTemplate("body"),
        PageBreak(),
    ]


def arrow(drawing: Drawing, x1: float, y: float, x2: float, color):
    drawing.add(Line(x1, y, x2 - 5, y, strokeColor=color, strokeWidth=1.5))
    drawing.add(Polygon([x2 - 5, y - 3, x2, y, x2 - 5, y + 3], fillColor=color, strokeColor=color))


def build_diagram(kind: str, accent) -> Drawing:
    width = 171 * mm
    height = 44 * mm
    drawing = Drawing(width, height)
    if kind == "project":
        labels = ["INPUT", "PREPROCESS", "SFM / MVS", "REPORT", "VIEWER"]
        descriptions = ["video + GPS", "frames + masks", "observed 3D", "trust + quality", "inspect safely"]
    elif kind == "ai":
        labels = ["OBSERVED", "PARTIAL TSDF", "GENERATE K", "REPROJECT", "LABEL"]
        descriptions = ["SfM / MVS", "free / unknown", "SDF hypotheses", "reject violations", "visual only"]
    else:
        labels = ["CONCEPT", "WHY", "IMPLEMENT", "VERIFY", "DEFEND"]
        descriptions = ["definition", "tradeoff", "code path", "test / metric", "honest answer"]
    box_w = 30 * mm
    gap = 5.2 * mm
    start = 1 * mm
    y = 9 * mm
    for index, (label, description) in enumerate(zip(labels, descriptions, strict=True)):
        x = start + index * (box_w + gap)
        fill = accent if index in {0, len(labels) - 1} else PALE
        label_color = WHITE if index in {0, len(labels) - 1} else NAVY
        drawing.add(Rect(x, y, box_w, 25 * mm, rx=4, ry=4, fillColor=fill, strokeColor=accent, strokeWidth=1))
        drawing.add(String(x + box_w / 2, y + 15.5 * mm, label, textAnchor="middle", fontName=BOLD_FONT, fontSize=7.4, fillColor=label_color))
        drawing.add(String(x + box_w / 2, y + 9 * mm, description, textAnchor="middle", fontName=BODY_FONT, fontSize=6.5, fillColor=label_color))
        if index < len(labels) - 1:
            arrow(drawing, x + box_w, y + 12.5 * mm, x + box_w + gap, accent)
    return drawing


def toc_page(spec: DocumentSpec):
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC0",
            fontName=BOLD_FONT,
            fontSize=10.5,
            leading=14,
            textColor=NAVY,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
        ),
        ParagraphStyle(
            "TOC1",
            fontName=BODY_FONT,
            fontSize=8.8,
            leading=12,
            textColor=INK,
            leftIndent=10,
            firstLineIndent=0,
        ),
        ParagraphStyle(
            "TOC2",
            fontName=BODY_FONT,
            fontSize=8.1,
            leading=10.5,
            textColor=MUTED,
            leftIndent=20,
            firstLineIndent=0,
        ),
    ]
    return [
        Paragraph("Contents", STYLES["TOCHeading"]),
        Paragraph(
            inline_markup(
                "Use this document as a technical reference during implementation, review, and viva preparation."
            ),
            STYLES["DocBody"],
        ),
        Spacer(1, 5 * mm),
        toc,
        PageBreak(),
    ]


def paragraph_from_lines(lines: list[str]):
    raw = " ".join(line.strip() for line in lines)
    return Paragraph(inline_markup(raw), STYLES["DocBody"])


def make_bullets(items: list[str]):
    flowables = [
        ListItem(Paragraph(inline_markup(item), STYLES["DocBullet"]), leftIndent=8)
        for item in items
    ]
    return ListFlowable(
        flowables,
        bulletType="bullet",
        start="circle",
        leftIndent=14,
        bulletFontName=BODY_FONT,
        bulletFontSize=7,
        bulletColor=CYAN,
        spaceAfter=5,
    )


def make_code(lines: list[str]):
    safe_lines: list[str] = []
    for line in lines:
        line = ascii_safe(line.expandtabs(4))
        if len(line) <= 98:
            safe_lines.append(line)
        else:
            safe_lines.extend(textwrap.wrap(line, width=98, subsequent_indent="    "))
    return XPreformatted("\n".join(safe_lines), STYLES["DocCode"])


def make_table(rows: list[list[str]]):
    header = rows[0]
    body = rows[2:] if len(rows) > 1 and all(re.fullmatch(r"[:\- ]+", cell) for cell in rows[1]) else rows[1:]
    data = [
        [Paragraph(f"<b>{inline_markup(cell)}</b>", STYLES["DocBody"]) for cell in header]
    ]
    data.extend(
        [Paragraph(inline_markup(cell), STYLES["DocBody"]) for cell in row]
        for row in body
    )
    column_count = max(len(row) for row in data)
    for row in data:
        row.extend([Paragraph("", STYLES["DocBody"])] * (column_count - len(row)))
    if column_count == 2:
        widths = [43 * mm, 128 * mm]
    elif column_count == 3:
        widths = [47 * mm, 54 * mm, 70 * mm]
    else:
        widths = [171 * mm / column_count] * column_count
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), BOLD_FONT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE_COLOR),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    table.setStyle(TableStyle(commands))
    return table


def parse_markdown(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    story = []
    paragraph: list[str] = []
    bullets: list[str] = []
    code: list[str] = []
    in_code = False
    index = 0

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            story.append(paragraph_from_lines(paragraph))
            paragraph = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            story.append(make_bullets(bullets))
            bullets = []

    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            flush_paragraph()
            flush_bullets()
            if in_code:
                story.append(make_code(code))
                code = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code.append(line)
            index += 1
            continue
        if line.startswith("# "):
            flush_paragraph()
            flush_bullets()
            index += 1
            continue
        heading = re.match(r"^(##|###|####)\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_bullets()
            level = len(heading.group(1)) - 1
            style = STYLES[{1: "DocH1", 2: "DocH2", 3: "DocH3"}[level]]
            story.append(Paragraph(inline_markup(heading.group(2)), style))
            index += 1
            continue
        if line.startswith("| ") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            flush_paragraph()
            flush_bullets()
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in table_lines]
            story.extend([make_table(rows), Spacer(1, 3 * mm)])
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", line)
        if bullet:
            flush_paragraph()
            bullets.append(bullet.group(1))
            index += 1
            continue
        if not line.strip():
            flush_paragraph()
            flush_bullets()
            index += 1
            continue
        if bullets:
            bullets[-1] += " " + line.strip()
        else:
            paragraph.append(line)
        index += 1
    flush_paragraph()
    flush_bullets()
    if code:
        story.append(make_code(code))
    return story


def build(spec: DocumentSpec):
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    document = TechnicalDocTemplate(str(spec.output), spec)
    story = [*title_page(spec), *toc_page(spec), *parse_markdown(spec.source)]
    document.multiBuild(story)


def main() -> int:
    specs = [
        DocumentSpec(
            source=ROOT / "docs" / "architecture" / "project-overview.md",
            output=OUTPUT / "SIH26_Trace3D_Project_Technical_Overview.pdf",
            title="Trace3D Project Technical Overview",
            subtitle="End-to-end workflow, file ownership, runtime contracts, libraries, tests, and limitations",
            label="Architecture handbook",
            accent=CYAN,
            statement=(
                "Outcome: a complete, interview-ready explanation of how video and telemetry become "
                "declared 3D artifacts and how the reorganized frontend/backend remain connected."
            ),
            diagram="project",
        ),
        DocumentSpec(
            source=ROOT / "docs" / "ai" / "surface-completion-blueprint.md",
            output=OUTPUT / "SIH26_AI_Occluded_Surface_Completion_Blueprint.pdf",
            title="AI Occluded-Surface Completion Blueprint",
            subtitle="Problem formulation, model architecture, data, training losses, evaluation, deployment, and risk controls",
            label="AI research and delivery plan",
            accent=PURPLE,
            statement=(
                "Scientific boundary: a camera cannot prove a surface it never observed. The model "
                "generates plausible hypotheses, and every generated region remains visual-only."
            ),
            diagram="ai",
        ),
        DocumentSpec(
            source=ROOT / "docs" / "viva" / "technical-question-bank.md",
            output=OUTPUT / "SIH26_Trace3D_Technical_Viva_Question_Bank.pdf",
            title="Trace3D Technical Viva Question Bank",
            subtitle="Seventy-five detailed questions and defensible answers across vision, AI, backend, frontend, testing, and deployment",
            label="Interview and judge preparation",
            accent=ORANGE,
            statement=(
                "Answer pattern: define the concept, connect it to the exact code path, state the "
                "tradeoff, and never claim stronger evidence than the data supports."
            ),
            diagram="viva",
        ),
    ]
    for spec in specs:
        build(spec)
        print(spec.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
