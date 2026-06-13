import re
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def clean_special_chars(text: str) -> str:
    """Normalizes smart quotes, dashes, and filters out non-BMP emoji characters to prevent ReportLab rendering errors."""
    if not text:
        return ""
    replacements = {
        '\u201c': '"',
        '\u201d': '"',
        '\u2018': "'",
        '\u2019': "'",
        '\u2013': '-',
        '\u2014': '--',
        '\u2022': '*',
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    # Filter out characters outside BMP (Basic Multilingual Plane) like emojis (ord > 0xFFFF)
    return "".join(c for c in text if ord(c) <= 0xFFFF)

def clean_md(text: str) -> str:
    """Escapes HTML symbols for ReportLab Paragraph and converts bold, italic, and code tags."""
    text = clean_special_chars(text)
    # Escape HTML special chars
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Bold **text** -> <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic *text* -> <i>text</i> or _text_ -> <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
    # Code `text` -> <font face="Courier">text</font>
    text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
    return text

def markdown_to_flowables(text: str, styles) -> list:
    """Parses markdown text into ReportLab Flowables (headings and paragraphs only)."""
    flowables = []
    if not text:
        return flowables
        
    lines = text.split('\n')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Heading 1 (# title)
        if stripped.startswith('# '):
            flowables.append(Paragraph(clean_md(stripped[2:]), styles['Heading1']))
            flowables.append(Spacer(1, 8))
        # Heading 2 (## title)
        elif stripped.startswith('## '):
            flowables.append(Paragraph(clean_md(stripped[3:]), styles['Heading2']))
            flowables.append(Spacer(1, 6))
        # Heading 3 (### title)
        elif stripped.startswith('### '):
            flowables.append(Paragraph(clean_md(stripped[4:]), styles['Heading3']))
            flowables.append(Spacer(1, 4))
        # Bullet items (simplifying to paragraph with bullet character)
        elif stripped.startswith('- ') or stripped.startswith('* '):
            flowables.append(Paragraph(f"&bull; {clean_md(stripped[2:])}", styles['Normal']))
            flowables.append(Spacer(1, 4))
        else:
            # Paragraph
            flowables.append(Paragraph(clean_md(stripped), styles['Normal']))
            flowables.append(Spacer(1, 6))
            
    return flowables

def get_pdf_styles():
    """Returns sample stylesheet customized with ResearchMind colors."""
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#1A365D") # Deep navy
    text_color = colors.HexColor("#2D3748")    # Charcoal
    accent_color = colors.HexColor("#4A5568")  # Slate gray
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=accent_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_color,
        spaceAfter=6
    )
    
    title_style = ParagraphStyle(
        'CustomTitle',
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        alignment=1, # Centered
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=14,
        textColor=accent_color,
        alignment=1, # Centered
        spaceAfter=20
    )

    return {
        'Title': title_style,
        'Subtitle': subtitle_style,
        'Heading1': h1_style,
        'Heading2': h2_style,
        'Heading3': h3_style,
        'Normal': normal_style,
        'BodyText': normal_style
    }

def add_header_footer(canvas, doc):
    """Draws running header and footer on each page."""
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor("#718096"))
    
    # Running header
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, doc.height + doc.bottomMargin + 10, doc.leftMargin + doc.width, doc.height + doc.bottomMargin + 10)
    canvas.drawString(doc.leftMargin, doc.height + doc.bottomMargin + 15, "ResearchMind session report")
    
    # Running footer
    canvas.line(doc.leftMargin, doc.bottomMargin - 10, doc.leftMargin + doc.width, doc.bottomMargin - 10)
    canvas.drawString(doc.leftMargin, doc.bottomMargin - 22, f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    page_num = canvas.getPageNumber()
    canvas.drawRightString(doc.leftMargin + doc.width, doc.bottomMargin - 22, f"Page {page_num}")
    canvas.restoreState()

def generate_chat_report(chat_history) -> BytesIO:
    """Generates PDF report of conversation Q&A history with timestamps."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = get_pdf_styles()
    story = []
    
    story.append(Paragraph("Conversation Q&A Report", styles['Title']))
    story.append(Paragraph("Exported history of interactions with research assistant", styles['Subtitle']))
    story.append(Spacer(1, 15))
    
    if not chat_history:
        story.append(Paragraph("No conversation history available in the current session.", styles['Normal']))
    else:
        for i, item in enumerate(chat_history, 1):
            q_text = item.get("question", "")
            a_text = item.get("answer", "")
            timestamp = item.get("timestamp", "")
            
            # Question block
            time_str = f" ({timestamp})" if timestamp else ""
            story.append(Paragraph(f"Question {i}{time_str}", styles['Heading2']))
            story.append(Paragraph(q_text, styles['Normal']))
            story.append(Spacer(1, 4))
            
            # Answer block
            story.append(Paragraph("Answer", styles['Heading3']))
            ans_flowables = markdown_to_flowables(a_text, styles)
            story.extend(ans_flowables)
            
            # Section divider line
            if i < len(chat_history):
                story.append(Spacer(1, 10))
                divider = Table([[""]], colWidths=[doc.width], rowHeights=[0.5])
                divider.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#E2E8F0")),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                    ('TOPPADDING', (0,0), (-1,-1), 0),
                ]))
                story.append(divider)
                story.append(Spacer(1, 10))
                
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    buffer.seek(0)
    return buffer

def generate_summary_report(summary_data) -> BytesIO:
    """Generates PDF report containing generated paper summaries (Abstract, Methodology, Key Findings, Limitations)."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = get_pdf_styles()
    story = []
    
    story.append(Paragraph("Research Paper Summarization Report", styles['Title']))
    story.append(Paragraph("Targeted section-by-section analytical summary", styles['Subtitle']))
    story.append(Spacer(1, 15))
    
    sections = [
        ("abstract", "Abstract Summary"),
        ("methodology", "Methodology Summary"),
        ("findings", "Key Findings"),
        ("limitations", "Limitations")
    ]
    
    has_content = False
    for key, title in sections:
        content = summary_data.get(key)
        if content:
            has_content = True
            story.append(Paragraph(title, styles['Heading1']))
            story.extend(markdown_to_flowables(content, styles))
            story.append(Spacer(1, 15))
            
    if not has_content:
        story.append(Paragraph("No summaries have been generated in this session yet.", styles['Normal']))
        
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    buffer.seek(0)
    return buffer

def generate_quiz_report(quiz_data) -> BytesIO:
    """Generates PDF report of multiple-choice quiz questions and answers."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = get_pdf_styles()
    story = []
    
    story.append(Paragraph("Research Paper Quiz Report", styles['Title']))
    story.append(Paragraph("Multiple-choice assessment to evaluate reading comprehension", styles['Subtitle']))
    story.append(Spacer(1, 15))
    
    if not quiz_data:
        story.append(Paragraph("No quiz questions have been generated in this session yet.", styles['Normal']))
    else:
        for i, mcq in enumerate(quiz_data, 1):
            q_text = mcq.get("question", "")
            options = mcq.get("options", {})
            correct_ans = mcq.get("correct_answer", {})
            explanation = mcq.get("explanation", "")
            
            # Question text
            story.append(Paragraph(f"Question {i}: {q_text}", styles['Heading2']))
            story.append(Spacer(1, 4))
            
            # MCQ Options
            for letter_opt, text in options.items():
                story.append(Paragraph(f"<b>{letter_opt}.</b> {text}", styles['Normal']))
                story.append(Spacer(1, 2))
                
            story.append(Spacer(1, 4))
            
            # Answer + Explanation
            ans_opt = correct_ans.get("option", "")
            ans_txt = correct_ans.get("text", "")
            story.append(Paragraph(f"<b>Correct Answer:</b> Option {ans_opt} ({ans_txt})", styles['Normal']))
            story.append(Paragraph(f"<i>Explanation:</i> {explanation}", styles['Normal']))
            story.append(Spacer(1, 12))
            
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    buffer.seek(0)
    return buffer

def generate_comparison_report(comparison_text, papers=None, dimension=None) -> BytesIO:
    """Generates PDF report comparing research papers side-by-side."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = get_pdf_styles()
    story = []
    
    story.append(Paragraph("Multi-PDF Comparison Report", styles['Title']))
    story.append(Paragraph("Comparative analysis report generated by ResearchMind", styles['Subtitle']))
    story.append(Spacer(1, 15))
    
    if papers:
        papers_str = ", ".join(papers)
        story.append(Paragraph(f"<b>Compared Papers:</b> {papers_str}", styles['Normal']))
    if dimension:
        story.append(Paragraph(f"<b>Comparison Dimension:</b> {dimension}", styles['Normal']))
    story.append(Spacer(1, 10))
    
    if not comparison_text:
        story.append(Paragraph("No comparison analysis report has been generated in this session yet.", styles['Normal']))
    else:
        story.extend(markdown_to_flowables(comparison_text, styles))
        
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    buffer.seek(0)
    return buffer

def generate_complete_report(chat_history, summary_data, quiz_data, comparison_text, comparison_meta=None) -> BytesIO:
    """Generates a combined PDF session report containing summaries, Q&A logs, comparisons, and quiz questions."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = get_pdf_styles()
    story = []
    
    # Cover / Header
    story.append(Paragraph("ResearchMind Complete Session Report", styles['Title']))
    story.append(Paragraph("Comprehensive research assistant exports summary", styles['Subtitle']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("This comprehensive document combines all summaries, conversational Q&A history, quiz questions, and comparative analysis reports generated during this active research session.", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Cover Divider
    cover_line = Table([[""]], colWidths=[doc.width], rowHeights=[1])
    cover_line.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1A365D")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(cover_line)
    story.append(PageBreak())
    
    section_index = 1
    
    # 1. Summaries
    has_summaries = any(summary_data.get(k) for k in ["abstract", "methodology", "findings", "limitations"]) if summary_data else False
    if has_summaries:
        story.append(Paragraph(f"{section_index}. Research Paper Summaries", styles['Heading1']))
        story.append(Spacer(1, 10))
        
        sections = [
            ("abstract", "Abstract Summary"),
            ("methodology", "Methodology Summary"),
            ("findings", "Key Findings"),
            ("limitations", "Limitations")
        ]
        
        for key, title in sections:
            content = summary_data.get(key)
            if content:
                story.append(Paragraph(title, styles['Heading2']))
                story.extend(markdown_to_flowables(content, styles))
                story.append(Spacer(1, 10))
                
        section_index += 1
        story.append(PageBreak())
        
    # 2. Q&A History
    if chat_history:
        story.append(Paragraph(f"{section_index}. Conversation Q&A History", styles['Heading1']))
        story.append(Spacer(1, 10))
        
        for i, item in enumerate(chat_history, 1):
            q_text = item.get("question", "")
            a_text = item.get("answer", "")
            timestamp = item.get("timestamp", "")
            
            time_str = f" ({timestamp})" if timestamp else ""
            story.append(Paragraph(f"Question {i}{time_str}", styles['Heading2']))
            story.append(Paragraph(q_text, styles['Normal']))
            story.append(Spacer(1, 4))
            
            story.append(Paragraph("Answer", styles['Heading3']))
            story.extend(markdown_to_flowables(a_text, styles))
            
            if i < len(chat_history):
                story.append(Spacer(1, 8))
                divider = Table([[""]], colWidths=[doc.width], rowHeights=[0.5])
                divider.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#E2E8F0")),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                    ('TOPPADDING', (0,0), (-1,-1), 0),
                ]))
                story.append(divider)
                story.append(Spacer(1, 8))
                
        section_index += 1
        story.append(PageBreak())
        
    # 3. Multi-PDF Comparison
    if comparison_text:
        story.append(Paragraph(f"{section_index}. Multi-PDF Comparison", styles['Heading1']))
        story.append(Spacer(1, 10))
        
        if comparison_meta:
            papers = comparison_meta.get("papers")
            dimension = comparison_meta.get("dimension")
            if papers:
                story.append(Paragraph(f"<b>Compared Papers:</b> {', '.join(papers)}", styles['Normal']))
            if dimension:
                story.append(Paragraph(f"<b>Comparison Dimension:</b> {dimension}", styles['Normal']))
            story.append(Spacer(1, 8))
            
        story.extend(markdown_to_flowables(comparison_text, styles))
        
        section_index += 1
        story.append(PageBreak())
        
    # 4. Quiz
    if quiz_data:
        story.append(Paragraph(f"{section_index}. Research Quiz Assessment", styles['Heading1']))
        story.append(Spacer(1, 10))
        
        for i, mcq in enumerate(quiz_data, 1):
            q_text = mcq.get("question", "")
            options = mcq.get("options", {})
            correct_ans = mcq.get("correct_answer", {})
            explanation = mcq.get("explanation", "")
            
            story.append(Paragraph(f"Question {i}: {q_text}", styles['Heading2']))
            story.append(Spacer(1, 4))
            
            for letter_opt, text in options.items():
                story.append(Paragraph(f"<b>{letter_opt}.</b> {text}", styles['Normal']))
                story.append(Spacer(1, 2))
                
            story.append(Spacer(1, 4))
            
            ans_opt = correct_ans.get("option", "")
            ans_txt = correct_ans.get("text", "")
            story.append(Paragraph(f"<b>Correct Answer:</b> Option {ans_opt} ({ans_txt})", styles['Normal']))
            story.append(Paragraph(f"<i>Explanation:</i> {explanation}", styles['Normal']))
            story.append(Spacer(1, 12))
            
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    buffer.seek(0)
    return buffer
