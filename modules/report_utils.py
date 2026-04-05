# modules/report_utils.py
import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_fairness_report(
    output_path: str,
    dataset_name: str,
    target_col: str,
    sensitive_feature: str,
    model_name: str,
    fairness_before: dict,
    fairness_after: dict,
    accuracy_before: float,
    accuracy_after: float,
):
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor("#2E86C1"),
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor("#34495E"),
        spaceBefore=15,
        spaceAfter=10,
        borderPadding=5,
        backColor=colors.HexColor("#EAEDED")
    )
    
    normal_style = styles["Normal"]
    
    story = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Title
    story.append(Paragraph("BiasScope Fairness Audit Report", title_style))
    story.append(Paragraph(f"<b>Date Generated:</b> {timestamp}", normal_style))
    story.append(Spacer(1, 20))

    # 1. Audit Meta-Data
    story.append(Paragraph("1. Audit Configuration", heading_style))
    meta_data = [
        ["Dataset Name:", dataset_name],
        ["Target Column:", target_col],
        ["Sensitive Feature:", sensitive_feature],
        ["Base Model Type:", model_name]
    ]
    t_meta = Table(meta_data, colWidths=[150, 300], hAlign='LEFT')
    t_meta.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.darkslategray),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 20))

    # 2. High-Level Summary
    story.append(Paragraph("2. Global Model Performance & Fairness", heading_style))
    
    dp_before = fairness_before.get("Demographic Parity Difference", 0)
    dp_after = fairness_after.get("Demographic Parity Difference", 0)
    eo_before = fairness_before.get("Equal Opportunity Difference", 0)
    eo_after = fairness_after.get("Equal Opportunity Difference", 0)

    summary_data = [
        ["Metric", "Original Model", "Mitigated Model", "Change"],
        [
            "Overall Accuracy", 
            f"{accuracy_before:.4f}", 
            f"{accuracy_after:.4f}", 
            f"{(accuracy_after - accuracy_before):+.4f}"
        ],
        [
            "Demographic Parity Diff.", 
            f"{dp_before:.4f}", 
            f"{dp_after:.4f}", 
            f"{(dp_after - dp_before):+.4f}"
        ],
        [
            "Equal Opportunity Diff.", 
            f"{eo_before:.4f}", 
            f"{eo_after:.4f}", 
            f"{(eo_after - eo_before):+.4f}"
        ]
    ]
    
    t_summary = Table(summary_data, colWidths=[180, 100, 100, 100], hAlign='LEFT')
    t_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E86C1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>* Note: For fairness metrics (DP and EO Difference), lower values approaching 0.0 indicate a fairer model.</i>", styles["Italic"]))
    story.append(Spacer(1, 20))

    # 3. Subgroup Breakdown
    story.append(Paragraph("3. Intersectional Subgroup Breakdown", heading_style))
    story.append(Paragraph("Detailed performance metrics for the protected groups identified during the audit.", normal_style))
    story.append(Spacer(1, 10))

    # Extract subgroup groups
    groups_before = list(fairness_before.get("Group Accuracy", {}).keys())
    
    if groups_before:
        subgroup_data = [["Subgroup", "Original Accuracy", "Mitigated Accuracy", "Original Sel. Rate", "Mitigated Sel. Rate"]]
        
        for g in groups_before:
            acc_b = fairness_before.get("Group Accuracy", {}).get(g, 0)
            acc_a = fairness_after.get("Group Accuracy", {}).get(g, 0)
            sel_b = fairness_before.get("Group Selection Rate", {}).get(g, 0)
            sel_a = fairness_after.get("Group Selection Rate", {}).get(g, 0)
            
            subgroup_data.append([
                str(g),
                f"{acc_b:.3f}",
                f"{acc_a:.3f}",
                f"{sel_b:.3f}",
                f"{sel_a:.3f}"
            ])
            
        t_sub = Table(subgroup_data, colWidths=[130, 90, 90, 90, 90], hAlign='LEFT')
        t_sub.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7F8C8D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_sub)
    else:
        story.append(Paragraph("No specific subgroup data was logged for this run.", normal_style))

    story.append(Spacer(1, 20))

    # 4. Final Conclusion
    story.append(Paragraph("4. Executive Summary", heading_style))
    
    fairness_change = dp_before - dp_after
    accuracy_drop = accuracy_before - accuracy_after
    
    if fairness_change > 0.05:
        summary_text = f"The mitigation successfully reduced the Demographic Parity disparity by <b>{fairness_change:.4f}</b>. "
        if accuracy_drop > 0.02:
            summary_text += f"This was achieved with an accuracy trade-off of -{accuracy_drop:.2%}."
        else:
            summary_text += "This was achieved with minimal impact on overall accuracy."
    elif fairness_change <= 0 and dp_before < 0.05:
        summary_text = "The original model already exhibited high fairness (Demographic Parity Difference < 0.05). Mitigation was largely unnecessary or resulted in minimal structural changes."
    else:
        summary_text = "Mitigation resulted in limited fairness improvements. Please review the GridSearch Pareto frontier to explore stricter fairness constraints."

    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 30))

    # Footer
    story.append(Paragraph("<font size=8 color='grey'>Generated automatically by BiasScope – An Interactive Framework for AI Fairness Discovery & Mitigation.</font>", normal_style))

    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=50, bottomMargin=50)
    doc.build(story)
    return output_path
