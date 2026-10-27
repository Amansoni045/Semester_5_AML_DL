import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def generate_pdf():
    html_path = REPO_ROOT / "reports" / "phase1_report_print.html"
    pdf_path = REPO_ROOT / "reports" / "phase1_report.pdf"

    cmd = [
        CHROME_BIN,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        str(html_path)
    ]

    print("Running headless Chrome to generate PDF...")
    subprocess.run(cmd, check=True)
    print(f"✓ PDF report successfully generated at: {pdf_path}")

    # Also generate PDF of the presentation slides!
    slides_html = REPO_ROOT / "reports" / "phase1_presentation_slides.html"
    slides_pdf = REPO_ROOT / "reports" / "phase1_presentation.pdf"
    cmd_slides = [
        CHROME_BIN,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={slides_pdf}",
        str(slides_html)
    ]
    print("Running headless Chrome to generate presentation PDF...")
    subprocess.run(cmd_slides, check=True)
    print(f"✓ Presentation PDF successfully generated at: {slides_pdf}")

if __name__ == "__main__":
    generate_pdf()
