"""
PDF Export Utility for Plotly Figures
Provides functions to export Plotly graphs to PDF for reports
"""

import os
from datetime import datetime


def export_figure_to_pdf(fig, filename=None, output_dir="exports"):
    """
    Export a Plotly figure to PDF
    
    Args:
        fig: Plotly figure object
        filename: Optional custom filename (without extension)
        output_dir: Directory to save PDF (default: "exports")
    
    Returns:
        str: Path to saved PDF file
    """
    try:
        import plotly.io as pio
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"topic_analysis_{timestamp}"
        
        # Ensure .pdf extension
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        
        filepath = os.path.join(output_dir, filename)
        
        # Export to PDF
        fig.write_image(filepath, format="pdf", width=1200, height=1000)
        
        print(f"✓ PDF exported successfully: {filepath}")
        return filepath
        
    except ImportError:
        print("⚠️  kaleido not installed. Run: pip install kaleido")
        return None
    except Exception as e:
        print(f"❌ PDF export failed: {e}")
        return None


def export_all_figures_to_pdf(figures_dict, base_filename="report"):
    """
    Export multiple figures to separate PDF files
    
    Args:
        figures_dict: Dict of {name: figure}
        base_filename: Base name for files
    
    Returns:
        list: Paths to all exported PDFs
    """
    exported_files = []
    
    for name, fig in figures_dict.items():
        # Clean name for filename
        clean_name = name.lower().replace(" ", "_").replace("/", "_")
        filename = f"{base_filename}_{clean_name}"
        
        filepath = export_figure_to_pdf(fig, filename)
        if filepath:
            exported_files.append(filepath)
    
    return exported_files


def add_export_button_to_html(html_path):
    """
    Add export button to temporary Plotly HTML file
    Injects JavaScript to enable PDF download directly from browser
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Add export button and JS code
        export_code = """
        <style>
            #export-btn {
                position: fixed;
                top: 10px;
                right: 10px;
                z-index: 999;
                background: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
            }
            #export-btn:hover {
                background: #2980b9;
            }
        </style>
        <button id="export-btn" onclick="exportToPDF()">Save as PDF</button>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <script>
            function exportToPDF() {
                const element = document.body;
                const opt = {
                    margin: 0.5,
                    filename: 'topic_analysis_' + new Date().toISOString().slice(0,10) + '.pdf',
                    image: { type: 'jpeg', quality: 0.98 },
                    html2canvas: { scale: 2 },
                    jsPDF: { unit: 'in', format: 'letter', orientation: 'landscape' }
                };
                html2pdf().set(opt).from(element).save();
            }
        </script>
        """
        
        # Insert before </body>
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', export_code + '</body>')
        else:
            html_content += export_code
        
        # Write back
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return True
    except Exception as e:
        print(f"Failed to add export button: {e}")
        return False
