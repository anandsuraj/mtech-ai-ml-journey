# PDF Conversion Instructions

## Files to Convert to PDF

You need to convert the following 3 Markdown files to PDF format for final submission:

1. **implementation_report.md** → implementation_report.pdf
2. **task_b_quality_improvement.md** → task_b_quality_improvement.pdf  
3. **literature_survey.md** → literature_survey.pdf

---

## Method 1: Using Pandoc (Recommended - Best Quality)

### Install Pandoc (if not installed)
```bash
# macOS
brew install pandoc
brew install basictex  # For PDF support
```

### Convert Files
```bash
cd /Applications/MAMP/htdocs/bits-course/semester-3/nlp-applications/assignment/assignment2

# Convert implementation report
pandoc implementation_report.md -o implementation_report.pdf --pdf-engine=xelatex

# Convert Task B
pandoc task_b_quality_improvement.md -o task_b_quality_improvement.pdf --pdf-engine=xelatex

# Convert literature survey
pandoc literature_survey.md -o literature_survey.pdf --pdf-engine=xelatex
```

---

## Method 2: Using VS Code Markdown PDF Extension

1. Open VS Code
2. Install "Markdown PDF" extension
3. Open each .md file
4. Right-click → "Markdown PDF: Export (pdf)"
5. Saves PDF in same directory

---

## Method 3: Using Typora (If You Have It)

1. Open Typora
2. Open each .md file
3. File → Export → PDF
4. Save with same name

---

## Method 4: Online Converter (Quick & Easy)

1. Go to https://www.markdowntopdf.com/
2. Upload each .md file
3. Click "Convert to PDF"
4. Download the PDF

---

## Verify PDFs

After conversion, check that:
- ✅ All sections are present
- ✅ Tables are properly formatted
- ✅ Code blocks are readable
- ✅ Headings are correct size/hierarchy  
- ✅ References/citations are included

---

## Final File Structure for Submission

```
assignment2_submission/
├── code/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── script.js
│
└── documentation/
    ├── implementation_report.pdf      ← Convert from .md
    ├── task_b_quality_improvement.pdf ← Convert from .md
    └── literature_survey.pdf          ← Convert from .md
```

---

## Quick Pandoc Command (All 3 at Once)

```bash
cd /Applications/MAMP/htdocs/bits-course/semester-3/nlp-applications/assignment/assignment2

for file in implementation_report task_b_quality_improvement literature_survey; do
    pandoc "${file}.md" -o "${file}.pdf" --pdf-engine=xelatex
done
```

This will create all three PDFs automatically!
