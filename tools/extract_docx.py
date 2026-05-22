
import json
import re
import sys
from docx import Document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

def iter_block_items(parent):
    """
    Generate a reference to each paragraph and table child within *parent*,
    in document order. Each returned value is an instance of either Table or
    Paragraph. *parent* would most commonly be a reference to a main
    Document object, but also works for a _Cell object.
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("something's not right")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def clean_text(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def extract_data(doc_path):
    document = Document(doc_path)
    
    data = []
    current_category = "Unknown"
    current_group = "General"
    
    # Regex to detect category headers
    # E.g. "I toifa", "II toifa", "yuqori darajada xavfli", etc.
    cat_pattern = re.compile(r"(I|II|III|IV)\s*toifa|yuqori|o'rtacha|past|mahalliy", re.IGNORECASE)
    
    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            text = clean_text(block.text)
            if not text:
                continue
            
            # Check for category indicators
            if "I toifa" in text or "II toifa" in text or "III toifa" in text or "IV toifa" in text:
                current_category = text
                print(f"Found Category Header: {current_category}")
            elif "yuqori" in text.lower() and "xavfli" in text.lower():
                 # Fallback if roman numeral not explicit but text matches descriptions
                 current_category = text
                 
        elif isinstance(block, Table):
            print(f"Processing Table under category: {current_category}")
            
            # Process rows
            # Skip header row (assume row 0 is header)
            
            headers = ["T/r", "Faoliyat", "Muddat", "To'lov"]
            
            for i, row in enumerate(block.rows):
                cells = row.cells
                row_text = [clean_text(c.text) for c in cells]
                
                # Heuristic: Header row usually has "Faoliyat" or similar
                if "Faoliyat" in row_text[1] or "T/p" in row_text[0]:
                    continue
                
                # Check for Group Header (merged cells spanning whole row)
                # If set(row_text) has 1 unique non-empty value, or very few unique cells match
                unique_cell_objects = []
                for c in cells:
                    if c not in unique_cell_objects:
                        unique_cell_objects.append(c)
                
                if len(unique_cell_objects) <= 2 and len(clean_text(cells[-1].text)) > 5 and not any(char.isdigit() for char in clean_text(cells[-1].text)):
                     # Likely a group header like "1. Transport..." 
                     # But be careful, sometimes last col is empty cost/time? 
                     # Actually, group headers usually have text in first cell and merged across.
                     if len(clean_text(cells[0].text)) > 5:
                         current_group = clean_text(cells[0].text)
                         print(f"  Found Group: {current_group}")
                         continue
                
                # Data Row
                # Standard map: 0=Tr, 1=Activity, 2=Time, 3=Cost
                # But handling merges:
                # If cells[0] is same object as previous row's cells[0], tr is continued (rare for tr)
                # If cells[1] is continued, activity is same (rare)
                # If cells[2] or [3] is continued, time/cost is same.
                
                tr = row_text[0]
                activity = row_text[1]
                time_val = row_text[2] if len(row_text) > 2 else ""
                cost_val = row_text[3] if len(row_text) > 3 else ""
                
                # Refine Time/Cost extraction
                # Sometimes user puts 4 cols, sometimes 5.
                # Let's assume the last column is Cost (BXM) and second to last is Time.
                # If only 3 cols? Check content.
                
                # Validation: Time is usually integer, Cost is float/int.
                # Activity is long text.
                
                if not activity and not tr:
                    continue
                    
                # If it's a group header masquerading as a row
                if not time_val and not cost_val and len(activity) > 20:
                    current_group = activity
                    continue

                item = {
                    "category_full": current_category,
                    "group": current_group,
                    "tr": tr,
                    "activity": activity,
                    "time": time_val,
                    "cost": cost_val
                }
                data.append(item)
    
    return data

if __name__ == "__main__":
    path = "илова.docx"
    try:
        data = extract_data(path)
        
        # Save JSON
        with open("extracted_knowledge.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        # Save Optimized Text for RAG
        with open("optimized_knowledge.txt", "w", encoding="utf-8") as f:
            for item in data:
                # Create a dense, keyword-rich block
                
                # Clean up category to just I/II/III/IV if possible
                cat_simplified = "Noma'lum"
                if "I" in item['category_full'] and "toifa" in item['category_full']: cat_simplified = "I (Birinchi)"
                if "II" in item['category_full'] and "toifa" in item['category_full']: cat_simplified = "II (Ikkinchi)"
                if "III" in item['category_full'] and "toifa" in item['category_full']: cat_simplified = "III (Uchinchi)"
                if "IV" in item['category_full'] and "toifa" in item['category_full']: cat_simplified = "IV (To'rtinchi)"
                
                block = (
                    f"## OBYEKT/FAOLIYAT: {item['activity']}\n"
                    f"- **Kategoriya/Toifa**: {cat_simplified} toifa\n"
                    f"- **Batafsil guruh**: {item['group']}\n"
                    f"- **Ekspertiza muddati**: {item['time']} ish kuni\n"
                    f"- **To'lov miqdori (BHM)**: {item['cost']} baravari\n"
                    f"- **Tartib raqami**: {item['tr']}\n"
                    f"- **Manba**: 541-sonli qaror Ilovasi (Jadval)\n"
                    f"\n"  # Separator
                )
                f.write(block)
                
        print(f"Extracted {len(data)} items.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

