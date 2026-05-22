
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
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    elif isinstance(parent, _Row):
        parent_elm = parent._tr
    else:
        raise ValueError("something's not right")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def clean_text(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()

def normalize_category(cat_text):
    cat_text = cat_text.lower()
    if "i toifa" in cat_text and "ii" not in cat_text and "iii" not in cat_text and "iv" not in cat_text:
        return "I (Yuqori xavfli)"
    if "ii toifa" in cat_text or ("o'rtacha" in cat_text and "xavfli" in cat_text):
        return "II (O'rtacha xavfli)"
    if "iii toifa" in cat_text or ("past" in cat_text and "xavfli" in cat_text):
        return "III (Past xavfli)"
    if "iv toifa" in cat_text or ("mahalliy" in cat_text and "ta'sir" in cat_text):
        return "IV (Mahalliy ta'sir)"
    return cat_text

def extract_data(doc_path):
    document = Document(doc_path)
    data = []
    
    # Initialize state
    current_category = "Noma'lum"
    current_group = "Umumiy"
    
    # Pre-scan paragraphs to see if title is at top
    for p in document.paragraphs:
        t = clean_text(p.text)
        if "I toifa" in t or "yuqori" in t.lower():
            if "ro'yxat" not in t.lower(): # Avoid title of doc
                current_category = normalize_category(t)
                break

    # Iterate block items to catch headers between tables or inside
    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            text = clean_text(block.text)
            if not text: continue
            
            # Detect category switch
            if re.search(r"(I|II|III|IV)\s*toifa", text, re.IGNORECASE):
                current_category = normalize_category(text)
                print(f"  [Header] Category Switch: {current_category}")
        
        elif isinstance(block, Table):
            print(f"Processing Table... Current Cat: {current_category}")
            for row in block.rows:
                cells = row.cells
                row_text = [clean_text(c.text) for c in cells]
                full_row_text = " ".join(row_text)
                
                # Check for category switch inside table (merged row)
                if re.search(r"(I|II|III|IV)\s*toifa", full_row_text, re.IGNORECASE):
                     # Likely a category header row
                     current_category = normalize_category(full_row_text)
                     print(f"  [Row] Category Switch: {current_category}")
                     continue

                # Header skip
                if "Faoliyat" in row_text[1] or "Т/р" in row_text[0]:
                    continue

                # Group detection
                # Heuristic: First cell is long text, others empty or digits
                # Or merged cells (cells[0] is same object as cells[-1]?)
                # Simplified: check unique texts
                unique_texts = sorted(list(set(row_text)))
                if len(unique_texts) <= 2 and len(full_row_text) > 10:
                     # Check if it looks like a group title "1. Transport..."
                     if "1." in row_text[0] or "2." in row_text[0] or "3." in row_text[0]: 
                         # But wait, data rows obey this too?
                         # Data rows usually have a number in col 0, text in col 1, number in col 2, number in col 3.
                         # Group rows usually have text in col 0/1 and nothing in col 3, or col 3 is same as col 0.
                         pass

                     # Better detection for group:
                     # If the text matches "X. Group Name"
                     if re.match(r"^\d+\.\s+\w+", full_row_text):
                          # Confirmed group?
                          # But item "1.1." is data.
                          # Item "1. Transport..." is group.
                          if re.match(r"^\d+\.\s+[^\d]", full_row_text): # 1. Word...
                           # Use the first unique non-empty text as group name
                           group_name = unique_texts[0] if unique_texts else full_row_text
                           current_group = group_name
                           print(f"    [Group] {current_group}")
                           continue

                # Data extraction
                tr = row_text[0]
                activity = row_text[1]
                
                # Fix for merged activity cells that might spawn multiple rows?
                # Usually python-docx repeats text.
                
                # Time/Cost
                # Sometimes 4 cols, sometimes 5.
                # Usually: Tr | Activity | Time | Cost
                # But sometimes: Tr | Activity | ... | Time | Cost
                time_val = row_text[-2] if len(row_text) >= 3 else ""
                cost_val = row_text[-1] if len(row_text) >= 3 else ""
                
                # Validation
                if not tr or not activity: continue
                if len(tr) > 10: continue # Likely a group header described in tr column
                
                # Clean up activity
                if activity == current_group: continue 
                
                item = {
                    "category": current_category,
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
        
        # Post-process to fix initial items if they missed category
        # If first items are "Noma'lum" but follow "I (Yuqori...)" logic (Cost 32.5), fix them.
        for item in data:
            if item['category'] == "Noma'lum":
                if item['cost'] == "32,5" or item['cost'] == "32.5":
                    item['category'] = "I (Yuqori xavfli)"
                elif item['cost'] == "19,5" or item['cost'] == "19.5":
                    item['category'] = "II (O'rtacha xavfli)"
        
        # Save JSON
        with open("extracted_knowledge.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        # Save Optimized Text
        with open("optimized_knowledge.txt", "w", encoding="utf-8") as f:
            for item in data:
                block = (
                    f"## OBYEKT/FAOLIYAT: {item['activity']}\n"
                    f"- **Kategoriya**: {item['category']}\n"
                    f"- **Batafsil guruh**: {item['group']}\n"
                    f"- **Ekspertiza muddati**: {item['time']} ish kuni\n"
                    f"- **To'lov miqdori (BHM)**: {item['cost']} baravari\n"
                    f"- **Tartib raqami**: {item['tr']}\n"
                    f"- **Manba**: 541-sonli qaror Ilovasi\n"
                    f"\n"
                )
                f.write(block)
                
        print(f"Extracted {len(data)} items.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
