
import os

def merge_files(file1, file2, output):
    print(f"Merging {file1} and {file2} into {output}...")
    
    try:
        with open(file1, 'r', encoding='utf-8') as f1:
            content1 = f1.read()
            
        with open(file2, 'r', encoding='utf-8') as f2:
            content2 = f2.read()
            
        with open(output, 'w', encoding='utf-8') as out:
            out.write(content1)
            out.write("\n\n") # Separator
            out.write(content2)
            
        print("Success! File saved with UTF-8 encoding.")
        
    except UnicodeDecodeError:
        print("Error: Could not decode one of the files as UTF-8. Trying fallback...")
        # Fallback debug
        try:
             with open(file1, 'r', encoding='cp1251') as f1: content1 = f1.read()
             print(f"{file1} seems to be CP1251")
        except: pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    merge_files("optimized_knowledge.txt", "541.txt", "knowledge_full.txt")
