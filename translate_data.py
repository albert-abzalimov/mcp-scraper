
import json
from langdetect import detect
from deep_translator import GoogleTranslator

# Load JSON
with open('details.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def translate_field(text):
    if not text:
        return text, None
    try:
        lang = detect(text)
        if lang != 'en':
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated, lang
        return text, 'en'
    except Exception:
        return text, None


count = 0
i = 0
reminder = 100
# Update each MCP and tool with translated fields and language flags
for mcp in data:
    mcp['MCP_description'], mcp['MCP_description_lang'] = translate_field(mcp.get('MCP_description', ''))
    for tool in mcp.get('tools', []):
        i += 1
        if (i >= reminder):
            print(str(count) + ": still going, went through another " + str(reminder) + " tools")
            i = 0
            count += 1
        tool['description'], tool['description_lang'] = translate_field(tool.get('description', ''))

# Save the translated JSON
with open('translated_details.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Translation complete. Saved to 'translated_details.json'.")
