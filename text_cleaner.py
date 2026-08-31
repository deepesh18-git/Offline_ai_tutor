import re


def clean_text(raw_text):

    text = raw_text

    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)

    # Normalize multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove empty lines
    lines = text.split('\n')
    cleaned_lines = [
        line.strip() for line in lines
        if line.strip()
    ]
    text = '\n'.join(cleaned_lines)

    # Remove special characters
    text = re.sub(
        r'[^\w\s\n\.\,\!\?\;\:\-\(\)\[\]\"\'\%\/\+\=]',
        '', text
    )

    # Fix hyphenated line breaks from PDF
    text = re.sub(r'-\n', '', text)

    return text