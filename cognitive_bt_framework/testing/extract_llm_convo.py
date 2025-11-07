import json
import html
import xml.dom.minidom

# Function to load conversation data from a properly formatted JSON file
def load_conversation_data(filepath):
    with open(filepath, 'r') as file:
        conversation_data = json.load(file)  # Directly load the JSON data
    return conversation_data

# Function to pretty-print XML content
def pretty_print_xml(xml_content):
    try:
        # Parse and pretty-print the XML content
        dom = xml.dom.minidom.parseString(xml_content)
        return dom.toprettyxml(indent="  ")  # Pretty print with an indentation of 2 spaces
    except Exception as e:
        # If the XML is not valid, just return the original content
        return xml_content

# Function to escape special characters in code-like text, like XML
def escape_special_chars(text):
    return html.escape(text)  # This will convert <, >, &, and others into HTML-safe characters

# Function to handle escaping and pretty-printing for both user and LLM content
def process_content(content):
    # Check if the content looks like XML by detecting the presence of < and >
    if "<" in content and ">" in content:
        # Try to pretty-print the XML
        pretty_xml = content
        return escape_special_chars(pretty_xml)
    else:
        # No XML detected, just escape special characters
        return escape_special_chars(content)

# Function to generate an HTML file to show conversation flow, ensuring code and XML is properly formatted
def generate_html(conversation_data, output_filepath):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>APOLLO Lab Yale: Zero-Knowledge Task Planning</title>
        <header>Task: Bring a mug of coffee to the table.</header>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f9;
            }
            .conversation-container {
                width: 100%;
                max-width: 800px;
                margin: auto;
            }
            .user-prompt, .llm-response, .system-instruction {
                margin-bottom: 20px;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .system-instruction {
                background-color: #e0f7fa;
                white-space: pre-wrap;
            }
            .user-prompt {
                background-color: #d0ebff;
                white-space: pre-wrap;
            }
            .llm-response {
                background-color: #fff1e6;
                white-space: pre-wrap;
            }
            .system-instruction pre, .user-prompt pre, .llm-response pre {
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 8px;
                white-space: pre-wrap;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <div class="conversation-container">
    """

    # Go through each conversation entry in the JSON
    for i, entry in enumerate(conversation_data):
        # Each entry is a list of messages
        for message in entry:
            role = message['role']
            content = message['content']

            # Process the content based on the role
            processed_content = content
            if type(content) == str:
                processed_content = process_content(content)


            if role == 'system':
                # Add system instruction to the HTML
                html_content += f"""
                    <div class="system-instruction">
                        <strong>System Instruction:</strong><br>
                        <pre>{processed_content}</pre>
                    </div>
                """
            elif role == 'user':
                # Add user prompt to the HTML
                html_content += f"""
                    <div class="user-prompt">
                        <strong>Prompt :</strong><br>
                        <pre>{processed_content}</pre>
                    </div>
                """
            elif role == 'llm':
                # Add LLM response to the HTML
                html_content += f"""
                    <div class="llm-response">
                        <strong>LLM Response :</strong><br>
                        <pre>{processed_content}</pre>
                    </div>
                """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Write HTML content to file
    with open(output_filepath, 'w') as html_file:
        html_file.write(html_content)
    print(f"HTML file created at {output_filepath}")

# Main function to load and parse conversation data from file and generate HTML
def main(filepath, output_filepath):
    conversation_data = load_conversation_data(filepath)
    generate_html(conversation_data, output_filepath)

# Call the main function with your conversation JSON file path and output HTML file path
conversation_file_path = 'conversation.json'  # Replace with your file path
output_html_path = '/home/liam/dev/ZeroKnowledgeTaskPlanning-Website/index.html'  # Replace with your output file path
main(conversation_file_path, output_html_path)
