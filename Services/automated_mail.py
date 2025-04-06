
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import gradio as gr
from transformers import GPTNeoForCausalLM, GPT2Tokenizer


# Credentials
smtp_host = "smtp.gmail.com"
email_user = "spiderman78778@gmail.com"
email_pass = "sibj cffm rmub sish"

# Validate email credentials
if not email_user or not email_pass:
    raise ValueError("Missing EMAIL_USER or EMAIL_PASS in .env file.")

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # Model name
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Function to generate text using GPT-Neo
def generate_text_gpt_neo(prompt):
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=150, temperature=0.7, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        return f"Error in text generation: {str(e)}"

# Function to send an email
def send_email(to_email, subject, content):
    try:
        if not to_email or not subject or not content:
            return "Error: Missing required fields."

        with smtplib.SMTP_SSL(smtp_host, 465) as server:
            server.login(email_user, email_pass)
            msg = EmailMessage()
            msg["From"] = email_user
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.set_content(content)
            server.send_message(msg)
            return f"Email sent successfully to {to_email} with subject: {subject}"
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return f"Error sending email: {str(e)}"

# Gradio interface
def gradio_interface():
    def compose_and_send_email(subject_prompt, recipient):
        if not subject_prompt or not recipient:
            return "Error: Subject prompt and recipient email are required.", ""

        # Generate email content based on the prompt
        content = generate_text_gpt_neo(f"Compose a professional email with the subject: {subject_prompt}")
        subject = f"Generated Email: {subject_prompt[:50]}"  # Truncate subject for clarity

        # Send the email
        send_status = send_email(recipient, subject, content)
        return send_status, content

    with gr.Blocks() as interface:
        gr.Markdown("# AI-Powered Email Composer")
        subject_prompt = gr.Textbox(label="Prompt for Email Content", placeholder="Describe the email topic...")
        recipient_email = gr.Textbox(label="Recipient Email", placeholder="Enter recipient email")
        # Use interactive=False instead of readonly
        send_status = gr.Textbox(label="Status", interactive=False)
        generated_email = gr.Textbox(label="Generated Email Content", interactive=False)
        send_btn = gr.Button("Generate and Send Email")

        send_btn.click(
            fn=compose_and_send_email,
            inputs=[subject_prompt, recipient_email],
            outputs=[send_status, generated_email]
        )

    return interface

if __name__ == "__main__":
    interface = gradio_interface()
    interface.launch()

"""# API calls"""

import requests

HF_API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom-560m"

headers = {
    "Authorization": f"Bearer {huggingface_api_key}", # type: ignore
    "Content-Type": "application/json"
}
payload = {
    "inputs": "Test input to check API key access.",
    "parameters": {"max_new_tokens": 150, "temperature": 0.7}
}

response = requests.post(HF_API_URL, headers=headers, json=payload)
print("Status Code:", response.status_code)
print("Response:", response.text)

"""# Use this Main functions

// Sending Mails via Bot
"""

import gradio as gr
import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="pass.env")

# Email credentials
email_user = os.getenv("EMAIL_USER")
email_pass = os.getenv("EMAIL_PASS")
smtp_host = "smtp.gmail.com"

def send_email_with_attachments(to_email, subject, body_content, attachments=[]):
    """
    Function to send an email with the given subject, body content, and optional attachments.
    """
    try:
        # Create an email message
        msg = EmailMessage()
        msg["From"] = email_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body_content)

        # Add attachments if any
        for file_path in attachments:
            with open(file_path.name, "rb") as file:
                file_data = file.read()
                file_name = os.path.basename(file_path.name)
                msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)

        # Send the email
        with smtplib.SMTP_SSL(smtp_host, 465) as server:
            server.login(email_user, email_pass)
            server.send_message(msg)
            return f"Email sent successfully to {to_email}."
    except Exception as e:
        return f"Error sending email: {str(e)}"

# Gradio interface
def gradio_email_interface(to_email, subject, body_content, attachments):
    return send_email_with_attachments(to_email, subject, body_content, attachments)

# Create Gradio interface
interface = gr.Interface(
    fn=gradio_email_interface,
    inputs=[
        gr.Textbox(label="Recipient Email", placeholder="Enter recipient's email"),
        gr.Textbox(label="Subject", placeholder="Enter email subject"),
        gr.Textbox(label="Message Content", placeholder="Type your email content here..."),
        gr.Files(label="Attachments (optional)")
    ],
    outputs="text",
    title="Email Sender",
    description="Send emails with optional attachments using Gradio."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()

