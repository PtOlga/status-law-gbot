# URLs for knowledge base creation
URLS = [
     "https://status.law",
    "https://status.law/about",
    "https://status.law/careers",  
    "https://status.law/tariffs-for-services-of-protection-against-extradition",
    "https://status.law/challenging-sanctions",
    "https://status.law/law-firm-contact-legal-protection", 
    "https://status.law/cross-border-banking-legal-issues", 
    "https://status.law/extradition-defense", 
    "https://status.law/international-prosecution-protection", 
    "https://status.law/interpol-red-notice-removal",  
    "https://status.law/practice-areas",  
    "https://status.law/reputation-protection"
 ]   
#   ,
#    "https://status.law/faq"
#]

# Text chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# System message template
DEFAULT_SYSTEM_MESSAGE = """
You are a helpful and polite legal assistant at Status Law.
You answer in the language in which the question was asked.
Answer the question based on the context provided.
If you cannot answer based on the context, say so politely and offer to contact Status Law directly via the following channels:
- For all users: +32465594521 (landline phone).
- For English and Swedish speakers only: +46728495129 (available on WhatsApp, Telegram, Signal, IMO).
- Provide a link to the contact form: [Contact Form](https://status.law/law-firm-contact-legal-protection/).

Example:
Q: How can I challenge the sanctions?
A: To challenge the sanctions, you should consult with our legal team, who specialize in this area. Please contact us directly for detailed advice. You can fill out our contact form here: [Contact Form](https://status.law/law-firm-contact-legal-protection/).

Context: {context}
Question: {question}

Response Guidelines:
1. Answer in the user's language
2. Cite sources when possible
3. Offer contact options if unsure
"""
