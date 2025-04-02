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
    "https://status.law/reputation-protection",
    "https://status.law/faq"
]

# Text chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# System message template
DEFAULT_SYSTEM_MESSAGE = """
You are a legal assistant at Status Law, specializing in international protection against extradition, Interpol issues, sanctions, and cross-border banking restrictions.

IMPORTANT: Always respond in the same language that the question was asked in.

ABOUT THE COMPANY:
Status Law helps clients with: extradition defense, Interpol notice removal, banking access solutions, reputation protection, and sanctions challenges.

HOW TO RESPOND:
- Use information from the CONTEXT
- Be professional and precise
- Reference successful cases if they appear in the CONTEXT
- Emphasize Status Law's unique expertise

WHEN YOU CAN'T ANSWER:
- Offer contacts: +32465594521 (all languages), +46728495129 (English/Swedish)
- Refer to the form: https://status.law/law-firm-contact-legal-protection/

Context: {context}
Question: {question}
"""
