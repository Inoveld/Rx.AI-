import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from googleapiclient.discovery import build
import os


# API Keys
OPENAI_API_KEY=je_openai_key
GOOGLE_API_KEY=je_google_key
CSE_ID=je_cse_id

def extract_keywords_with_chatgpt(user_input):
    """Gebruik ChatGPT om relevante steekwoorden te extraheren."""
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(
        input_variables=["vraag"],
        template="Geef een lijst van maximaal 6 steekwoorden voor de volgende zoekopdracht, geef alleen de steekwoorden zonder nummers, punten of andere tekens, gescheiden door spaties: {vraag}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"vraag": user_input})
    
    
    response_cleaned = ' '.join([word for word in response.split() if not word.isdigit()])
    
    return response_cleaned.split()

def google_search(query, num_results=10):
    
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    search_results = []
    try:

        result = service.cse().list(q=query, cx=CSE_ID, num=num_results).execute()
        search_results.extend(result.get("items", []))
    except Exception as e:
        print(f"Fout bij zoeken: {e}")
    return search_results


def scrape_website(url):
    """Scrape de website en haal de belangrijkste tekst op."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return ""
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])
    return text

def split_text(text):
    """Splits de tekst in kleinere stukken voor verwerking."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)

def generate_answer(text_chunks, user_input):
    """Genereer een antwoord op basis van de samengevoegde tekst."""
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(
        input_variables=["informatie", "vraag"],
        template="""Je bent een deskundige ziekenhuisapotheker. Geef een medisch onderbouwd antwoord.
        Vraag: {vraag}
        Informatie: {informatie}
        """
    )
    combined_text = "\n".join(text_chunks)
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run({"informatie": combined_text, "vraag": user_input})
    return answer.strip()

def main():
    print("Welkom bij PharmAssist! Stel je medische vraag of typ 'exit' om te stoppen.")
    while True:
        user_input = input(">> ")
        if user_input.lower() == "exit":
            print("Bedankt voor het gebruik van PharmAssist. Tot ziens!")
            break
        try:
            print("Analyseren van vraag...")
            keywords = extract_keywords_with_chatgpt(user_input)
            if not keywords:
                print("Geen relevante zoektermen gevonden.")
                continue
            
            query = " ".join(keywords)
            print(f"Uitvoeren van zoekopdracht: {query}")
            search_results = google_search(query, num_results=10)  
            
            if not search_results:
                print("Geen relevante resultaten gevonden.")
                continue
            
            print("Gevonden resultaten:")
            all_text = ""
            for result in search_results:
                if len(all_text) >= 8000:
                    break  # Stop zodra we 2000 tekens hebben
                print(f"- {result.get('title')}: {result.get('link')}")
                text = scrape_website(result.get("link"))
                print(f"Gescraapte tekst van {result.get('link')}: {text[:8000]}...")  # Laat een stukje van de gescrapete tekst zien
                all_text += text[:8000 - len(all_text)]  # Voeg toe tot limiet is bereikt
            
            text_chunks = split_text(all_text)
            print("Genereren van antwoord...")
            answer = generate_answer(text_chunks, user_input)
            print(f"\nAntwoord: {answer}")
        except Exception as e:
            print(f"Er is een fout opgetreden: {e}")

if __name__ == "__main__":
    main()

