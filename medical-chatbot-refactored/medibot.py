import os
import streamlit as st
import requests
import folium
from streamlit_folium import folium_static

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

DB_FAISS_PATH = os.environ.get("DB_FAISS_PATH", "vectorstore/db_faiss")


# ------------------ Vectorstore ------------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    index_file = os.path.join(DB_FAISS_PATH, "index.faiss")

    # If FAISS index missing, create a dummy one
    if not os.path.exists(index_file):
        st.info("FAISS index not found. Building a new one...")
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        # Replace with your actual documents
        documents = ["Document 1 text", "Document 2 text"]
        db = FAISS.from_texts(documents, embedding_model)
        db.save_local(DB_FAISS_PATH)
    else:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    return db


# ------------------ Prompts ------------------
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


# ------------------ Location & Hospital ------------------
def get_lat_lon(city):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
        headers = {'User-Agent': 'SymptomChatbot/1.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        pass
    return None, None


def get_hospitals(lat, lon, radius=10000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})["amenity"~"hospital|clinic|doctors"];
      way(around:{radius},{lat},{lon})["amenity"~"hospital|clinic|doctors"];
      relation(around:{radius},{lat},{lon})["amenity"~"hospital|clinic|doctors"];
    );
    out center;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        response.raise_for_status()
        data = response.json()
        facilities = []
        for element in data.get('elements', []):
            if 'tags' in element:
                name = element['tags'].get('name', 'Unnamed Facility')
                center = element.get('center', {})
                elem_lat = center.get('lat') or element.get('lat')
                elem_lon = center.get('lon') or element.get('lon')
                if elem_lat and elem_lon:
                    facility = {'name': name, 'lat': float(elem_lat), 'lon': float(elem_lon)}
                    facility['details'] = get_hospital_details(name)
                    facilities.append(facility)
        return facilities
    except Exception:
        return []


# ------------------ Mock Hospital Details ------------------
HOSPITAL_DETAILS_MOCKUP = {
    "Lilavati Hospital": {
        "operation_costs": "₹50,000-₹2,00,000",
        "doctor_fees": {"General Physician": "₹1,500-₹2,000", "Cardiologist": "₹2,500-₹3,500"},
        "facilities": [{"name": "MRI Scan", "fee": "₹8,000-₹12,000"}],
        "pm_schemes_accepted": ["Ayushman Bharat (PM-JAY) - Yes"],
        "state_schemes_accepted": ["Maharashtra Aarogyasri - Yes"]
    },
    "Hinduja Hospital": {
        "operation_costs": "₹40,000-₹1,50,000",
        "doctor_fees": {"General Physician": "₹1,200-₹1,800", "Neurologist": "₹2,000-₹3,000"},
        "facilities": [{"name": "MRI Scan", "fee": "₹7,500-₹11,000"}],
        "pm_schemes_accepted": ["Ayushman Bharat (PM-JAY) - Yes"],
        "state_schemes_accepted": ["Maharashtra Aarogyasri - Yes"]
    },
    "Bombay Hospital": {
        "operation_costs": "₹30,000-₹1,00,000",
        "doctor_fees": {"General Physician": "₹800-₹1,200", "Surgeon": "₹1,500-₹2,500"},
        "facilities": [{"name": "MRI Scan", "fee": "₹6,000-₹9,000"}],
        "pm_schemes_accepted": ["Ayushman Bharat (PM-JAY) - Yes"],
        "state_schemes_accepted": ["Rajiv Gandhi Jeevandayee Arogya Yojana - Yes"]
    }
}


def get_hospital_details(hospital_name):
    return HOSPITAL_DETAILS_MOCKUP.get(hospital_name, {
        "operation_costs": "Data not available",
        "doctor_fees": "Data not available",
        "facilities": [],
        "pm_schemes_accepted": ["Data not available"],
        "state_schemes_accepted": ["Data not available"]
    })


def create_hospital_popup(details):
    html = f"""
    <div style="width:300px; font-family: Arial; font-size:12px;">
        <h3>{details['name']}</h3>
        <h4>Operation Costs</h4>
        <p>{details['details']['operation_costs']}</p>
        <h4>Doctor Fees</h4>
    """
    if isinstance(details['details']['doctor_fees'], dict):
        html += "<ul>"
        for doc, fee in details['details']['doctor_fees'].items():
            html += f"<li>{doc}: {fee}</li>"
        html += "</ul>"
    else:
        html += f"<p>{details['details']['doctor_fees']}</p>"

    html += "<h4>Facilities</h4><ul>"
    for fac in details['details']['facilities']:
        html += f"<li>{fac['name']}: {fac['fee']}</li>"
    html += "</ul>"

    html += "<h4>PM Health Schemes</h4><ul>"
    for scheme in details['details']['pm_schemes_accepted']:
        html += f"<li>{scheme}</li>"
    html += "</ul>"

    html += "<h4>State Health Schemes</h4><ul>"
    for scheme in details['details']['state_schemes_accepted']:
        html += f"<li>{scheme}</li>"
    html += "</ul></div>"
    return folium.Popup(folium.Html(html, script=True))


# ------------------ Main App ------------------
def main():
    st.title("Ask Chatbot!")

    # Sidebar input
    city = st.sidebar.text_input("Your City:", placeholder="e.g., New York, Mumbai")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        prompt_lower = prompt.lower()
        is_location_query = "nearby" in prompt_lower and any(word in prompt_lower for word in ["doctor", "hospital", "clinic"])

        if is_location_query:
            with st.chat_message('assistant'):
                if not city:
                    result = "Please enter your city in the sidebar to find nearby hospitals or doctors."
                    st.warning(result)
                else:
                    lat, lon = get_lat_lon(city)
                    if not lat or not lon:
                        result = f"Could not locate '{city}'. Please check the spelling."
                        st.warning(result)
                    else:
                        with st.spinner("Searching nearby facilities..."):
                            facilities = get_hospitals(lat, lon)
                        if not facilities:
                            result = f"No hospitals or clinics found near {city}."
                            st.warning(result)
                        else:
                            result = f"Found {len(facilities)} nearby facilities in {city} (within ~10km):\n\n"
                            for i, f in enumerate(facilities, 1):
                                result += f"{i}. {f['name']}\n"
                            st.markdown(result)

                            # Map
                            m = folium.Map(location=[lat, lon], zoom_start=13)
                            folium.CircleMarker(location=[lat, lon], radius=8, popup="Your location", color="blue", fill=True, fillColor="blue").add_to(m)
                            for f in facilities:
                                f['details']['name'] = f['name']
                                popup = create_hospital_popup(f)
                                folium.Marker([f['lat'], f['lon']], popup=popup, tooltip=f['name'], icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
                            folium_static(m, width=700, height=500)

                st.session_state.messages.append({'role': 'assistant', 'content': result})
        else:
            # LLM + RAG chain
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load vectorstore")

                GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.5,
                    max_tokens=512,
                    api_key=GROQ_API_KEY,
                )

                retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
                combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
                rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

                response = rag_chain.invoke({'input': prompt})
                result = response.get("answer", "No answer returned.")
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})
            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
