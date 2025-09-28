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


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()


DB_FAISS_PATH="vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def get_lat_lon(city):
    url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
    headers = {'User-Agent': 'SymptomChatbot/1.0'}
    response = requests.get(url, headers=headers)
    data = response.json()
    if data:
        return float(data[0]['lat']), float(data[0]['lon'])
    return None, None


def get_hospitals(lat, lon, radius=10000):  # Increased radius to 10km for better coverage
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
    response = requests.get(overpass_url, params={'data': overpass_query})
    if response.status_code != 200:
        return []
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
                # Enrich with mockup details if available
                facility['details'] = get_hospital_details(name)
                facilities.append(facility)
    return facilities


# Mockup data for Hospital Details (expand with real data from APIs/datasets)
HOSPITAL_DETAILS_MOCKUP = {
    "Lilavati Hospital": {
        "operation_costs": "Average admission cost: ₹50,000-₹2,00,000 (depending on procedure)",
        "doctor_fees": {
            "General Physician": "₹1,500-₹2,000 per consultation",
            "Cardiologist": "₹2,500-₹3,500 per consultation"
        },
        "facilities": [
            {"name": "MRI Scan", "fee": "₹8,000-₹12,000"},
            {"name": "CT Scan", "fee": "₹6,000-₹10,000"},
            {"name": "ICU Bed (per day)", "fee": "₹15,000-₹25,000"}
        ],
        "pm_schemes_accepted": ["Ayushman Bharat (PM-JAY) - Yes, empanelled for 1,500+ procedures"],
        "state_schemes_accepted": ["Maharashtra Aarogyasri - Yes", "Rajiv Gandhi Jeevandayee Arogya Yojana - Yes"]
    },
    "Hinduja Hospital": {
        "operation_costs": "Average admission cost: ₹40,000-₹1,50,000 (depending on procedure)",
        "doctor_fees": {
            "General Physician": "₹1,200-₹1,800 per consultation",
            "Neurologist": "₹2,000-₹3,000 per consultation"
        },
        "facilities": [
            {"name": "MRI Scan", "fee": "₹7,500-₹11,000"},
            {"name": "X-Ray", "fee": "₹1,500-₹3,000"},
            {"name": "Operation Theatre (per hour)", "fee": "₹10,000-₹20,000"}
        ],
        "pm_schemes_accepted": ["Ayushman Bharat (PM-JAY) - Yes, empanelled for secondary/tertiary care"],
        "state_schemes_accepted": ["Maharashtra Aarogyasri - Yes"]
    },
    "Bombay Hospital": {
        "operation_costs": "Average admission cost: ₹30,000-₹1,00,000 (depending on procedure)",
        "doctor_fees": {
            "General Physician": "₹800-₹1,200 per consultation",
            "Surgeon": "₹1,500-₹2,500 per consultation"
        },
        "facilities": [
            {"name": "MRI Scan", "fee": "₹6,000-₹9,000"},
            {"name": "Ultrasound", "fee": "₹2,000-₹4,000"},
            {"name": "Lab Tests (CBC)", "fee": "₹500-₹1,000"}
        ],
        "pm_schemes_accepted": ["Ayushman Bharat (PM-JAY) - Yes"],
        "state_schemes_accepted": ["Rajiv Gandhi Jeevandayee Arogya Yojana - Yes"]
    }
    # Add more hospitals as needed; for real data, fetch from APIs
}


def get_hospital_details(hospital_name):
    return HOSPITAL_DETAILS_MOCKUP.get(hospital_name, {
        "operation_costs": "Data not available",
        "doctor_fees": "Data not available",  # Kept as string for fallback
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
    # Check if doctor_fees is a dictionary or string
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
    html += "</ul>"
    
    html += "</div>"
    return folium.Popup(folium.Html(html, script=True))


def main():
    st.title("Ask Chatbot!")

    # Sidebar for city input
    city = st.sidebar.text_input("Your City:", placeholder="e.g., New York, Mumbai")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        
        prompt_lower = prompt.lower()
        is_location_query = "nearby" in prompt_lower and any(word in prompt_lower for word in ["doctor", "hospital", "clinic"])
                
        if is_location_query:
            with st.chat_message('assistant'):
                if not city:
                    result = "Please enter your city in the sidebar to find nearby doctors, hospitals, or clinics."
                    st.warning(result)
                else:
                    lat, lon = get_lat_lon(city)
                    if not lat or not lon:
                        result = f"Could not locate '{city}'. Please check the spelling or try a different city name."
                        st.error(result)
                    else:
                        with st.spinner("Searching for nearby facilities..."):
                            facilities = get_hospitals(lat, lon)
                        if not facilities:
                            result = f"No hospitals, clinics, or doctors found near {city}. Try increasing the search area or checking the city name."
                            st.warning(result)
                        else:
                            result = f"Found {len(facilities)} nearby facilities in {city} (within ~10km):\n\n"
                            for i, f in enumerate(facilities, 1):
                                result += f"{i}. {f['name']}\n"
                            st.markdown(result)
                            
                            # Create and display map with enhanced popups
                            m = folium.Map(location=[lat, lon], zoom_start=13)
                            # User's location marker
                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=8,
                                popup="Your approximate location",
                                color="blue",
                                fill=True,
                                fillColor="blue"
                            ).add_to(m)
                            # Facilities markers with detailed popups
                            for f in facilities:
                                f['details']['name'] = f['name']  # Add name to details for popup
                                popup = create_hospital_popup(f)
                                folium.Marker(
                                    [f['lat'], f['lon']],
                                    popup=popup,
                                    tooltip=f['name'],
                                    icon=folium.Icon(color='red', icon='info-sign')
                                ).add_to(m)
                            folium_static(m, width=700, height=500)
                
                st.session_state.messages.append({'role':'assistant', 'content': result})
        else:
            try: 
                vectorstore=get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")

                GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
                GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Change to any supported Groq model
                llm = ChatGroq(
                    model=GROQ_MODEL_NAME,
                    temperature=0.5,
                    max_tokens=512,
                    api_key=GROQ_API_KEY,
                )
                
                retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

                # Document combiner chain (stuff documents into prompt)
                combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

                # Retrieval chain (retriever + doc combiner)
                rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

                response=rag_chain.invoke({'input': prompt})

                result=response["answer"]
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role':'assistant', 'content': result})

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()


    # to run:
    # .\.venv\Scripts\Activate
    # streamlit run medibot.py