import os
import re
import sys
import json
import time
import logging
import tiktoken
import argparse
import pandas as pd
import wikipediaapi
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from manifesto_categories import ALL_CATEGORIES

# Load environment variables
load_dotenv()

# Get paths from environment with defaults
RESULTS_DIR = os.getenv('RESULTS_DIR')
NOTEBOOKS_DIR = os.getenv('NOTEBOOKS_DIR', os.path.join(Path(__file__).resolve().parent.parent, 'notebooks'))
DOCS_DIR = os.getenv('DOCS_DIR', os.path.join(Path(__file__).resolve().parent.parent, 'docs', 'topics'))
CACHE_PATH = os.getenv('CACHE_PATH', os.path.join(DOCS_DIR, 'summary_cache.csv'))

# Get topic summaries paths
TOPIC_SUMMARIES_FILE = os.getenv('TOPIC_SUMMARIES_FILE', 'v2.0_people_summaries_un.csv')
TOPIC_SUMMARIES_UPDATED_FILE = os.getenv('TOPIC_SUMMARIES_UPDATED_FILE', 'v2.0_people_summaries_updated.csv')
SUMMARIES_PATH = os.path.join(DOCS_DIR, TOPIC_SUMMARIES_FILE)
SUMMARIES_UPDATED_PATH = os.path.join(DOCS_DIR, TOPIC_SUMMARIES_UPDATED_FILE)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process manifesto tags.')
parser.add_argument('--check-tags', action='store_true', help='Only check existing tags without processing new ones')
parser.add_argument("--input", type=str, help="Input file with people to process")
parser.add_argument("--output", type=str, help="Output file for results")
parser.add_argument("--cleanup", type=str, help="Clean up boolean values in specified file")
args = parser.parse_args()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='ideological-spectrum-llms/1.0'
)

def fetch_person_summary(client, name):
    """Fetch a summary directly from Wikipedia."""
    try:
        # Try to get the page directly
        page = wiki_wiki.page(name)
        
        if not page.exists():
            logger.warning(f"No Wikipedia page found for {name}")
            return None
            
        # Get the summary
        summary = page.summary
        
        # Log the summary for verification
        logger.info(f"\nFetched Wikipedia summary for {name}:")
        logger.info(f"URL: {page.fullurl}")
        logger.info(f"Summary preview: {summary[:200]}...")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error fetching Wikipedia summary for {name}: {e}")
        return None

def num_tokens_from_string(text, model="gpt-4-0125-preview"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def verify_tag_data(df):
    """Verify that entries in the dataframe actually have tag data."""
    result_columns = [col for col in df.columns if col.endswith('.result')]
    
    def has_valid_tags(row):
        for col in result_columns:
            val = str(row[col]).lower() if pd.notna(row[col]) else ""
            if val in ('true', 'false'):
                return True
        return False
    
    # Get sets of names with and without valid tags
    tagged_names = set(df[df.apply(has_valid_tags, axis=1)]['name'])
    present_names = set(df['name'])
    
    return tagged_names, present_names

def fix_boolean_values(df):
    """Standardize boolean values to Python's True/False"""
    result_columns = [col for col in df.columns if col.endswith('.result')]
    
    for col in result_columns:
        # Convert string booleans to Python booleans
        df[col] = df[col].map({'true': True, 'false': False}, na_action='ignore')
    
    return df

def find_untagged_people(tag_files, all_names):
    """Find people who have no valid tags across any category files."""
    tagged_in_any = set()
    present_in_any = set()
    
    # Collect names that have valid tags in any file
    for category_info in tag_files.values():
        if "names" in category_info:
            tagged_in_any.update(category_info["names"])
        if "df" in category_info:
            present_in_any.update(set(category_info["df"]["name"].tolist()))
    
    # Find people with no tags
    untagged = set(all_names) - tagged_in_any
    
    # Find people present in files but without tags
    present_but_untagged = present_in_any - tagged_in_any
    
    return untagged, present_but_untagged

def load_existing_tags(notebooks_dir=None):
    """Load all existing tag files and report on their contents."""
    tag_files = defaultdict(dict)
    total_tagged = set()
    total_present = set()
    
    # Set default notebooks directory if none provided
    if notebooks_dir is None:
        current_dir = Path(__file__).resolve().parent
        notebooks_dir = current_dir.parent / "notebooks"
    else:
        notebooks_dir = Path(notebooks_dir)
    
    if not notebooks_dir.exists():
        logger.warning(f"Notebooks directory not found at {notebooks_dir}")
        return tag_files, total_tagged
    
    pattern = "manifesto_tagged_topics_*.csv"
    logger.info(f"Looking for existing tag files in {notebooks_dir}")
    
    for file_path in notebooks_dir.glob(pattern):
        category = file_path.stem.replace("manifesto_tagged_topics_", "")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            tagged_names, present_names = verify_tag_data(df)
            
            tag_files[category]["df"] = df
            tag_files[category]["names"] = tagged_names
            total_tagged.update(tagged_names)
            total_present.update(present_names)
            
            logger.info(f"Loaded {file_path.name}:")
            logger.info(f"  - {len(tagged_names)} people with valid tags")
            logger.info(f"  - {len(present_names)} people present in file")
            logger.info(f"  - {len(present_names - tagged_names)} people without valid tags")
            
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    if not tag_files:
        logger.warning(f"No tag files found in {notebooks_dir}")
    else:
        logger.info(f"Summary across all files:")
        logger.info(f"  - {len(tag_files)} tag files found")
        logger.info(f"  - {len(total_tagged)} unique people with valid tags")
        logger.info(f"  - {len(total_present)} unique people present in files")
        logger.info(f"  - {len(total_present - total_tagged)} unique people without valid tags")
    
    return tag_files, total_tagged

class AssistantCreator:
    def __init__(self, client, model="gpt-4o"):
        self.client = client
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

    def create(self, document_name, instructions):
        try:
            self.logger.info(f"Creating an assistant for {document_name}...")
            assistant = client.beta.assistants.create(
                name=document_name,
                instructions=instructions,
                model=self.model,  # Assuming this model is available and suitable
                response_format={"type": "json_object"},
            )
            return assistant
        except Exception as e:
            self.logger.error(f"Error creating assistant for {document_name}: {e}")
            return None

    @staticmethod
    def _respond_tool_config():
        return {
            'type': 'function',  
            'function': {  
                'name': 'respond',  
                'description': 'Respond to the user in a structured manner.', 
                'parameters': {  
                    'properties': {
                        'structured_response': {
                            'type': 'object',
                            'description': 'Your structured response. Only accepts JSON format!'
                        }
                    },
                    'required': ['structured_response'],
                    'type': 'object'
                }
            }
        }

class MessageHandler:
    def __init__(self, client, model="gpt-4o", document_name="DefaultDocument", initial_instructions=""):
        self.client = client
        self.model = model
        self.assistant_creator = AssistantCreator(client, model)
        self.document_name = document_name
        self.initial_instructions = initial_instructions
        self.assistant = self.create_assistant(document_name, initial_instructions)
        self.thread = self.create_thread(self.assistant.id)
        
    def create_assistant(self, document_name, instructions):
        return self.assistant_creator.create(document_name, instructions)

    def create_thread(self, assistant_id):
        thread = self.client.beta.threads.create()
        return thread

    def create_file(self, file_path):
        with open(file_path, "rb") as file:
            response = self.client.files.create(file=file, purpose="assistants")
        return response.id

    def create_message(self, content):
        response = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=content,
        )
        return response

    def check_run_status(self, run_id):
        run_status = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run_id)
        while run_status.status not in ["completed", "expired", "failed", "cancelled", "requires_action"]:
            time.sleep(1)
            run_status = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run_id)
        return run_status

    def send_message_and_get_response(self, message_content):
        self.create_message(content=message_content)
        run = self.client.beta.threads.runs.create(thread_id=self.thread.id, assistant_id=self.assistant.id)
        run_status = self.check_run_status(run.id)
        response_messages = self.client.beta.threads.messages.list(thread_id=self.thread.id, order="asc").data
        return response_messages, run_status

class SimpleInstructionParser:
    def __init__(self, client, document_name, initial_instructions):
        self.client = client
        self.document_name = document_name
        self.initial_instructions = initial_instructions
        self.accumulated_message = ""
        self.total_context_tokens_count = 0
        self.message_handler = MessageHandler(client)

    def parse(self, text):
        """Parse the text using the initial instructions."""
        # Combine instructions with text
        message = f"{self.initial_instructions}\n{text}"
        
        try:
            # Send message to OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes text and provides structured responses in JSON format."},
                    {"role": "user", "content": message}
                ],
                response_format={ "type": "json_object" }
            )
            
            # Extract and return the JSON response
            return {self.document_name: response.choices[0].message.content}
            
        except Exception as e:
            logger.error(f"Error parsing text for {self.document_name}: {e}")
            return None

def flatten_dict_with_section(d):
    """Flatten a dictionary containing JSON string responses into a list of dictionaries."""
    flattened_data = []
    try:
        for topic, content_str in d.items():
            content_dict = json.loads(content_str)
            content_dict["name"] = topic
            flattened_data.append(content_dict)
    except Exception as e:
        print(f"Error: {e}")
    return flattened_data

# Define base instructions template
base_instructions = """Given the following summary, tell me what tags apply to this person based on the provided list of tags. Present the results in JSON format. 
Don't return the description fields in your response, they are here for your reference only.

Output the results in the following JSON format:

json
{
    "name": "Person Name",
    "basic_information": {
        "birth_date": "YYYY-MM-DD",
        "death_date": "YYYY-MM-DD",
        "nationality": "Nationality",
        "ethnicity": "Ethnicity",
        "gender": "Gender"
    },
    "era_and_geographical_context": {
        "historical_period": "Period",
        "region_or_country_of_influence": "Region/Country"
    },
    "categories": {json.dumps(ALL_CATEGORIES, indent=4)}
}

Summary:
"""

# Define the category groups
CATEGORY_GROUPS = {
    "external_relations": (100, 199),  # 100-series
    "freedom_and_democracy": (200, 299),  # 200-series
    "political_system": (300, 399),  # 300-series
    "economy": (400, 499),  # 400-series
    "welfare_and_quality_of_life": (500, 599),  # 500-series
    "fabric_of_society": (600, 699),  # 600-series
    "social_groups": (700, 799),  # 700-series
}

def get_categories_for_group(all_categories, group_range):
    """Extract categories that fall within the specified number range"""
    start, end = group_range
    categories = {}
    for cat_id, cat_data in all_categories.items():
        try:
            # Handle special cases like "000" and category variants like "108_a"
            if cat_id == "000":
                categories[cat_id] = cat_data
                continue
                
            # Extract the numeric part for comparison
            base_num = int(''.join(filter(str.isdigit, cat_id.split('_')[0])))
            if start <= base_num <= end:
                categories[cat_id] = cat_data
        except ValueError:
            continue
    return categories

def create_prompt_for_group(group_name, categories):
    """Create a prompt for a specific category group"""
    return f"""Given the following summary, tell me what tags apply to this person based on the provided list of tags for {group_name}. Present the results in JSON format. 
Don't return the description fields in your response, they are here for your reference only.

Output the results in the following JSON format:

json
{{
    "name": "Person Name",
    "basic_information": {{
        "birth_date": "YYYY-MM-DD",
        "death_date": "YYYY-MM-DD",
        "nationality": "Nationality",
        "ethnicity": "Ethnicity",
        "gender": "Gender"
    }},
    "era_and_geographical_context": {{
        "historical_period": "Period",
        "region_or_country_of_influence": "Region/Country"
    }},
    "categories": {json.dumps(categories, indent=4)}
}}

Summary:
"""

# Create prompts for each category group
initial_instructions = {
    group_name: create_prompt_for_group(
        group_name, 
        get_categories_for_group(ALL_CATEGORIES, group_range)
    )
    for group_name, group_range in CATEGORY_GROUPS.items()
}

# List of people to process
people_to_process = [
    'Viktor Zubkov', 'Najib Razak', 'Egor Gajdar', 'Marcos Pérez Jiménez',
    'Tshisekedi Tshilombo Felix', 'Gustavo Díaz-Ordaz polancoas',
    'Mohammad-Bagher Ghalibaf', 'Isaac Shamir', 'Michail Fradkov',
    'Isaac Ben-Zvi', 'Almazbek Atambaev', 'Allen W. Dulles', 'Sergei Kirienko',
    'Vjačeslav Volodin', 'Lech Walesa', 'Hugo Chávez',
    'Victor Emmanuel III of Italy', 'Nursultan Nazarbaev', 'Michail Mišustin',
    'Victor Yanukovych', 'Constantine Chernenko', 'Victor Chernomyrdin',
    'Alexander Dugin', 'Hermann Fegelein (SS General)', 'Sergej Sobjanin',
    'Mohammed V of Morocco', 'Ilham Aliev', 'Sergej Šoigu', 'Victor Yushchenko',
    'Felipe Del Sagrado Corazón De Jesus Calderón hinojosa', 'Heinrich Müller',
    'Mohammed Bin Zayed Al Nahyan', 'Henry Cavill-Viscuso',
    'Alicia Augello Cook', 'Liam Payne', 'Lauta marti', 'Anne, Princess Royal',
    'Osamu Dazai', 'Jordan Belfort', 'Charles III', 'Richard Grenell',
    'Ahmed Ouyahia', 'George Stinney', 'San Edith Stein', 'Kahlil Gibran',
    'Ferruccio Lamborghini', 'Santa Faustina Kowalska', 'Florentino Pérez',
    'Clementine Churchill', 'Abolhassan Banisadr', 'Joaquín Balaguer',
    'Olexy Honcharuk', 'Sergei Aksyonov', 'Amanullah Khan', 'Ovadia Joseph',
    'Sergej Naryškin', 'Susie Wolff', 'Mehmed VI', 'Santiago Abascal',
    'Heydar Aliev', 'Sergej Lavrov', 'Saparmurat Niyazov', 'B. J. Habibie',
    'Jim Mattis', 'Hyun Bin', 'Achraf Hakimi', 'Amado Carrillo',
    'Miguel Ángel Félix Gallardo', 'René Emilio Barrientos Ortuño',
    'Otto von Habsburg', 'Adolfo Rodríguez-Saá', 'Stephen Harden',
    'Stanisław Šuškievič', 'Jonathan Netanyahu', 'Princess Muna Al-Hussein',
    'Michail Kasjanov', 'Princess Margarita of Greece and Denmark',
    'Abdul Razak Hussein', 'Cipriano Castro', 'Charles I of Austria',
    'Kim Jong-chul', 'Abdülmecid II', 'Olexander Turchynov',
    'Baudouin I of Belgium', 'Michael I of Romania', 'Saud of Saudi Arabiه',
    'amin berrabahgay', 'Andrej Belousov', 'Isaac Navon', 'July Edelstein',
    'Johanna Bormann', 'Maria Kiselyova'
]

def load_or_create_summary_cache():
    """Load existing summary cache or create a new one."""
    if os.path.exists(CACHE_PATH):
        cache_df = pd.read_csv(CACHE_PATH)
        logger.info(f"Loaded {len(cache_df)} cached summaries")
        return cache_df
    else:
        cache_df = pd.DataFrame(columns=["name-en", "summary-en"])
        logger.info("Created new summary cache")
        return cache_df

def save_summary_cache(cache_df):
    """Save the summary cache to disk."""
    cache_df.to_csv(CACHE_PATH, index=False)
    logger.info(f"Saved {len(cache_df)} summaries to cache")

# Load summary data and cache
summary_df = pd.read_csv(SUMMARIES_PATH)
cache_df = load_or_create_summary_cache()

# Combine existing summaries with cache
summary_df = pd.concat([summary_df, cache_df[~cache_df["name-en"].isin(summary_df["name-en"])]], ignore_index=True)
names = summary_df["name-en"].tolist()
summaries = summary_df["summary-en"].tolist()
logger.info(f"Loaded {len(names)} people from summary file and cache")

# Check which people from our list are in the dataset
logger.info("\nChecking which people are in our dataset:")
missing_people = []
found_people = []
for person in people_to_process:
    if person in names:
        found_people.append(person)
    else:
        missing_people.append(person)

logger.info(f"\nSummary of people to process:")
logger.info(f"Total people in list: {len(people_to_process)}")
logger.info(f"People with summaries: {len(found_people)} ({len(found_people)/len(people_to_process)*100:.1f}%)")
logger.info(f"People missing summaries: {len(missing_people)} ({len(missing_people)/len(people_to_process)*100:.1f}%)")

logger.info(f"\nFound {len(found_people)} people in dataset:")
for person in found_people[:10]:  # Show first 10
    logger.info(f"  - {person}")
if len(found_people) > 10:
    logger.info(f"  ... and {len(found_people) - 10} more")

logger.info(f"\nMissing {len(missing_people)} people from dataset:")
for person in missing_people:
    logger.info(f"  - {person}")

# Fetch summaries for missing people
if missing_people:
    logger.info("\nFetching summaries for missing people...")
    new_summaries = []
    for person in tqdm(missing_people, desc="Fetching summaries"):
        summary = fetch_person_summary(client, person)
        if summary:
            new_summaries.append({
                "name-en": person,
                "summary-en": summary
            })
            logger.info(f"Got summary for {person}")
        time.sleep(1)  # Rate limiting

    if new_summaries:
        # Create DataFrame with new summaries
        new_df = pd.DataFrame(new_summaries)
        
        # Add to cache
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        save_summary_cache(cache_df)
        
        # Combine with existing summaries
        summary_df = pd.concat([summary_df, new_df], ignore_index=True)
        
        # Save updated summaries
        summary_df.to_csv(SUMMARIES_UPDATED_PATH, index=False)
        logger.info(f"Saved updated summaries to: {SUMMARIES_UPDATED_PATH}")
        
        # Update our working data
        names = summary_df["name-en"].tolist()
        summaries = summary_df["summary-en"].tolist()
        logger.info(f"Updated dataset now contains {len(names)} people")
        
        # Print final statistics
        logger.info("\nFinal summary after fetching:")
        found_after = sum(1 for person in people_to_process if person in names)
        missing_after = len(people_to_process) - found_after
        logger.info(f"People with summaries: {found_after} ({found_after/len(people_to_process)*100:.1f}%)")
        logger.info(f"People still missing summaries: {missing_after} ({missing_after/len(people_to_process)*100:.1f}%)")

# Load existing tag files with explicit path
logger.info(f"Looking for tag files in: {NOTEBOOKS_DIR}")
if not os.path.exists(NOTEBOOKS_DIR):
    logger.warning(f"Notebooks directory does not exist: {NOTEBOOKS_DIR}")
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    logger.info(f"Created notebooks directory: {NOTEBOOKS_DIR}")

tag_files, total_tagged = load_existing_tags(NOTEBOOKS_DIR)

# After loading tag files
if args.check_tags:
    logger.info("Checking existing tags for specified people...")
    for person in people_to_process:
        logger.info(f"\nChecking tags for: {person}")
        for category, info in tag_files.items():
            if "df" not in info:
                continue
                
            df = info["df"]
            if person in df["name"].values:
                # Get all result columns
                result_cols = [col for col in df.columns if col.endswith('.result')]
                person_data = df[df["name"] == person].iloc[0]
                
                # Check which categories have valid tags
                valid_tags = []
                for col in result_cols:
                    val = str(person_data[col]).lower() if pd.notna(person_data[col]) else ""
                    if val and val != "true/false" and val != "nan" and val != "false" and val != "true":
                        cat_name = col[:-7]
                        valid_tags.append(cat_name)
                
                if valid_tags:
                    logger.info(f"  {category}: {len(valid_tags)} tags")
                    for tag in valid_tags:
                        logger.info(f"    - {tag}")
                else:
                    logger.info(f"  {category}: No valid tags")
            else:
                logger.info(f"  {category}: Not present in file")
    sys.exit(0)

# Process each category group separately
for group_name in CATEGORY_GROUPS:
    output_file = os.path.join(NOTEBOOKS_DIR, f"manifesto_tagged_topics_{group_name}.csv")
    logger.info(f"\nProcessing {group_name}:")
    logger.info(f"Output file will be: {output_file}")
    
    # Load previous results
    if group_name in tag_files:
        previous_df = tag_files[group_name]["df"]
        logger.info(f"Loaded previous results for {group_name}")
    else:
        previous_df = None
        logger.info(f"No previous results found for {group_name}")

    responses = []
    # Only process people from our list who are in the dataset
    to_process = [name for name in people_to_process if name in names]
    logger.info(f"Will process {len(to_process)} people")
    
    # Process each person
    for name in tqdm(to_process, desc=f"Processing {group_name}", total=len(to_process)):
        try:
            # Get the summary for this person
            idx = names.index(name)
            summary = summaries[idx]
            
            logger.debug(f"Processing {name}")
            
            # Combine instructions with summary
            message = f"{initial_instructions[group_name]}\n{summary}"
            
            # Send request to OpenAI
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes text and provides structured responses in JSON format."},
                    {"role": "user", "content": message}
                ],
                response_format={ "type": "json_object" }
            )
            
            # Extract the response
            json_response = {name: response.choices[0].message.content}
            responses.append(json_response)
            logger.debug(f"Successfully processed {name}")
            
            # Add a small delay to avoid rate limits
            time.sleep(1)
            
        except ValueError:
            logger.error(f"Could not find summary for {name}")
            continue
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            continue

    logger.info(f"Processed {len(responses)} new responses")

    # Save results if we have any new responses
    if responses:
        # Only process responses that actually have content
        valid_responses = [r for r in responses if r is not None]
        if valid_responses:
            flat_responses = [flatten_dict_with_section(response) for response in valid_responses]
            if flat_responses:
                new_df = pd.concat([pd.DataFrame.from_dict(pd.json_normalize(resp)) 
                                  for resp in flat_responses], ignore_index=True)
                
                if previous_df is not None:
                    # Remove any existing entries for these people that got valid responses
                    processed_names = [r.get("name") for r in valid_responses]
                    previous_df = previous_df[~previous_df["name"].isin(processed_names)]
                    # Combine with new results
                    df = pd.concat([previous_df, new_df], ignore_index=True)
                else:
                    df = new_df
                    
                df.sort_values(by="name", inplace=True)
                df.to_csv(output_file, index=False)
                logger.info(f"Saved results to {output_file}")
            
logger.info("Processing complete")
