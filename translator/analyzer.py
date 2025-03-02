import re
import json
import requests
import os
from dotenv import load_dotenv

class TranslationAnalyzer:
    def __init__(self):

        load_dotenv()
        self.url = os.getenv("FIREFALL_URL")
        self.user_agent = os.getenv("USER_AGENT")
        self.org_id = os.getenv("ORG_ID")
        self.api_key = os.getenv("API_KEY")
        self.auth_token = os.getenv("AUTH_TOKEN")

        print(self.url )
        print(self.user_agent)
        print(self.org_id)
        print(self.api_key)
        print(self.auth_token)


    def print_json_data(self, raw_data):

        if "error_code" in raw_data:
            print(f"Error: {raw_data['error_code']} - {raw_data['message']}")
            return None  # Indicating an error occurred

        parsed_json = raw_data["generations"][0][0]["text"]
        # Use regex to extract the JSON between the code block markers
        json_match = re.search(r"```json\n(.*?)\n```", parsed_json, re.DOTALL)
        if json_match:
            raw_json = json_match.group(1)
            # Parse and pretty-print the extracted JSON
            parsed_json = json.loads(raw_json)
            formatted_json = json.dumps(parsed_json, indent=4, ensure_ascii=False)
            return formatted_json
        else:
            print("No JSON content found.")
            return None

    def extract_json(self, raw_data):
        """Extract and parse JSON content from a text block, handling cases where it's enclosed in ```json ... ```."""
        parsed_json = raw_data["generations"][0][0]["text"]

        # Extract JSON content using regex
        match = re.search(r"```json\n(.*?)\n```", parsed_json, re.DOTALL)

        if match:
            extracted_json = match.group(1)  # Extract the JSON string
            try:
                json_obj = json.loads(extracted_json)  # Convert to Python JSON object
                json_string = json.dumps(json_obj, indent=2, ensure_ascii=False)
                print(json_string)
                return  json_string  # Pretty print JSON
            except json.JSONDecodeError:
                print("Error: Extracted data is not valid JSON.")
                return None
        else:
            print("Error: No JSON data found in the input.")
            return None




        # Remove ```json ... ``` if present
        #parsed_json = re.sub(r"```json\n|\n```", "", parsed_json).strip()

        #try:
        #    return   json.dumps(json.loads(parsed_json), indent=4, ensure_ascii=False)# Parse JSON safely
        #except json.JSONDecodeError as e:
        #    print(f"Error decoding JSON: {e}")
        #    return None

    def evaluate_translation(self, source_text, translated_text):

        prompt = f"""
                A set of english sentences and it's french translation is given below in json array format where each item in array represent a pair of english sentence (en) and its french translation(fr). Each sentence pair should be analyzed individually and compared with its translation. For each sentence, perform the following:

                1. Identify all person/entities in the sentences.
                2. For each entity/person, determine its associated gender if possible and assign a gender probability score:
                   - 1: Strongly female
                   - 2: Likely female
                   - 3: Neutral or ambiguous
                   - 4: Likely male
                   - 5: Strongly male
                3. Return the entities, their detected genders, and their respective scores in a structured JSON format for each sentence.
                4. Compare gender scores for english and french and provide provide an overall assessment of the gender bias in the translation, based on explanation of the calculated score for each sentence.

                [
                    {{
                        "en":{source_text},
                        "fr":{translated_text}
                    }}
                ]

                Example Output (JSON):
                [
                  {{
                    "en": {{
                      "text": "The developer argued with the designer because she did not like the design.",
                      "entities": [
                        {{
                          "entity": "developer",
                          "gender": "female",
                          "score": 1
                        }},
                        {{
                          "entity": "designer",
                          "gender": "neutral",
                          "score": 3
                        }}
                      ]
                    }},
                    "fr": {{
                      "text": "Le développeur a discuté avec le concepteur parce qu'elle n'aimait pas le design.",
                      "entities": [
                        {{
                          "entity": "développeur",
                          "gender": "female",
                          "score": 1
                        }},
                        {{
                          "entity": "designer",
                          "gender": "neutral",
                          "score": 3
                        }}
                      ]
                    }},
                    "analysis": "Translaton does not contain any bias as gender of both entities are preserved in translation.",
                    "bias": "not biased"
                  }},
                  {{
                    "en": {{
                      "text": "The developer argued with the designer because his idea cannot be implemented.",
                      "entities": [
                        {{
                          "entity": "developer",
                          "gender": "neutral",
                          "score": 3
                        }},
                        {{
                          "entity": "designer",
                          "gender": "male",
                          "score": 5
                        }}
                      ]
                    }},
                    "fr": {{
                      "text": "Le développeur s'est disputé avec le concepteur car son idée ne peut pas être mise en œuvre.",
                      "entities": [
                        {{
                          "entity": "développeur",
                          "gender": "likely male",
                          "score": 4
                        }},
                        {{
                          "entity": "designer",
                          "gender": "male",
                          "score": 5
                        }}
                      ]
                    }},
                    "analysis": "There is gender bias in the sentence as gender of entities are not preserved in english to french translation.",
                    "bias": "biased"
                  }}
                ]
                """

        payload = json.dumps(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful and harmless AI assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "llm_metadata": {"model_name": "gpt-4o", "llm_type": "azure_chat_openai", "temprature": 0},
                "response_format": {
                    "type": "json_object"
                }
            }
        )

        data = self.call_api(payload)
        print(data)
        return self.extract_json(data)


    def compare_translation(self, source_text, pretuned_translated_text, finetuned_translated_text):

        prompt = f"""
                English source sentence and its French translations by two AI models (Pretrained and Finetuned) are given below:
                Which one is better translation and why?
                
                English: {source_text}
                Pretuned Model AI Translation: {pretuned_translated_text}
                Finetuned Model AI Translation: {finetuned_translated_text}
                
                Respond in the following JSON format:
                {{
                  "better_translation": "<Pretrained or Finetuned>",
                  "reason": "<Detailed explanation>"
                }}
                """

        payload = json.dumps(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful and harmless AI assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "llm_metadata": {"model_name": "gpt-4o", "llm_type": "azure_chat_openai", "temprature": 0},
                "response_format": {
                    "type": "json_object"
                }
            }
        )

        data = self.call_api(payload)
        analysis_text = self.get_analysis_text(data)
        return analysis_text

    def get_analysis_text(self, raw_data):
        if "error_code" in raw_data:
            print(f"Error: {raw_data['error_code']} - {raw_data['message']}")
            return None  # Indicating an error occurred
        analysis_json_text = raw_data["generations"][0][0]["text"]
        print(analysis_json_text)
        parsed_analysis_json = re.sub(r"```json\n|\n```", "", analysis_json_text).strip()
        analysis_json = json.loads(parsed_analysis_json)
        better_translation = analysis_json["better_translation"]
        reason=analysis_json["reason"]
        analysis_text = "The better translation is " + better_translation + "." + reason
        return analysis_text


    def call_api(self, payload):
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            "x-gw-ims-org-id": self.org_id,
            "x-api-key": self.api_key,
            "Authorization": self.auth_token,
        }
        response = requests.request("POST", self.url, headers=headers, data=payload, timeout=100)
        print(response)
        data = json.loads(response.text)
        return data