import re

def extract_session_id(session_str: str):
    match_id = re.search(r"/sessions/(.*?)/contexts/", session_str)
    if match_id:
        extracted_str = match_id.group(1)
        return extracted_str
    return ""

def get_str_from_food_dict(food_dict: dict):
    return ", ".join([f"{int(value)} {key}" for key, value in food_dict.items()])

# if __name__ == "__main__":
#     print(extract_session_id("projects/food-web-chatbot-qtxk/agent/sessions/aa312fb6-3e8d-d56a-b926-66079cd88ee0/contexts/ongoing-order"))