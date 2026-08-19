import json
from openai import OpenAI
import os
import base64
import io

from PIL import Image
import numpy as np

import ast
import re
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# For captions
system_prompt_captions = '''
You are an agent specializing in accurate captioning objects in an image.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output in a structured format, the captions for the objects.

You will also be given a text list of the numeric ids and names of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...]

The names were obtained from a simple object detection system and may be inaacurate.

Your response should be in the format of a list of dictionaries, where each dictionary contains the id, name, and caption of an object. Your response will be evaluated as a python list of dictionaries, so make sure to format it correctly. An example of the expected response format is as follows:
[
    {"id": "1", "name": "object1", "caption": "concise description of the object1 in the image"},
    {"id": "2", "name": "object2", "caption": "concise description of the object2 in the image"},
    {"id": "3", "name": "object3", "caption": "concise description of the object3 in the image"}
    ...
]

And each caption must be a concise description of the object in the image.
'''

system_prompt_consolidate_captions = '''
You are an agent specializing in consolidating multiple captions for the same object into a single, clear, and accurate caption.

You will be provided with several captions describing the same object. Your task is to analyze these captions, identify the common elements, remove any noise or outliers, and consolidate them into a single, coherent caption that accurately describes the object.

Ensure the consolidated caption is clear, concise, and captures the essential details from the provided captions.

Here is an example of the input format:
[
    {"id": "3", "name": "cigar box", "caption": "rectangular cigar box on the side cabinet"},
    {"id": "9", "name": "cigar box", "caption": "A small cigar box placed on the side cabinet."},
    {"id": "7", "name": "cigar box", "caption": "A small cigar box is on the side cabinet."},
    {"id": "8", "name": "cigar box", "caption": "Box on top of the dresser"},
    {"id": "5", "name": "cigar box", "caption": "A cigar box placed on the dresser next to the coffeepot."},
]

Your response should be a JSON object with the format:
{
    "consolidated_caption": "A small rectangular cigar box on the side cabinet."
}

Do not include any additional information in your response.
'''

# For scene vocabulary discovery: enumerating all objects visible in a whole frame
#
# Restricted to easily-movable objects (see the "single ordinary person" test below)
# because the discovered names become YOLO-World's detection vocabulary for the whole
# scan: a name that gets in here gets a node in the graph, and large/fixed furniture
# nodes are unreliable across two separate scans (reconstruction noise reads as a
# spurious change) and prone to swallowing whatever smaller object sits on or against
# them during fusion. Filtering at the vocabulary stage stops that class of object from
# ever being detected at all, rather than only flagging it after the fact.
system_prompt_frame_objects = '''
You are an agent specializing in identifying objects in an image of an indoor scene.

Your task is to list every distinct type of physical object visible in the image that a single ordinary person could pick up, carry, or drag to a new spot as part of everyday life -- without tools, help, or disassembly. This includes things like "bottle", "pillow", "remote", "chair", "stool", "box", "towel", "rug", "bag", "book", "plant" -- ordinary items someone moves, replaces, or puts away routinely.

Do NOT include large or fixed/built-in furniture and structural elements that people do not casually relocate, such as "sofa", "bed", "bathtub", "sink", "toilet", "countertop", "cabinet", "closet", "wall-mounted shelf", "mirror", "wall", "floor", "ceiling", "door", "window", "refrigerator", "washing machine", or "staircase". If you are unsure whether something is fixed or movable, leave it out.

Name each object using as few words as possible -- ideally 1-2 words, never more than 3 -- and use a short, generic category name (e.g. "lamp", "pillow", "mug"), not a proper noun or brand name. List each object type only once, even if multiple instances of it are visible.

Your response must be a JSON object with the format:
{
    "objects": ["object type 1", "object type 2", ...]
}
'''

# For scene vocabulary discovery: naming a single cropped object segment
#
# Same "easily movable" restriction as system_prompt_frame_objects above, and for the
# same reason -- this crop-level pass is the other half of the same vocabulary.
system_prompt_segment_name = '''
You are an agent specializing in naming a single object shown in a cropped image.

The image shows one segmented object, possibly with some surrounding context. Name this object using as few words as possible -- ideally 1-2 words, never more than 3 -- but ONLY if it is something a single ordinary person could pick up, carry, or drag to a new spot as part of everyday life, without tools, help, or disassembly (e.g. "coffee mug", "chair", "towel"). Use a short, generic category name, not a proper noun or brand name.

Return an empty string for the name if the crop does not show such a movable object. This includes a piece of wall, floor, texture, shadow, or empty space, AND large or fixed/built-in furniture and structural elements a person would not casually relocate -- e.g. sofa, bed, bathtub, sink, toilet, countertop, cabinet, closet, wall-mounted shelf, mirror, door, window, refrigerator, washing machine, staircase. If you are unsure whether the object is fixed or movable, return an empty string.

Your response must be a JSON object with the format:
{
    "name": "object name"
}
'''

# 모델 이름을 로컬 vLLM 서버에 로드된 모델로 변경
gpt_model = "Qwen/Qwen3-VL-8B-Instruct"

def get_openai_client():
    # OpenAI 클라이언트가 로컬 vLLM 서버를 가리키도록 base_url 및 api_key 변경
    client = OpenAI(
        base_url="http://localhost:8019/v1",
        api_key="EMPTY"
    )
    return client

# Function to encode the image as base64
def encode_image_for_openai(image_path: str, resize = False, target_size: int=512):
    print(f"Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not resize:
        # Open the image
        # print(f"Opening image from path: {image_path}")
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            # print("Image encoded in base64 format.")
        return encoded_image
    
    # print(f"Opening image from path: {image_path}")
    with Image.open(image_path) as img:
        # Determine scaling factor to maintain aspect ratio
        original_width, original_height = img.size
        # print(f"Original image dimensions: {original_width} x {original_height}")
        
        if original_width > original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height = int(original_height * scale)
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width = int(original_width * scale)

        # print(f"Resized image dimensions: {new_width} x {new_height}")

        # Resizing the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        # print("Image resized successfully.")
        
        # Convert the image to bytes and encode it in base64
        with open("temp_resized_image.jpg", "wb") as temp_file:
            img_resized.save(temp_file, format="JPEG")
            # print("Resized image saved temporarily for encoding.")
        
        # Open the temporarily saved image for base64 encoding
        with open("temp_resized_image.jpg", "rb") as temp_file:
            encoded_image = base64.b64encode(temp_file.read()).decode('utf-8')
            # print("Image encoded in base64 format.")
        
        # Clean up the temporary file
        os.remove("temp_resized_image.jpg")
        # print("Temporary file removed.")

    return encoded_image

def encode_image_for_openai_from_pil(image: Image.Image) -> str:
    """Same base64 contract as encode_image_for_openai(), for an in-memory PIL
    image (e.g. a SAM-segment crop) that was never written to disk."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def get_frame_object_list(client: OpenAI, image_path: str) -> list:
    """Asks the VLM to enumerate every distinct object type visible in a whole
    frame. Used for scene vocabulary discovery (see general_utils.discover_scene_vocabulary).
    Returns a list of lowercase, stripped object names -- empty on any failure."""
    base64_image = encode_image_for_openai(image_path)

    messages = [
        {"role": "system", "content": system_prompt_frame_objects},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        },
    ]

    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            response_format={"type": "json_object"}
        )
        objects = json.loads(response.choices[0].message.content.strip()).get("objects", [])
        return [obj.strip().lower() for obj in objects if isinstance(obj, str) and obj.strip()]
    except Exception as e:
        print(f"An error occurred in get_frame_object_list: {str(e)}")
        return []


def get_segment_object_name(client: OpenAI, image_crop: Image.Image) -> str:
    """Asks the VLM to name a single cropped object segment in as few words as
    possible. Used for scene vocabulary discovery (see
    general_utils.discover_scene_vocabulary). Returns a lowercase, stripped
    name, or "" on any failure or if the crop isn't a distinct object."""
    base64_image = encode_image_for_openai_from_pil(image_crop)

    messages = [
        {"role": "system", "content": system_prompt_segment_name},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        },
    ]

    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            response_format={"type": "json_object"}
        )
        name = json.loads(response.choices[0].message.content.strip()).get("name", "")
        return name.strip().lower() if isinstance(name, str) else ""
    except Exception as e:
        print(f"An error occurred in get_segment_object_name: {str(e)}")
        return ""


def consolidate_captions(client: OpenAI, captions: list):
    # Formatting the captions into a single string prompt
    captions_text = "\n".join([f"{cap['caption']}" for cap in captions if cap['caption'] is not None])
    user_query = f"Here are several captions for the same object:\n{captions_text}\n\nPlease consolidate these into a single, clear caption that accurately describes the object."

    messages = [
        {
            "role": "system",
            "content": system_prompt_consolidate_captions
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    consolidated_caption = ""
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        consolidated_caption_json = response.choices[0].message.content.strip()
        consolidated_caption = json.loads(consolidated_caption_json).get("consolidated_caption", "")
        print(f"Consolidated Caption: {consolidated_caption}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        consolidated_caption = ""

    return consolidated_caption
    
def vlm_extract_object_captions(text: str):
    # Replace newlines with spaces for uniformity
    text = text.replace('\n', ' ')
    
    # Pattern to match the list of objects
    pattern = r'\[(.*?)\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Try to convert the entire string to a list of dictionaries
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            # If the whole string conversion fails, process each element individually
            elements = re.findall(r'{.*?}', list_str)
            result = []
            for element in elements:
                try:
                    obj = ast.literal_eval(element)
                    if isinstance(obj, dict):
                        result.append(obj)
                except (ValueError, SyntaxError):
                    print(f"Error processing element: {element}")
            return result
    else:
        # No matching pattern found
        print("No list of objects found in the text.")
        return []
    
def get_obj_captions_from_image_gpt4v(client: OpenAI, image_path: str, label_list: list):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)

    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."
    
    messages=[
        {
            "role": "system",
            "content": system_prompt_captions
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    
    vlm_answer_captions = []
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages
        )
        
        vlm_answer_str = response.choices[0].message.content
        # print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        
        vlm_answer_captions = vlm_extract_object_captions(vlm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    # print(f"Line 68, user_query: {user_query}")
    # print(f"Line 97, vlm_answer: {vlm_answer_captions}")
    
    
    return vlm_answer_captions