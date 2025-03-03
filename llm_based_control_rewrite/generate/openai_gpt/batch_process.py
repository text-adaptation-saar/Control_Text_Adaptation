import argparse
import json
import os
import time
from dotenv import dotenv_values
from openai import OpenAI
import re

from llm_based_control_rewrite.utils.helpers import load_yaml, extract_rewritten_sentences


def create_openai_client(config_yaml):
    env_vars = dotenv_values(".env")
    openai_key_name = None if "openai_key_name" not in config_yaml else config_yaml["openai_key_name"]
    if openai_key_name is None:
        print("OpenAI API key Initialization with keyname: OPEN_API_KEY_CSE ...")
        api_key = env_vars["OPEN_API_KEY_CSE"]
    else:
        print(f'OpenAI API key Initialization with keyname: {openai_key_name} ...')
        api_key = env_vars[openai_key_name]
    client = OpenAI(api_key=api_key)
    return client


def send_batch_request(batch_request_jsonl_file, config_yaml, batch_request_details_config):

    client = create_openai_client(config_yaml)

    # Create a file handle to upload
    batch_input_file = client.files.create(
        file=open(batch_request_jsonl_file, "rb"),
        purpose="batch"
    )
    time.sleep(120)  # Sleep for 120 seconds
    print("BATCH openai.files.create() Uploaded jsonl file response:")
    print(batch_input_file)
    batch_input_file_id = batch_input_file.id

    # Create a batch request
    response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    print("BATCH client.batches.create() response:")
    print(response)
    batch_id = response.id

    # Data to be written
    data = {
        "local_batch_input_file_path": batch_request_jsonl_file,
        "batch_input_file_id": batch_input_file_id,
        "batch_id": batch_id
    }

    # Write data to JSON file
    with open(batch_request_details_config, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Details written to {batch_request_details_config}")



def obtain_batch_response(batch_request_details_path, config_yaml ):

    client = create_openai_client(config_yaml)

    with open(batch_request_details_path, 'r') as file:
        data = json.load(file)
    batch_id = data['batch_id']
    print("Batch ID:", batch_id)

    retrieve_status = client.batches.retrieve(batch_id)
    print("BATCH  openai.batches.retrieve() response:")
    print(retrieve_status)

    if retrieve_status.output_file_id:
        output_file_id=retrieve_status.output_file_id
        content = client.files.content(output_file_id)
        print(f"BATCH obtained output_file_id: {output_file_id} content:")
        print(content)

        batch_output_file_path=data['local_batch_input_file_path'].replace("batch_request", "batch_response")
        with open(batch_output_file_path, 'wb') as file:
            file.write(content.content)
        print(f"Content successfully saved to {batch_output_file_path}")

        # Load data from JSONL file into a dictionary for quick access
        entries_by_custom_id = {}
        with open(batch_output_file_path, 'r') as file:
            for line in file:
                entry = json.loads(line.strip())
                # Assuming 'custom_id' is always present in each entry
                entries_by_custom_id[entry['custom_id']] = entry

        # Iterate from 1 to 200 and generate custom_id
        outputs=[]
        number_of_lines = int(config_yaml["number_of_lines_to_test"]) + 1
        for i in range(1, number_of_lines):
            custom_id = "request-{}".format(i)
            # Find the entry corresponding to the custom_id
            if custom_id in entries_by_custom_id:
                entry = entries_by_custom_id[custom_id]
                # Extract the assistant's content
                try:
                    content = entry['response']['body']['choices'][0]['message']['content']
                    assistant_content = " ".join(line.strip() for line in content.strip().splitlines())
                    rewritten_sentences = extract_rewritten_sentences(assistant_content)
                    rewritten_sentences = rewritten_sentences.replace('." "', '. ')
                    rewritten_sentences = rewritten_sentences.replace(' "', '')
                    rewritten_sentences = rewritten_sentences.replace('."', '.')
                    rewritten_sentences = rewritten_sentences.replace('.,', '. ')
                    rewritten_sentences = rewritten_sentences.replace('",', ', ')
                    # rewritten_sentences = extract_rewritten_sentences_custom(assistant_content)
                    outputs.append(rewritten_sentences.strip())
                    # print(f"Custom ID: {custom_id} - Assistant Content: {assistant_content}")
                    print(f"Custom ID: {custom_id} - Rewritten Sentences: {rewritten_sentences}")
                except KeyError as e:
                    print(f"KeyError: Missing key {e} in data for custom_id {custom_id}")
            else:
                print(f"No entry found for custom_id {custom_id}")

        directory_path = os.path.dirname(batch_output_file_path)
        with open(directory_path+"/output.txt", 'w') as file:
            # Join the list items into a single string with each item on a new line
            output_str = '\n'.join(outputs)
            file.write(output_str + "\n")

    return retrieve_status.output_file_id


def obtain_batch_response_for_one_line(batch_request_details_path, config_yaml, custom_id ):

    client = create_openai_client(config_yaml)

    with open(batch_request_details_path, 'r') as file:
        data = json.load(file)
    batch_id = data['batch_id']
    print(f"Batch ID: {batch_id}")

    retrieve_status = client.batches.retrieve(batch_id)
    print(f"BATCH  openai.batches.retrieve() response: {retrieve_status}")

    if retrieve_status.output_file_id:
        output_file_id=retrieve_status.output_file_id
        content = client.files.content(output_file_id)
        print(f"BATCH obtained output_file_id: {output_file_id} content: {content}")

        batch_output_file_path=data['local_batch_input_file_path'].replace("batch_request", "batch_response")
        with open(batch_output_file_path, 'wb') as file:
            file.write(content.content)
        print(f"Content successfully saved to {batch_output_file_path}")

        # Load data from JSONL file into a dictionary for quick access
        entries_by_custom_id = {}
        with open(batch_output_file_path, 'r') as file:
            for line in file:
                entry = json.loads(line.strip())
                # Assuming 'custom_id' is always present in each entry
                entries_by_custom_id[entry['custom_id']] = entry

        # Find the entry corresponding to the custom_id
        if custom_id in entries_by_custom_id:
            entry = entries_by_custom_id[custom_id]
            # Extract the assistant's content
            try:
                content = entry['response']['body']['choices'][0]['message']['content']
                assistant_content = " ".join(line.strip() for line in content.strip().splitlines())
                # rewritten_sentences = extract_rewritten_sentences(assistant_content)
                rewritten_sentences = extract_rewritten_sentences_for_no_system_prompt(assistant_content)
                rewritten_sentences = rewritten_sentences.replace('." "', '. ')
                rewritten_sentences = rewritten_sentences.replace(' "', '')
                rewritten_sentences = rewritten_sentences.replace('."', '.')
                rewritten_sentences = rewritten_sentences.replace('.,', '. ')
                rewritten_sentences = rewritten_sentences.replace('",', ', ')
                # rewritten_sentences = extract_rewritten_sentences_custom(assistant_content)
                # print(f"Custom ID: {custom_id} - Assistant Content: {assistant_content}")
                print(f"Custom ID: {custom_id} - Rewritten Sentences: {rewritten_sentences}")
                return retrieve_status.output_file_id, assistant_content, rewritten_sentences.strip()

            except KeyError as e:
                print(f"KeyError: Missing key {e} in data for custom_id {custom_id}")
        else:
            print(f"No entry found for custom_id {custom_id}")

    return retrieve_status.output_file_id, None, None


def extract_rewritten_sentences_for_no_system_prompt(final_out):
    # This pattern matches non-empty sequences inside curly braces
    pattern = r"\{([^}]+)\}"
    matches = re.findall(pattern, final_out)

    # Checking how many {} pairs are consecutively at the end
    end_pattern = r"(\{[^}]+\}\s*)+$"
    end_match = re.search(end_pattern, final_out)
    if end_match:
        # Counting the number of ending curly braces
        end_blocks = re.findall(pattern, end_match.group())
        # Join the end_blocks into a single string separated by a specified separator
        return " ".join(end_blocks)  # Join with a space or any other preferred separator
    else:
        # This pattern matches non-empty sequences inside curly braces
        pattern = r"\{([^}]+)\}"
        matches = re.findall(pattern, final_out)

        # Select the last match if there are any matches
        if matches:
            rewritten_sentence = matches[-1]  # Last match
        else:
            # rewritten_sentence = ""
            rewritten_sentence = final_out

        return rewritten_sentence


# def extract_rewritten_sentences_custom(final_out):
#     # This pattern matches text inside double quotes
#     pattern = r'\"([^\"]+)\"'
#     matches = re.findall(pattern, final_out)
#
#     # Select the first match if there are any matches
#     if matches:
#         extracted_sentence = matches[0]  # First match
#     else:
#         extracted_sentence = final_out
#
#     return extracted_sentence


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="yaml config file for prompt based generation")
    parser.add_argument("--batch_request_jsonl_file", required=False, help="Target sentence's dependency depth")
    parser.add_argument("--batch_request_details_path", required=False, help="Target sentence's dependency length")

    args = vars(parser.parse_args())
    config_yaml = load_yaml(args["config"])
    obtain_batch_response(args["batch_request_details_path"], config_yaml)