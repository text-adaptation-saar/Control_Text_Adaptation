import time
import backoff
import openai
from dotenv import dotenv_values
from llama_cpp import Llama


def prepare_messages_for_chat_model(system_prompt, user_prompt, examples, src, user_prompt_control_token, target_sent_ideal_max_dept_depth):
    messages = []
    system_message = {"role": "system", "content": system_prompt}
    test_input = {"role": "user",
                  "content":
                      user_prompt.replace(user_prompt_control_token, str(target_sent_ideal_max_dept_depth)).replace(
                          "{input_src}", src.strip()).strip()}
    messages.append(system_message)
    for example in examples:
        messages.append(example)
    messages.append(test_input)
    return messages

def prepare_messages_for_chat_model_wc(system_prompt, user_prompt, examples, src, control_token, target_sent_ideal_max_dept_depth):
    messages = []
    system_message = {"role": "system", "content": system_prompt.replace(control_token, str(target_sent_ideal_max_dept_depth))}
    test_input = {"role": "user", "content": user_prompt.replace("{input_src}", src.strip()).replace(control_token, str(target_sent_ideal_max_dept_depth)).strip()}
    messages.append(system_message)
    for example in examples:
        messages.append(example)
    messages.append(test_input)
    return messages


# def initialize_langchain_openai(openai_api_key, temperature):
#
#     os.environ["OPENAI_API_KEY"] = openai_api_key
#     os.environ["OPENAI_ORGANIZATION"] = "org-GSzyW0NGMxXMlaszASnHwz7L"
#     llm = OpenAI(model_name='text-davinci-003', temperature=temperature)
#     print(llm)
#     return llm

def generate_with_prompts_langchain(llm_model, few_shot_prompt_input, path_to_write_output_generation):
    time.sleep(300)  # Sleep for 3 seconds
    try:
        output = llm_model(few_shot_prompt_input)
        # print("input: " + few_shot_prompt_input + "\n output: "+ output.strip())
    except Exception as e:
        print(e)

    # output = llm_model(few_shot_prompt_input)
    # print("input: " + few_shot_prompt_input + "\n output: "+ output.strip())

    with open(path_to_write_output_generation, 'a') as fp:
        fp.writelines(output.strip() + "\n")

    return output.strip()

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=10)
def send_text_completion_model_request(temperature, final_prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=final_prompt,
        temperature=float(temperature),
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response

def generate_with_prompts_openai_textmodel(temperature, final_prompt, path_to_write_output_generation, path_to_write_input_data, src_input_sent):

    try:
        response = send_text_completion_model_request(temperature, final_prompt)
    except openai.error.RateLimitError:
        print("[ERROR] Rate limit exceeded. Please try again later...")
        time.sleep(120)  # Sleep for 120 seconds
        try:
            response = send_text_completion_model_request(temperature, final_prompt)
        except openai.error.RateLimitError:
            print("[ERROR] [2nd Round] Rate limit exceeded. Please try again later...")

    choice = response.get('choices')[0]
    # final_out = choice.get('text').strip().splitlines() [0] # Only take first sentence! (in max dep i used this one)
    final_out = " ".join(line.strip() for line in choice.get('text').strip().splitlines()) # join all sentences with space

    with open(path_to_write_output_generation, 'a') as fp_o:
        fp_o.writelines(final_out + "\n")
    with open(path_to_write_input_data, 'a') as fp_i:
        fp_i.writelines(src_input_sent + "\n")
    print("Prompt: %s\n Output_by_davinci: %s " %(final_prompt, final_out))
    return final_out

def set_api_key_if_required(config_yaml):
    if "gpt" in config_yaml["model"] or "davinci" in config_yaml["model"]:
        env_vars = dotenv_values(".env")
        # api_key = env_vars["OPEN_API_KEY_CSE"]
        api_key = env_vars["OPEN_API_KEY_COLI"]
        openai.api_key = api_key

def send_chat_completion_model_request(config_yaml, messages):

    print("OpenAI Chat Completion request parameters: model:%s, temperature:%s, max_tokens=%s"
          %(config_yaml["model"], config_yaml["temperature"], config_yaml["max_tokens"]))
    response = openai.ChatCompletion.create(
        model=config_yaml["model"],
        messages=messages,
        temperature=float(config_yaml["temperature"]),
        max_tokens=int(config_yaml["max_tokens"]),
    )
    return response

def generate_with_prompts_openai_chatmodel(config_yaml, final_prompt, path_to_write_output_generation, path_to_write_input_data, src_input_sent):

    try:
        response = send_chat_completion_model_request(config_yaml, final_prompt)
    except openai.error.RateLimitError:
        print("[ERROR] Rate limit exceeded. Please try again later...")
        time.sleep(60) # Sleep for 60 seconds
        try:
            response = send_chat_completion_model_request(config_yaml, final_prompt)
        except openai.error.RateLimitError:
            print("[ERROR] [2nd Round] Rate limit exceeded. Please try again later...")

    choice = response.get('choices')[0]
    message = choice.get('message')
    final_out = " ".join(line.strip() for line in message.get('content').strip().splitlines()) # join all sentences with space

    with open(path_to_write_output_generation, 'a') as fp_o:
        fp_o.writelines(final_out + "\n")
    with open(path_to_write_input_data, 'a') as fp_i:
        fp_i.writelines(src_input_sent + "\n")
    print("Prompt: %s\n Output_give_by_gpt4_chatmodel: %s " %(final_prompt, final_out))
    return final_out

def pick_examples_for_max_dep(config, src_sent_max_dep, target_sent_ideal_max_dept_len, requested_feature_dict_vals):
    number_of_examples = config["number_of_examples_to_include_in_prompt"]
    example_val_dataset_src = config["example_val_dataset_src"]
    example_val_dataset_tgt = config["example_val_dataset_tgt"]
    example_val_dataset_feature_values = config["example_val_dataset_feature_values"]
    count = 0
    examples = []

    if number_of_examples <= 0:
        return examples

    with open(example_val_dataset_feature_values, 'r') as fp:
        for i, line in enumerate(fp):
            # max_depth_source, 7, max_depth_target, 7, depth_ratio, 1.0
            line_splits = line.split(",")
            ex_src_line_max_dep = int(line_splits[1])
            ex_tgt_line_max_dep = int(line_splits[3])
            # print("!!!!! ex_src_line_max_dep:%s \t ex_tgt_line_max_dep:%s "% (str(ex_src_line_max_dep), str(ex_tgt_line_max_dep)))
            # Best match exactly src and tgt max dep values are match. else same tgt max dep and src maxdep better to be higher than tgt.
            if (src_sent_max_dep == ex_src_line_max_dep and target_sent_ideal_max_dept_len == ex_tgt_line_max_dep) or \
                    (target_sent_ideal_max_dept_len == ex_tgt_line_max_dep and ex_src_line_max_dep > target_sent_ideal_max_dept_len and requested_feature_dict_vals["MaxDep"] < 1 ) or \
                    (target_sent_ideal_max_dept_len == ex_tgt_line_max_dep and ex_src_line_max_dep < target_sent_ideal_max_dept_len and requested_feature_dict_vals["MaxDep"] > 1 ) or \
                    (target_sent_ideal_max_dept_len == ex_tgt_line_max_dep and ex_src_line_max_dep == target_sent_ideal_max_dept_len and requested_feature_dict_vals["MaxDep"] == 1):
                count += 1
                if count > number_of_examples:
                    break
                with open(example_val_dataset_src, 'r') as ex_src_file, open(example_val_dataset_tgt, 'r') as ex_tgt_file:
                    ex_src = ex_src_file.readlines()[i]
                    ex_tgt = ex_tgt_file.readlines()[i]
                    example_dict = {"i": ex_src.strip(), "o": ex_tgt.strip(),  "max_dept_len_output": target_sent_ideal_max_dept_len}
                    print("Examples: Requested_tgt_max_dep:%s\tFound_tgt_maxdep:%s\tFound_src_maxdep:%s \t %s \t %s" %
                          (str(target_sent_ideal_max_dept_len), str(ex_tgt_line_max_dep),
                           str(ex_src_line_max_dep),ex_src.strip(), ex_tgt.strip()))
                    examples.append(example_dict)
    return examples

def pick_examples_for_max_dep_for_gpt4(config, src_sent_max_dep, target_sent_ideal_max_dept_len, requested_feature_dict_vals, prompt):
    number_of_examples = config["number_of_examples_to_include_in_prompt"]
    example_val_dataset_src = config["example_val_dataset_src"]
    example_val_dataset_tgt = config["example_val_dataset_tgt"]
    example_val_dataset_feature_values = config["example_val_dataset_feature_values"]
    count = 0
    examples = []

    if number_of_examples <= 0:
        return examples

    with open(example_val_dataset_feature_values, 'r') as fp:
        for i, line in enumerate(fp):
            # max_depth_source, 7, max_depth_target, 7, depth_ratio, 1.0
            line_splits = line.split(",")
            ex_src_line_max_dep = int(line_splits[1])
            ex_tgt_line_max_dep = int(line_splits[3])
            # print("!!!!! ex_src_line_max_dep:%s \t ex_tgt_line_max_dep:%s "% (str(ex_src_line_max_dep), str(ex_tgt_line_max_dep)))
            # Best match exactly src and tgt max dep values are match. else same tgt max dep and src MaxDepDepth better to be higher than tgt.
            if (src_sent_max_dep == ex_src_line_max_dep and target_sent_ideal_max_dept_len == ex_tgt_line_max_dep) or \
                    (target_sent_ideal_max_dept_len == ex_tgt_line_max_dep and ex_src_line_max_dep > target_sent_ideal_max_dept_len and requested_feature_dict_vals["MaxDepDepth"] < 1 ) or \
                    (target_sent_ideal_max_dept_len == ex_tgt_line_max_dep and ex_src_line_max_dep < target_sent_ideal_max_dept_len and requested_feature_dict_vals["MaxDepDepth"] > 1 ) or \
                    (target_sent_ideal_max_dept_len == ex_tgt_line_max_dep and ex_src_line_max_dep == target_sent_ideal_max_dept_len and requested_feature_dict_vals["MaxDepDepth"] == 1):
                count += 1
                if count > number_of_examples:
                    break
                with open(example_val_dataset_src, 'r') as ex_src_file, open(example_val_dataset_tgt, 'r') as ex_tgt_file:
                    ex_src = ex_src_file.readlines()[i]
                    ex_tgt = ex_tgt_file.readlines()[i]
                    example_user = {"role": "user", "content": prompt.replace("{max_dept_len}", str(target_sent_ideal_max_dept_len)).replace("{input_src}",ex_src.strip())}
                    example_assistant = {"role": "assistant", "content": ex_tgt.strip()}
                    print("Examples: Requested_tgt_max_dep:%s\tFound_tgt_MaxDepDepth:%s\tFound_src_MaxDepDepth:%s \t %s \t %s" %
                          (str(target_sent_ideal_max_dept_len), str(ex_tgt_line_max_dep),
                           str(ex_src_line_max_dep),ex_src.strip(), ex_tgt.strip()))
                    examples.append(example_user)
                    examples.append(example_assistant)

    return examples

def pick_examples_for_max_dep_length_for_gpt4(config, src_sent_max_dep_length, target_sent_ideal_max_dep_length, requested_feature_dict_vals, user_prompt):
    number_of_examples = config["number_of_examples_to_include_in_prompt"]
    example_val_dataset_src = config["example_val_dataset_src"]
    example_val_dataset_tgt = config["example_val_dataset_tgt"]
    example_val_dataset_feature_values = config["example_val_dataset_feature_values"]
    count = 0
    examples = []

    if number_of_examples <= 0:
        return examples

    with open(example_val_dataset_feature_values, 'r') as fp:
        for i, line in enumerate(fp):
            # max_dep_length_source, 37, max_dep_length_target, 37, dep_length_ratio, 1.0
            line_splits = line.split(",")
            ex_src_line_max_dep_length = int(line_splits[1])
            ex_tgt_line_max_dep_length = int(line_splits[3])
            # Best match exactly src and tgt max dep values are match. else same tgt max dep and src maxdep better to be higher than tgt.
            if (src_sent_max_dep_length == ex_src_line_max_dep_length and target_sent_ideal_max_dep_length == ex_tgt_line_max_dep_length) or \
                    (target_sent_ideal_max_dep_length == ex_tgt_line_max_dep_length and ex_src_line_max_dep_length > target_sent_ideal_max_dep_length and requested_feature_dict_vals["MaxDepLength"] < 1 ) or \
                    (target_sent_ideal_max_dep_length == ex_tgt_line_max_dep_length and ex_src_line_max_dep_length < target_sent_ideal_max_dep_length and requested_feature_dict_vals["MaxDepLength"] > 1 ) or \
                    (target_sent_ideal_max_dep_length == ex_tgt_line_max_dep_length and ex_src_line_max_dep_length == target_sent_ideal_max_dep_length and requested_feature_dict_vals["MaxDepLength"] == 1):
                count += 1
                if count > number_of_examples:
                    break
                with open(example_val_dataset_src, 'r') as ex_src_file, open(example_val_dataset_tgt, 'r') as ex_tgt_file:
                    ex_src = ex_src_file.readlines()[i]
                    ex_tgt = ex_tgt_file.readlines()[i]
                    example_user = {"role": "user", "content": user_prompt.replace("{max_dep_length}", str(target_sent_ideal_max_dep_length)).replace("{input_src}",ex_src.strip())}
                    example_assistant = {"role": "assistant", "content": ex_tgt.strip()}
                    print("Examples: Requested_tgt_maxdeplength:%s\tFound_tgt_maxdeplength:%s\tFound_src_maxdeplength:%s \t %s \t %s" %
                          (str(target_sent_ideal_max_dep_length), str(ex_tgt_line_max_dep_length),
                           str(ex_src_line_max_dep_length),ex_src.strip(), ex_tgt.strip()))
                    examples.append(example_user)
                    examples.append(example_assistant)

    return examples

def pick_examples_for_wc_for_gpt4(config_yaml, src_input_sent_word_count, target_sent_ideal_word_count, requested_feature_dict_vals, user_prompt):
    number_of_examples = config_yaml["number_of_examples_to_include_in_prompt"]
    example_val_dataset_src = config_yaml["example_val_dataset_src"]
    example_val_dataset_tgt = config_yaml["example_val_dataset_tgt"]
    # example_val_dataset_feature_values = config["example_val_dataset_feature_values"]
    count = 0
    examples = []

    if number_of_examples <= 0:
        return examples

    with open(example_val_dataset_src, 'r') as ex_src_file, open(example_val_dataset_tgt, 'r') as ex_tgt_file:
        for i, (ex_src_line, ex_tgt_line) in enumerate(zip(ex_src_file, ex_tgt_file)):
            ex_src_line_word_count = len(ex_src_line.split(" "))
            ex_tgt_line_word_count = len(ex_tgt_line.split(" "))

            # EXAMPLE selection CASE-1: Best examples match exactly as src and tgt word count. Else same tgt max dep and src maxdep better to be higher than tgt.
            # if (src_sent_word_count == ex_src_line_word_count and target_sent_ideal_word_count == ex_tgt_line_word_count):

            # EXAMPLE selection CASE-2: Pick examples which has same tgt word count.
            if (src_input_sent_word_count == ex_src_line_word_count and target_sent_ideal_word_count == ex_tgt_line_word_count) or \
            (target_sent_ideal_word_count == ex_tgt_line_word_count and ex_src_line_word_count > target_sent_ideal_word_count and requested_feature_dict_vals["WordCount"] < 1 ) or \
            (target_sent_ideal_word_count == ex_tgt_line_word_count and ex_src_line_word_count < target_sent_ideal_word_count and requested_feature_dict_vals["WordCount"] > 1 ) or \
            (target_sent_ideal_word_count == ex_tgt_line_word_count and ex_src_line_word_count == target_sent_ideal_word_count and requested_feature_dict_vals["WordCount"] == 1):

            # Pick by same Ratio
            # found_word_count_ratio = word_count_ratio(ex_src_line, ex_tgt_line)
            # if round(float(requested_feature_dict_vals["Word_Count_Ratio"]), 1) == round(float(found_word_count_ratio),
            #                                                                              1):
                count += 1
                if count > number_of_examples:
                    break
                example_user = {"role": "user", "content": user_prompt.replace("{input_src}", ex_src_line.strip())}
                example_assistant = {"role": "assistant", "content": ex_tgt_line.strip()}
                examples.append(example_user)
                examples.append(example_assistant)
                print("Examples: Requested_tgt_wc:test_src_wc=%s:%s\tfound_ex_tgt_wc:found_ex_src_wc:%s:%s\t\t %s \t %s" %
                    (target_sent_ideal_word_count, src_input_sent_word_count, ex_tgt_line_word_count, ex_src_line_word_count,
                     ex_src_line.strip(), ex_tgt_line.strip()))
    return examples

def pick_examples_for_wc(config_yaml, src_input_sent_word_count, target_sent_ideal_word_count, requested_feature_dict_vals):
    number_of_examples = config_yaml["number_of_examples_to_include_in_prompt"]
    example_val_dataset_src = config_yaml["example_val_dataset_src"]
    example_val_dataset_tgt = config_yaml["example_val_dataset_tgt"]
    # example_val_dataset_feature_values = config["example_val_dataset_feature_values"]
    count = 0
    examples = []

    if number_of_examples <= 0:
        return examples

    with open(example_val_dataset_src, 'r') as ex_src_file, open(example_val_dataset_tgt, 'r') as ex_tgt_file:
        for i, (ex_src_line, ex_tgt_line) in enumerate(zip(ex_src_file, ex_tgt_file)):
            ex_src_line_word_count = len(ex_src_line.split(" "))
            ex_tgt_line_word_count = len(ex_tgt_line.split(" "))

            # EXAMPLE selection CASE-1: Best examples match exactly as src and tgt word count. Else same tgt max dep and src maxdep better to be higher than tgt.
            # if (src_sent_word_count == ex_src_line_word_count and target_sent_ideal_word_count == ex_tgt_line_word_count):

            # EXAMPLE selection CASE-2: Pick examples which has same tgt word count.
            if (src_input_sent_word_count == ex_src_line_word_count and target_sent_ideal_word_count == ex_tgt_line_word_count) or \
            (target_sent_ideal_word_count == ex_tgt_line_word_count and ex_src_line_word_count > target_sent_ideal_word_count and requested_feature_dict_vals["WordCount"] < 1 ) or \
            (target_sent_ideal_word_count == ex_tgt_line_word_count and ex_src_line_word_count < target_sent_ideal_word_count and requested_feature_dict_vals["WordCount"] > 1 ) or \
            (target_sent_ideal_word_count == ex_tgt_line_word_count and ex_src_line_word_count == target_sent_ideal_word_count and requested_feature_dict_vals["WordCount"] == 1):

            # Pick by same Ratio
            # found_word_count_ratio = word_count_ratio(ex_src_line, ex_tgt_line)
            # if round(float(requested_feature_dict_vals["Word_Count_Ratio"]), 1) == round(float(found_word_count_ratio),
            #                                                                              1):
                count += 1
                if count > number_of_examples:
                    break
                example = {"input": ex_src_line.strip(), "output": ex_tgt_line.strip()}
                examples.append(example)
                print("Examples: Requested_tgt_wc:test_src_wc=%s:%s\tfound_ex_tgt_wc:found_ex_src_wc:%s:%s\t\t %s \t %s" %
                    (target_sent_ideal_word_count, src_input_sent_word_count, ex_tgt_line_word_count, ex_src_line_word_count,
                     ex_src_line.strip(), ex_tgt_line.strip()))
    return examples

def pick_examples_for_maxdepdepth(config, src_sent_max_dep, target_sent_ideal_max_dept_len, requested_feature_dict_vals):
    number_of_examples = config["number_of_examples_to_include_in_prompt"]
    example_val_dataset_src = config["example_val_dataset_src"]
    example_val_dataset_tgt = config["example_val_dataset_tgt"]
    example_val_dataset_feature_values = config["example_val_dataset_feature_values"]
    count = 0
    examples = []

    if number_of_examples <= 0:
        return examples

    with open(example_val_dataset_feature_values, 'r') as fp:
        for i, line in enumerate(fp):
            # max_depth_source, 7, max_depth_target, 7, depth_ratio, 1.0
            line_splits = line.split(",")
            ex_src_line_max_dep = int(line_splits[1])
            ex_tgt_line_max_dep = int(line_splits[3])
            # print("!!!!! ex_src_line_max_dep:%s \t ex_tgt_line_max_dep:%s "% (str(ex_src_line_max_dep), str(ex_tgt_line_max_dep)))
            # Best match exactly src and tgt max dep values are match. else same tgt max dep and src MaxDepDepth better to be higher than tgt.
            if (src_sent_max_dep == ex_src_line_max_dep and target_sent_ideal_max_dept_len == ex_tgt_line_max_dep) or \
                    (target_sent_ideal_max_dept_len == ex_tgt_line_max_dep and ex_src_line_max_dep > target_sent_ideal_max_dept_len and requested_feature_dict_vals["MaxDepDepth"] < 1 ) or \
                    (target_sent_ideal_max_dept_len == ex_tgt_line_max_dep and ex_src_line_max_dep < target_sent_ideal_max_dept_len and requested_feature_dict_vals["MaxDepDepth"] > 1 ) or \
                    (target_sent_ideal_max_dept_len == ex_tgt_line_max_dep and ex_src_line_max_dep == target_sent_ideal_max_dept_len and requested_feature_dict_vals["MaxDepDepth"] == 1):
                count += 1
                if count > number_of_examples:
                    break
                with open(example_val_dataset_src, 'r') as ex_src_file, open(example_val_dataset_tgt, 'r') as ex_tgt_file:
                    ex_src = ex_src_file.readlines()[i]
                    ex_tgt = ex_tgt_file.readlines()[i]
                    example = {"input": ex_src.strip(), "output": ex_tgt.strip()}
                    examples.append(example)
                    print("Examples: Requested_tgt_max_dep:%s\tFound_tgt_MaxDepDepth:%s\tFound_src_MaxDepDepth:%s \t %s \t %s" %
                          (str(target_sent_ideal_max_dept_len), str(ex_tgt_line_max_dep),
                           str(ex_src_line_max_dep),ex_src.strip(), ex_tgt.strip()))

    return examples


def generate_with_prompts_llama_cpp(config_yaml, final_prompt, path_to_write_output_generation, path_to_write_input_data, src_input_sent):
    import subprocess
    # /proj/corpora/gpt4-x-vicuna-13B-HF/ggml-model-q4_0.bin
    # specify the command to run
    command = f"/proj/contrib/llama.cpp/main -m %s -p \"%s\" -n %s" % (config_yaml["model"].strip(), final_prompt.strip(), config_yaml["max_tokens"])
    print(command)
    # run the command
    process = subprocess.run(command, shell=True, text=True, capture_output=True)

    # check if the command ran successfully
    if process.returncode != 0:
        print(f"Execution of command failed.")
    else:
        # print(process.stdout)  # print the output of the command
        final_out = process.stdout.split(":") [-1].strip()

        with open(path_to_write_output_generation, 'a') as fp_o:
            fp_o.writelines(final_out + "\n")
        with open(path_to_write_input_data, 'a') as fp_i:
            fp_i.writelines(src_input_sent + "\n")
        # print("Prompt: %s\n Output_give_by_gpt4_chatmodel: %s " %(final_prompt, final_out))
        print("Output_via_llama.cpp: %s " %(final_out))
        return final_out

def generate_with_prompts_llama_cpp_python(config_yaml, final_prompt, path_to_write_full_output_generation,
                                           path_to_write_output_generation, path_to_write_input_data, src_input_sent):

    print("llama cpp python Completion request parameters: model:%s, max_tokens=%s, temperature=%s"
          % (config_yaml["model"], config_yaml["max_tokens"], config_yaml["temperature"]))
    llm = Llama(model_path=config_yaml["model"].strip())
    output = llm(final_prompt, max_tokens=config_yaml["max_tokens"], echo=True, temperature=config_yaml["temperature"])

    choice = output.get('choices')[0]
    output_string = " ".join(line.strip() for line in choice.get('text').strip().splitlines())  # join all sentences with space
    final_out= output_string.split(":") [-1].strip()

    with open(path_to_write_full_output_generation, 'a') as fp_o:
        fp_o.writelines(output_string + "\n")
    with open(path_to_write_output_generation, 'a') as fp_o:
        fp_o.writelines(final_out + "\n")
    with open(path_to_write_input_data, 'a') as fp_i:
        fp_i.writelines(src_input_sent + "\n")
    print("Prompt: %s\n Output_via_llama_cpp_python: %s " %(final_prompt, final_out))
    return final_out

def prepare_messages_for_llama_cpp_text_model(system_prompt, user_prompt, examples, src, control_token, control_token_value):

    system_prompt=system_prompt.replace(control_token, str(control_token_value))

    for example in examples:
        prompt = user_prompt.replace("{input_src}",example.get("input").strip()).strip().replace(control_token, str(control_token_value)) \
                 + " " +  example.get("output").strip()
        system_prompt += "\n" + prompt

    test_input=user_prompt.replace("{input_src}", src.strip()).replace(control_token, str(control_token_value))
    prompt = f"%s\n%s" % (system_prompt, test_input)
    return prompt

