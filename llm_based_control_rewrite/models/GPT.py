import time

import numpy as np
from dotenv import dotenv_values

from llm_based_control_rewrite.generate.openai_gpt.batch_process import send_batch_request, obtain_batch_response_for_one_line
from llm_based_control_rewrite.models.LLModels import LLModels
import openai
from llm_based_control_rewrite.utils.feature_extraction import get_max_dep_depth_of_given_sent, \
    get_target_feature_value, get_max_dep_length_of_given_sent, dependency_tree_print_with_depth, \
    dependency_tree_print_with_length, get_freq_rank_of_given_sent, \
    get_word_count_of_given_sent, get_no_of_difficult_words_of_given_sent, print_difficult_words, print_word_count, \
    split_sentences, get_grade_level_of_given_sent, print_char_list, get_length_of_given_sent
import pandas as pd

import re

from llm_based_control_rewrite.utils.helpers import read_content, read_csv, extract_rewritten_sentences
import json
import pandas as pd
import time
from datetime import datetime

class GPT(LLModels):

    def __init__(self,
                 requested_dependency_depth,
                 requested_dependency_length,
                 requested_difficult_words,
                 requested_word_count,
                 requested_frequency,
                 requested_length,
                 requested_levenshtein,
                 model_path,
                 temperature,
                 system_prompt = "",
                 user_prompt = "",
                 output_prompt = "",
                 requested_absolute_feature_value=False,
                 openai_key_name=None,
                 requested_grade=None
                 ):

        super().__init__(requested_dependency_depth=requested_dependency_depth,
                         requested_dependency_length=requested_dependency_length,
                         requested_difficult_words=requested_difficult_words,
                         requested_word_count=requested_word_count,
                         requested_frequency=requested_frequency,
                         requested_length=requested_length,
                         requested_levenshtein=requested_levenshtein,
                         requested_absolute_feature_value=requested_absolute_feature_value,
                         requested_grade=requested_grade)

        self.model_path=model_path
        self.temperature=temperature
        self.env_vars = dotenv_values(".env")
        self.client=None
        if "gpt" in self.model_path or "davinci" in self.model_path:
            if openai_key_name is None:
                print("OpenAI API key Initialization with keyname: OPEN_API_KEY_CSE ...")
                api_key = self.env_vars["OPEN_API_KEY_CSE"]
            else:
                print(f'OpenAI API key Initialization with keyname: {openai_key_name} ...')
                api_key = self.env_vars[openai_key_name]
            # api_key = self.env_vars["OPEN_API_KEY_COLI"]
            openai.api_key = api_key
        else:
            print(f"Client Initialization for model: {model_path}")
            self.client = openai.Client(base_url=self.model_path, api_key=self.env_vars[openai_key_name])

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.output_prompt = output_prompt
        self.given_sentences_feature_dict = {}
        self.examples_messages=[]

    def prepare_prompt_for_multiple_features(self, config_yaml, test_src_sent, output_generation_path, calibration_round=0):

        self.number_of_examples = config_yaml["number_of_examples_to_include_in_prompt"]
        print("Prepare prompt for multiple features")

        for index, feature in enumerate(config_yaml["feature_names"].split(','), start=0):
            fussy_range_for_feature_list = [int(x.strip()) for x in config_yaml["feature_range"].split(',')] if config_yaml["feature_range"] else [0,0,0,0,0]

            print(f"feature: {feature}")
            # Constructing the regex pattern with the variable
            pattern = rf"\{{[^{{}}]*_{re.escape(feature)}\}}"
            # if feature == "MaxDepDepth" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
            if feature == "MaxDepDepth":
                # Initialize the nested dictionary for 'MaxDepDepth' if it doesn't exist
                self.given_sentences_feature_dict.setdefault("MaxDepDepth", {})
                self.given_sentences_feature_dict["MaxDepDepth"] ["src"] = get_max_dep_depth_of_given_sent(test_src_sent['src'])
                self.given_sentences_feature_dict["MaxDepDepth"] ["tgt_ideal"] = get_target_feature_value(self.given_sentences_feature_dict["MaxDepDepth"] ["src"],
                                                                           self.dependency_depth,
                                                                           self.requested_absolute_feature_value)
                self.given_sentences_feature_dict["MaxDepDepth"] ["tgt_ideal_range"] = fussy_range_for_feature_list[index]
                # if  self.given_sentences_feature_dict["MaxDepDepth"] ["tgt_ideal"] == 0:
                #     self.given_sentences_feature_dict["MaxDepDepth"]["tgt_ideal"] = 1
                print("hit max_dep_depth prompt: src_max_dep_depth=%s,\t ideal_tgt_max_dep_depth=%s" % (self.given_sentences_feature_dict["MaxDepDepth"] ["src"],
                                                                                                        self.given_sentences_feature_dict["MaxDepDepth"] ["tgt_ideal"] ))

            # if feature == "MaxDepLength" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
            if feature == "MaxDepLength":
                # Initialize the nested dictionary for 'MaxDepDepth' if it doesn't exist
                self.given_sentences_feature_dict.setdefault("MaxDepLength", {})
                self.given_sentences_feature_dict["MaxDepLength"] ["src"] = get_max_dep_length_of_given_sent(test_src_sent['src'])
                self.given_sentences_feature_dict["MaxDepLength"] ["tgt_ideal"]  = get_target_feature_value( self.given_sentences_feature_dict["MaxDepLength"] ["src"],
                                                                             self.dependency_length,
                                                                             self.requested_absolute_feature_value)
                self.given_sentences_feature_dict["MaxDepLength"] ["tgt_ideal_range"] = fussy_range_for_feature_list[index]

                # if self.given_sentences_feature_dict["MaxDepLength"]["tgt_ideal"] == 0:
                #     self.given_sentences_feature_dict["MaxDepLength"]["tgt_ideal"] = 1
                print("hit max_dept_length prompt: src_max_dept_length=%s,\t ideal_tgt_max_dept_length=%s" % ( self.given_sentences_feature_dict["MaxDepLength"] ["src"],
                                                                                                               self.given_sentences_feature_dict["MaxDepLength"] ["tgt_ideal"]))

            # if feature == "DiffWords" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
            if feature == "DiffWords":
                # Initialize the nested dictionary for 'MaxDepDepth' if it doesn't exist
                self.given_sentences_feature_dict.setdefault("DiffWords", {})
                self.given_sentences_feature_dict["DiffWords"] ["src"] = get_no_of_difficult_words_of_given_sent(test_src_sent['src'])
                self.given_sentences_feature_dict["DiffWords"] ["tgt_ideal"] = get_target_feature_value(self.given_sentences_feature_dict["DiffWords"] ["src"],
                                                                             self.difficult_words,
                                                                             self.requested_absolute_feature_value)
                self.given_sentences_feature_dict["DiffWords"] ["tgt_ideal_range"] = fussy_range_for_feature_list[index]

                print("hit difficult_words prompt: src_difficult_words=%s,\t ideal_tgt_difficult_words=%s" % (self.given_sentences_feature_dict["DiffWords"] ["src"],
                                                                                                              self.given_sentences_feature_dict["DiffWords"] ["tgt_ideal"] ))

            # if feature == "WordCount" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
            if feature == "WordCount":
                # Initialize the nested dictionary for 'MaxDepDepth' if it doesn't exist
                self.given_sentences_feature_dict.setdefault("WordCount", {})
                print(f"test src: {test_src_sent['src']} and wordcount: {get_word_count_of_given_sent(test_src_sent['src'])} ")
                self.given_sentences_feature_dict["WordCount"] ["src"]  = get_word_count_of_given_sent(test_src_sent['src'])
                self.given_sentences_feature_dict["WordCount"] ["tgt_ideal"]  = get_target_feature_value(self.given_sentences_feature_dict["WordCount"] ["src"],
                                                                        self.word_count,
                                                                        self.requested_absolute_feature_value)
                self.given_sentences_feature_dict["WordCount"] ["tgt_ideal_range"] = fussy_range_for_feature_list[index]

                print("hit word_count prompt: src_word_count=%s,\t ideal_tgt_word_count=%s" % (self.given_sentences_feature_dict["WordCount"] ["src"],
                                                                                               self.given_sentences_feature_dict["WordCount"] ["tgt_ideal"]  ))

            # if feature == "Length" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
            if feature == "Length":
                # Initialize the nested dictionary for 'MaxDepDepth' if it doesn't exist
                self.given_sentences_feature_dict.setdefault("Length", {})
                print(f"test src: {test_src_sent['src']} and Length: {get_length_of_given_sent(test_src_sent['src'])} ")
                self.given_sentences_feature_dict["Length"] ["src"]  = get_length_of_given_sent(test_src_sent['src'])
                self.given_sentences_feature_dict["Length"] ["tgt_ideal"]  = get_target_feature_value(self.given_sentences_feature_dict["Length"] ["src"],
                                                                        self.length,
                                                                        self.requested_absolute_feature_value)
                self.given_sentences_feature_dict["Length"] ["tgt_ideal_range"] = fussy_range_for_feature_list[index]

                print("hit Length prompt: src_length=%s,\t ideal_tgt_length=%s" % (self.given_sentences_feature_dict["Length"] ["src"],
                                                                                               self.given_sentences_feature_dict["Length"] ["tgt_ideal"]  ))

            # if feature == "FreqRank" and "{frequen/cy_category}" in self.system_prompt or "{frequency_category}" in self.user_prompt:
            #     print("hit frequency_category prompt, we will convert freq_rank to freq category")
            #     # Initialize the nested dictionary for 'MaxDepDepth' if it doesn't exist
            #     self.given_sentences_feature_dict.setdefault("FreqRank", {})
            #     self.given_sentences_feature_dict["FreqRank"] ["src"]   = get_freq_rank_of_given_sent(test_src_sent['src'])
            #     self.given_sentences_feature_dict["FreqRank"] ["tgt_ideal"]   = get_target_feature_value(self.given_sentences_feature_dict["FreqRank"] ["src"],
            #                                                                  self.frequency,
            #                                                                  self.requested_absolute_feature_value)
            #     target_sent_freq_category = self.freq_rank_to_freq_catagory(self.given_sentences_feature_dict["FreqRank"] ["tgt_ideal"])
            #     print("hit frequency rank prompt: src_freq_rank=%s,\t ideal_tgt_freq_rank=%s\t"
            #           "ideal_freq_rank_category: is %s" % (self.given_sentences_feature_dict["FreqRank"] ["src"],
            #                                                self.given_sentences_feature_dict["FreqRank"] ["tgt_ideal"], target_sent_freq_category))

        # if feature == "Grade":
        # Initialize the nested dictionary for 'Grade' if it doesn't exist
        self.given_sentences_feature_dict.setdefault("Grade", {})
        self.given_sentences_feature_dict["Grade"]["src"] = get_grade_level_of_given_sent(test_src_sent['src'])
        self.given_sentences_feature_dict["Grade"]["tgt_ideal"] = int(get_target_feature_value(
            self.given_sentences_feature_dict["Grade"]["src"],
            self.grade,
            self.requested_absolute_feature_value))
        self.given_sentences_feature_dict["Grade"]["tgt_ideal_range"] = fussy_range_for_feature_list[index]

        print("hit grade-level prompt: src_grade=%s,\t ideal_tgt_grade=%s" % (
            self.given_sentences_feature_dict["Grade"]["src"],
            self.given_sentences_feature_dict["Grade"]["tgt_ideal"]))

        examples = self.pick_grade_level_examples(config_yaml)

        if (self.number_of_examples > 0 and len(examples) == 0):
            print("!!! hit no examples found, skipping !!!")
            return None, None
        else:
            # print("Found_example_count: %s \tPrompt: %s\n"%(len(examples),messages))
            print("Found_example_count: %s "%(len(examples)))
            messages = self.prepare_messages_for_chat_model_with_multiple_features(config_yaml, examples, test_src_sent['src'])
            # output evaluation wrt. each feature
            # Obtaining the list of keys
            keys_list = list(self.given_sentences_feature_dict.keys())
            # Extracting 'src' and 'tgt_ideal' values in the order of the keys
            src_feature_values = [self.given_sentences_feature_dict[key]["src"] for key in keys_list if
                                  "src" in self.given_sentences_feature_dict[key]]
            tgt_ideal_values = [self.given_sentences_feature_dict[key]["tgt_ideal"] for key in keys_list if
                                "tgt_ideal" in self.given_sentences_feature_dict[key]]
            tgt_ideal_range = [self.given_sentences_feature_dict[key]["tgt_ideal_range"] for key in keys_list if
                                "tgt_ideal_range" in self.given_sentences_feature_dict[key]]

            self.write_test_prompt_details(len(examples), str(keys_list), str(src_feature_values),
                                           str(tgt_ideal_values), output_generation_path, line_number=test_src_sent['line'], calibration_round= calibration_round,
                                           feature_value_fussy_range=tgt_ideal_range)
            # print(f"Found_example_count: {len(examples)}\t messages:{messages}")
            return messages, len(examples)


    def generate_with_chain_of_thought(self, test_src_sent, config_yaml, path_to_write_output_generation,
                 path_to_write_input_data, path_to_gold_ref, output_generation_path, batch_request_jsonl_file=None):

        # messages, found_example_count = self.prepare_prompt(config_yaml, test_src_sent, output_generation_path)
        messages, found_example_count = self.prepare_prompt_for_multiple_features(config_yaml, test_src_sent, output_generation_path)

        # examples not found cases for FS
        if messages is None:
            return None, None, None
        else:
            batch_request = bool(config_yaml.get("batch_request", False))
            cot_feedback_loop = bool(config_yaml.get("chain_of_thought", False))

            if batch_request and not cot_feedback_loop:
                print(f"BATCH request enabled, adding request for line-{test_src_sent['line']}")
                output_generation_path_array = output_generation_path.split("/")
                batch_request_jsonl_file = f"{output_generation_path}/batch_request_{output_generation_path_array[-2]}_{output_generation_path_array[-1]}.jsonl"
                request_json_for_a_line, custom_id_for_each_request = self.prepare_jsonl_batch_object(config_yaml, messages, test_src_sent["line"])
                print("JSONL object:\n" + str(request_json_for_a_line))
                with open(batch_request_jsonl_file, 'a') as fp_fo:
                    fp_fo.writelines(request_json_for_a_line + "\n")
                final_out = "Created batch request jsonl file!"
            elif batch_request and cot_feedback_loop:
                print(f"BATCH with Feedbackloop enabled, adding request for line-{test_src_sent['line']}")
                output_generation_path_array = output_generation_path.split("/")
                batch_request_jsonl_file = f"{output_generation_path}/batch_request_for_line-{test_src_sent['line']}_{output_generation_path_array[-2]}_{output_generation_path_array[-1]}.jsonl"
                request_json_for_a_line, custom_id_for_each_request = self.prepare_jsonl_batch_object(config_yaml, messages,
                                                                                        test_src_sent["line"])
                print("JSONL object:\n" + str(request_json_for_a_line))
                with open(batch_request_jsonl_file, 'w') as fp_fo:
                    fp_fo.writelines(request_json_for_a_line + "\n")


                batch_request_details_config = f'{config_yaml["path_to_output_generation"]}/line-{test_src_sent["line"]}-grade_{self.grade}-batch_request_details.json'
                send_batch_request(batch_request_jsonl_file, config_yaml, batch_request_details_config)

                output_file_id = None
                while not output_file_id:
                    print("Sleeping for 60s...")
                    time.sleep(60)  # Sleep for 120 seconds
                    output_file_id, assistant_response_content, rewritten_sentences = obtain_batch_response_for_one_line(batch_request_details_config, config_yaml,
                                                                        custom_id_for_each_request)
                print(f"Assistant_Response:- {assistant_response_content}\n"
                      f"Extracted_rewritten-sentece(s):- {rewritten_sentences}")


                for i in range(1,11):
                    if i == 1:
                        # save first round results for CoT FS case (w/o feedback).
                        with open(output_generation_path + "/CoT_FS_iternation-1_full_output_response.txt", 'a') as fp_fo:
                            fp_fo.writelines(assistant_response_content + "\n")
                        with open(output_generation_path + "/CoT_FS_iternation-1_output.txt", 'a') as fp_fo:
                            fp_fo.writelines(rewritten_sentences + "\n")
                    assistant_response_content, rewritten_sentences = self.cot_execution_method_as_batch(test_src_sent["line"],
                                                                                                         batch_request_jsonl_file,
                                                                                                         config_yaml,
                                                                                                         assistant_response_content,
                                                                                                         messages,
                                                                                                         rewritten_sentences,
                                                                                                         i,
                                                                                                         output_generation_path)

                # after cot/ or calibration or default, this is the common output writing code.
                with open(output_generation_path + "/full_output_response.txt", 'a') as fp_fo:
                    fp_fo.writelines(assistant_response_content + "\n")
                with open(path_to_write_output_generation, 'a') as fp_o:
                    fp_o.writelines(rewritten_sentences + "\n")
                final_out = assistant_response_content
            else:
                # return "final_out", messages, found_example_count
                response, messages = self.chat_completion_request(config_yaml, messages)
                print("Response:\n" + str(response))

                # choice = response.get('choices')[0]
                # message = choice.get('message')
                # final_out = " ".join(line.strip() for line in message.get('content').strip().splitlines())  # join all sentences with space

                # choice = response.choices[0].message.content
                choice = response.choices[0]
                message = choice.message
                final_out = " ".join(line.strip() for line in message.content.strip().splitlines())  # join all sentences with space
                rewritten_sentences = extract_rewritten_sentences(final_out)

                if bool(config_yaml.get("chain_of_thought", False)):
                    for i in range(1,11):
                        if i == 1:
                            # save first round results for CoT FS case (w/o feedback).
                            with open(output_generation_path + "/CoT_FS_iternation-1_full_output_response.txt", 'a') as fp_fo:
                                fp_fo.writelines(final_out + "\n")
                            with open(output_generation_path + "/CoT_FS_iternation-1_output.txt", 'a') as fp_fo:
                                fp_fo.writelines(rewritten_sentences + "\n")
                        final_out, rewritten_sentences, messages = self.cot_execution_method(test_src_sent, config_yaml, final_out, messages,
                                                                               rewritten_sentences,i, output_generation_path)

                if bool(config_yaml.get("calibration", False)):
                    for i in range(1):
                        for requested_feature_key in self.given_sentences_feature_dict.keys():
                            if "MaxDepDepth" == requested_feature_key:
                                llm_generated_output_feature_value = get_max_dep_depth_of_given_sent(rewritten_sentences)
                                src_feature_value = self.given_sentences_feature_dict["MaxDepDepth"] ["src"]
                                obtained_ratio = round(llm_generated_output_feature_value/src_feature_value, 2)
                                print(f"MaxDepDepth Requested ratio: {self.dependency_depth} \t Obtained ratio: {obtained_ratio}")
                                if round(self.dependency_depth, 1) != round(obtained_ratio,1) \
                                    and self.requested_absolute_feature_value == False:
                                    new_ratio = (self.dependency_depth/obtained_ratio) * self.dependency_depth
                                    self.dependency_depth = new_ratio

                            if "MaxDepLength" == requested_feature_key:
                                llm_generated_output_feature_value = get_max_dep_length_of_given_sent(rewritten_sentences)
                                src_feature_value = self.given_sentences_feature_dict["MaxDepLength"]["src"]
                                obtained_ratio = round(llm_generated_output_feature_value / src_feature_value, 2)
                                print(f"MaxDepLength Requested ratio: {self.dependency_length} \t Obtained ratio: {obtained_ratio}")
                                if round(self.dependency_length, 1) != round(obtained_ratio, 1) \
                                        and self.requested_absolute_feature_value == False:
                                    new_ratio = (self.dependency_length / obtained_ratio) * self.dependency_length
                                    self.dependency_length = new_ratio

                            if "DiffWords" == requested_feature_key:
                                llm_generated_output_feature_value = get_no_of_difficult_words_of_given_sent(rewritten_sentences)
                                src_feature_value = self.given_sentences_feature_dict["DiffWords"]["src"]
                                obtained_ratio = round(((llm_generated_output_feature_value if llm_generated_output_feature_value !=0 else 0.5) / src_feature_value), 2)
                                print(f"DiffWords Requested ratio: {self.difficult_words} \t Obtained ratio: {obtained_ratio}")
                                print(f"Skipping calibration for DiffWords")
                                # if round(self.difficult_words, 1) != round(obtained_ratio, 1) \
                                #         and self.requested_absolute_feature_value == False:
                                #     new_ratio = (self.difficult_words / obtained_ratio) * self.difficult_words
                                #     self.difficult_words = new_ratio

                            if "WordCount" == requested_feature_key:
                                llm_generated_output_feature_value = get_word_count_of_given_sent(rewritten_sentences)
                                src_feature_value = self.given_sentences_feature_dict["WordCount"]["src"]
                                obtained_ratio = round(llm_generated_output_feature_value / src_feature_value, 2)
                                print(f"WordCount Requested ratio: {self.word_count} \t Obtained ratio: {obtained_ratio}")
                                if round(self.word_count, 1) != round(obtained_ratio, 1) \
                                        and self.requested_absolute_feature_value == False:
                                    new_ratio = (self.word_count / obtained_ratio) * self.word_count
                                    self.word_count = new_ratio
                        # calibration round-1 after adjusting the ratios
                        messages, found_example_count = self.prepare_prompt_for_multiple_features(config_yaml,
                                                                                                  test_src_sent,
                                                                                                  output_generation_path,
                                                                                                  calibration_round=i+1)
                        response, messages = self.chat_completion_request(config_yaml, messages)
                        print("Response:\n" + str(response))

                        choice = response.get('choices')[0]
                        message = choice.get('message')
                        final_out = " ".join(line.strip() for line in message.get('content').strip().splitlines())  # join all sentences with space
                        rewritten_sentences = extract_rewritten_sentences(final_out)

                # after cot/ or calibration or default, this is the common output writing code.
                with open(output_generation_path +"/full_output_response.txt", 'a') as fp_fo:
                    fp_fo.writelines(final_out + "\n")
                with open(path_to_write_output_generation, 'a') as fp_o:
                    fp_o.writelines(rewritten_sentences + "\n")

            with open(path_to_write_input_data, 'a') as fp_i:
                fp_i.writelines(test_src_sent['src'] + "\n")
            # write ref sent.
            with open(path_to_gold_ref, 'a') as fp_gr:
                fp_gr.writelines(test_src_sent['gold_ref'] + "\n")
            print("prompt_given_to_request: %s\n Output_give_by_OpenAI_chatmodel: %s " % (messages, final_out))
            return final_out, messages, found_example_count

    def cot_execution_method(self, test_src_sent, config_yaml, final_out, messages, rewritten_sentences, iteration, output_generation_path):
        check_any_feature_mismatch = False
        for requested_feature_key in self.given_sentences_feature_dict.keys():
            if "MaxDepDepth" == requested_feature_key and \
                    get_max_dep_depth_of_given_sent(rewritten_sentences) != \
                    self.given_sentences_feature_dict["MaxDepDepth"]["tgt_ideal"]:
                check_any_feature_mismatch = True
                self.given_sentences_feature_dict["MaxDepDepth"]["src"] = get_max_dep_depth_of_given_sent(
                    rewritten_sentences)
                print(f"CoT Feedback check for {requested_feature_key} requested: {self.given_sentences_feature_dict['MaxDepDepth']['tgt_ideal']}"
                    f"\t Obtained: {self.given_sentences_feature_dict['MaxDepDepth']['src']}")

            if "MaxDepLength" == requested_feature_key and \
                    get_max_dep_length_of_given_sent(rewritten_sentences) != \
                    self.given_sentences_feature_dict["MaxDepLength"]["tgt_ideal"]:
                check_any_feature_mismatch = True
                self.given_sentences_feature_dict["MaxDepLength"]["src"] = get_max_dep_length_of_given_sent(
                    rewritten_sentences)
                print(
                    f"CoT Feedback check for {requested_feature_key} requested: {self.given_sentences_feature_dict['MaxDepLength']['tgt_ideal']}"
                    f"\t Obtained: {self.given_sentences_feature_dict['MaxDepLength']['src']}")

            if "DiffWords" == requested_feature_key and \
                    get_no_of_difficult_words_of_given_sent(rewritten_sentences) != \
                    self.given_sentences_feature_dict["DiffWords"]["tgt_ideal"]:
                check_any_feature_mismatch = True
                self.given_sentences_feature_dict["DiffWords"]["src"] = get_no_of_difficult_words_of_given_sent(
                    rewritten_sentences)
                print(
                    f"CoT Feedback check for {requested_feature_key} requested: {self.given_sentences_feature_dict['DiffWords']['tgt_ideal']}"
                    f"\t Obtained: {self.given_sentences_feature_dict['DiffWords']['src']}")

            if "WordCount" == requested_feature_key and \
                    get_word_count_of_given_sent(rewritten_sentences) != self.given_sentences_feature_dict["WordCount"][
                "tgt_ideal"]:
                check_any_feature_mismatch = True
                self.given_sentences_feature_dict["WordCount"]["src"] = get_word_count_of_given_sent(
                    rewritten_sentences)
                print(
                    f"CoT Feedback check for {requested_feature_key} requested: {self.given_sentences_feature_dict['WordCount']['tgt_ideal']}"
                    f"\t Obtained: {self.given_sentences_feature_dict['WordCount']['src']}")

        if check_any_feature_mismatch:
            print(f"Executing Chain-of-Thought prompting - iteration: {iteration}")
            messages.append({"role": "assistant", "content": final_out})
            if not rewritten_sentences or all(not sentence.strip() for sentence in rewritten_sentences):
                # This block will execute if rewritten_sentences is empty or contains only whitespace strings
                print("The rewritten_sentences list is empty or contains only whitespace.")

                user_chain_prompt = "" if "cot_reasoning_user_prompt" not in config_yaml else read_csv(
                    csv_file=config_yaml["prompt_csv"],
                    level_value=config_yaml["cot_reasoning_user_prompt"],
                    column_name="cot_reason_empty")
                user_chain_message = {"role": "user", "content": user_chain_prompt.strip()}

            else:
                # This block will execute if rewritten_sentences contains non-whitespace strings
                print("The rewritten_sentences list contains valid sentences.")
                user_chain_prompt = "" if "cot_reasoning_user_prompt" not in config_yaml else read_csv(
                    csv_file=config_yaml["prompt_csv"],
                    level_value=config_yaml["cot_reasoning_user_prompt"],
                    column_name="cot_reason")
                user_chain_message = {"role": "user", "content": user_chain_prompt.replace(
                    "{output_text}", rewritten_sentences).replace("{print_dependency_tree_with_depth}",
                                                                  str(dependency_tree_print_with_depth(
                                                                      config_yaml["lang"],
                                                                      rewritten_sentences))).replace(
                    "{print_dependency_tree_with_length}",
                    str(dependency_tree_print_with_length(config_yaml["lang"], rewritten_sentences))).replace(
                    "{print_difficult_words_list}", str(print_difficult_words(rewritten_sentences))).replace(
                    "{print_word_list}", str(print_word_count(rewritten_sentences))).replace(
                    "{no_of_sentences}", str(len(split_sentences(rewritten_sentences)))).replace(
                    "{print_char_list}", str(print_char_list(rewritten_sentences))).strip()}


            user_chain_message = self.fill_prompt_with_feature_values(user_chain_message,
                                                                      self.given_sentences_feature_dict)
            messages.append(user_chain_message)
            updated_messages=messages
            if iteration == 1 and self.number_of_examples > 3:
                # Remove elements at indices 1, 2, 3, 4, 5, 6
                updated_messages = messages[:1] + messages[7:]
                self.number_of_examples= self.number_of_examples - 3
                print(f"iteration - 1: Setting self.number_of_examples == {self.number_of_examples}")
            if iteration > 2:
                keep_first_elements = 1 + self.number_of_examples * 2 + 1
                updated_messages = messages[:keep_first_elements] + messages[keep_first_elements+2:]

            messages = updated_messages
            response, messages = self.chat_completion_request(config_yaml, messages)
            print("Response:\n" + str(response))

            # choice = response.get('choices')[0]
            # message = choice.get('message')
            # final_out = " ".join(
            #     line.strip() for line in message.get('content').strip().splitlines())  # join all sentences with space
            choice = response.choices[0]
            message = choice.message
            final_out = " ".join(
                line.strip() for line in message.content.strip().splitlines())  # join all sentences with space
            rewritten_sentences = extract_rewritten_sentences(final_out)
            # save first round results for CoT FS case (w/o feedback).
            with open(output_generation_path + "/Feedback_loop_iternations.csv", 'a') as fp_fo:
                fp_fo.writelines(f"line_number, {test_src_sent['line']}, iternation, {iteration}, response, {final_out}\n")

        return final_out, rewritten_sentences, messages

    def cot_execution_method_as_batch(self, test_src_line, batch_request_jsonl_file, config_yaml, assistant_response_content, messages, rewritten_sentences, iteration, output_generation_path):
        check_any_feature_mismatch = False
        for requested_feature_key in self.given_sentences_feature_dict.keys():
            if "MaxDepDepth" == requested_feature_key and \
                    get_max_dep_depth_of_given_sent(rewritten_sentences) != \
                    self.given_sentences_feature_dict["MaxDepDepth"]["tgt_ideal"]:
                check_any_feature_mismatch = True
                self.given_sentences_feature_dict["MaxDepDepth"]["src"] = get_max_dep_depth_of_given_sent(
                    rewritten_sentences)
                print(f"CoT Feedback check for {requested_feature_key} requested: {self.given_sentences_feature_dict['MaxDepDepth']['tgt_ideal']}"
                    f"\t Obtained: {self.given_sentences_feature_dict['MaxDepDepth']['src']}")

            if "MaxDepLength" == requested_feature_key and \
                    get_max_dep_length_of_given_sent(rewritten_sentences) != \
                    self.given_sentences_feature_dict["MaxDepLength"]["tgt_ideal"]:
                check_any_feature_mismatch = True
                self.given_sentences_feature_dict["MaxDepLength"]["src"] = get_max_dep_length_of_given_sent(
                    rewritten_sentences)
                print(
                    f"CoT Feedback check for {requested_feature_key} requested: {self.given_sentences_feature_dict['MaxDepLength']['tgt_ideal']}"
                    f"\t Obtained: {self.given_sentences_feature_dict['MaxDepLength']['src']}")

            if "DiffWords" == requested_feature_key and \
                    get_no_of_difficult_words_of_given_sent(rewritten_sentences) != \
                    self.given_sentences_feature_dict["DiffWords"]["tgt_ideal"]:
                check_any_feature_mismatch = True
                self.given_sentences_feature_dict["DiffWords"]["src"] = get_no_of_difficult_words_of_given_sent(
                    rewritten_sentences)
                print(
                    f"CoT Feedback check for {requested_feature_key} requested: {self.given_sentences_feature_dict['DiffWords']['tgt_ideal']}"
                    f"\t Obtained: {self.given_sentences_feature_dict['DiffWords']['src']}")

            if "WordCount" == requested_feature_key and \
                    get_word_count_of_given_sent(rewritten_sentences) != self.given_sentences_feature_dict["WordCount"][
                "tgt_ideal"]:
                check_any_feature_mismatch = True
                self.given_sentences_feature_dict["WordCount"]["src"] = get_word_count_of_given_sent(
                    rewritten_sentences)
                print(
                    f"CoT Feedback check for {requested_feature_key} requested: {self.given_sentences_feature_dict['WordCount']['tgt_ideal']}"
                    f"\t Obtained: {self.given_sentences_feature_dict['WordCount']['src']}")

        if check_any_feature_mismatch:
            print(f"Executing Chain-of-Thought prompting - iteration: {iteration}")
            messages.append({"role": "assistant", "content": assistant_response_content})
            user_chain_prompt = "" if "cot_reasoning_user_prompt" not in config_yaml else read_csv(csv_file=config_yaml["prompt_csv"],
                                     level_value=config_yaml["cot_reasoning_user_prompt"],
                                     column_name="cot_reason")
            user_chain_message = {"role": "user", "content": user_chain_prompt.replace(
                "{output_text}", rewritten_sentences).replace("{print_dependency_tree_with_depth}", str(dependency_tree_print_with_depth(config_yaml["lang"],rewritten_sentences))).replace(
                "{print_dependency_tree_with_length}", str(dependency_tree_print_with_length(config_yaml["lang"], rewritten_sentences))).replace(
                "{print_difficult_words_list}", str(print_difficult_words(rewritten_sentences))).replace(
                "{print_word_list}", str(print_word_count(rewritten_sentences))).replace(
                "{no_of_sentences}", str(len(split_sentences(rewritten_sentences)))).replace(
                "{print_char_list}", str(print_char_list(rewritten_sentences))).strip()}

            user_chain_message = self.fill_prompt_with_feature_values(user_chain_message,
                                                                      self.given_sentences_feature_dict)
            messages.append(user_chain_message)

            request_json_for_a_line, custom_id_for_each_request = self.prepare_jsonl_batch_object(config_yaml, messages,
                                                                                                  f"{test_src_line}-loop_iteration-{iteration}")
            print("JSONL object:\n" + str(request_json_for_a_line))
            with open(batch_request_jsonl_file, 'w') as fp_fo:
                fp_fo.writelines(request_json_for_a_line + "\n")

            batch_request_details_config = f'{config_yaml["path_to_output_generation"]}/line-{test_src_line}-grade_{self.grade}-batch_request_details.json'
            send_batch_request(batch_request_jsonl_file, config_yaml, batch_request_details_config)

            output_file_id = None
            while not output_file_id:
                print("Sleeping for 60s...")
                time.sleep(60)  # Sleep for 120 seconds
                output_file_id, assistant_response_content, rewritten_sentences = obtain_batch_response_for_one_line(
                    batch_request_details_config, config_yaml,
                    custom_id_for_each_request)
            print(f"Executing Chain-of-Thought prompting iteration-{iteration}   Assistant_Response:- {assistant_response_content}\n"
                  f" Extracted_rewritten-sentece(s):- {rewritten_sentences}")
            # save first round results for CoT FS case (w/o feedback).
            with open(output_generation_path + "/Feedback_loop_iternations.csv", 'a') as fp_fo:
                fp_fo.writelines(
                    f"line_number, {test_src_line}, iternation, {iteration}, response, {assistant_response_content}")

        return assistant_response_content, rewritten_sentences

    def chat_completion_request(self, config_yaml, messages):
        max_attempts = 10  # Maximum number of attempts to resolve BadRequestError
        attempt = 0
        response = None

        while attempt < max_attempts:
            try:
                response = self.send_chat_completion_model_request(config_yaml, messages)
                break  # Exit the loop if the request is successful
            except openai.RateLimitError as e:
                print(f"[ERROR] Rate limit exceeded. Please try again later... error_message: {e}")
                time.sleep(60)  # Sleep for 60 seconds
            except openai.APITimeoutError as e:
                print(f"[ERROR] Timeout occurred! Please try again later... error_message: {e}")
                time.sleep(120)  # Sleep for 120 seconds
            except openai.BadRequestError as e:
                print(f"[ERROR] BadRequestError: {e}")
                # Adjusting the messages to handle token limit
                attempt += 1
                if attempt < max_attempts:
                    delete_from = 1 + (self.number_of_examples * 2) + 3  # Example: 14
                    remaining_messages = messages[delete_from + 4:]

                    if len(remaining_messages) < 2:
                        # Keep element at index 0 and delete elements at indices 1 and 2
                        updated_messages = [messages[0]] + messages[3:delete_from] + remaining_messages
                        self.number_of_examples = self.number_of_examples - 1
                        print(f"[INFO] Setting self.number_of_examples == {self.number_of_examples}")
                    else:
                        updated_messages = messages[:delete_from] + remaining_messages

                    print(f"[INFO] Adjusting messages and retrying... Attempt {attempt}")
                    messages = updated_messages
                else:
                    print(f"[ERROR] Failed after {max_attempts} attempts to resolve BadRequestError: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")
                break

        return response, messages

    def send_chat_completion_model_request(self, config_yaml, messages):
        # Record the start time
        start_time = time.time()
        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"API request sent at: {start_timestamp}")

        print("OpenAI Chat Completion request parameters: model:%s, model_name:%s, temperature:%s, max_tokens=%s, seed=%s"
              % (config_yaml["model"], "default" if "model_name" not in config_yaml else config_yaml["model_name"],
                 config_yaml["temperature"], config_yaml["max_tokens"], config_yaml["seed"]))
        print("Request prompt: %s" % (messages))
        # response = openai.ChatCompletion.create(
        #     model=config_yaml["model"],
        #     messages=messages,
        #     temperature=float(config_yaml["temperature"]),
        #     max_tokens=int(config_yaml["max_tokens"])
        # )
        if self.client == None:
            response = openai.chat.completions.create(
                model=config_yaml["model"],
                messages=messages,
                temperature=float(config_yaml["temperature"]),
                max_tokens=int(config_yaml["max_tokens"]),
                seed=123,
            )
        else:
            response = self.client.chat.completions.create(
                model="default" if "model_name" not in config_yaml else config_yaml["model_name"],
                messages=messages,
                temperature=float(config_yaml["temperature"]),
                max_tokens=int(config_yaml["max_tokens"]),
                seed=int(config_yaml["seed"]),
                # logprobs=True,
                # n=4,
            )

        # Record the end time
        end_time = time.time()
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = end_time - start_time

        # Convert elapsed time to hours, minutes, and seconds
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print(f"API response at: {end_timestamp}")
        print(f"API response duration: {hours} hours, {minutes} minutes, {seconds} seconds")

        return response

    def prepare_jsonl_batch_object(self, config_yaml, messages, line_number):

        print("OpenAI Chat Completion BATCH request parameters: model:%s, temperature:%s, max_tokens=%s"
              % (config_yaml["model"], config_yaml["temperature"], config_yaml["max_tokens"]))
        print("Request prompt: %s" % (messages))

        # Construct the JSON object for each line
        json_line = json.dumps({
            "custom_id": f"request-{line_number}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": config_yaml["model"],
                "messages": messages,
                "max_tokens": config_yaml["max_tokens"],
                "temperature": float(config_yaml["temperature"]),
                "seed": 123
            }
        })
        return json_line, f"request-{line_number}"

    def pick_examples_for_single_feature(self, config, src_sent_control_token_value, target_sent_ideal_control_token_value):

        examples = []
        if self.number_of_examples <= 0:
            return examples
        else:
            self.example_val_dataset_src = config["example_val_dataset_src"]
            self.example_val_dataset_tgt = config["example_val_dataset_tgt"]
            self.example_val_dataset_feature_values = config["example_val_dataset_feature_values"]
            count = 0

            with open(self.example_val_dataset_feature_values, 'r') as fp:
                for i, line in enumerate(fp):
                    # max_depth_source, 7, max_depth_target, 7, depth_ratio, 1.0
                    # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
                    line_splits = line.split(",")
                    ex_src_line_max_dep = int(line_splits[1])
                    ex_tgt_line_max_dep = int(line_splits[3])
                    # print("!!!!! ex_src_line_max_dep:%s \t ex_tgt_line_max_dep:%s "% (str(ex_src_line_max_dep), str(ex_tgt_line_max_dep)))
                    # Best match exactly src and tgt max dep values are match. else same tgt max dep and src MaxDepDepth better to be higher than tgt.
                    if (src_sent_control_token_value == ex_src_line_max_dep and target_sent_ideal_control_token_value == ex_tgt_line_max_dep) or \
                            (target_sent_ideal_control_token_value == ex_tgt_line_max_dep and ex_src_line_max_dep > target_sent_ideal_control_token_value and (target_sent_ideal_control_token_value/src_sent_control_token_value) < 1 ) or \
                            (target_sent_ideal_control_token_value == ex_tgt_line_max_dep and ex_src_line_max_dep < target_sent_ideal_control_token_value and (target_sent_ideal_control_token_value/src_sent_control_token_value) > 1 ) or \
                            (target_sent_ideal_control_token_value == ex_tgt_line_max_dep and ex_src_line_max_dep == target_sent_ideal_control_token_value and (target_sent_ideal_control_token_value/src_sent_control_token_value) == 1):
                        count += 1
                        if count > self.number_of_examples:
                            break
                        with open(self.example_val_dataset_src, 'r') as ex_src_file, open(self.example_val_dataset_tgt, 'r') as ex_tgt_file:
                            ex_src = ex_src_file.readlines()[i]
                            ex_tgt = ex_tgt_file.readlines()[i]
                            example = {"Input": ex_src.strip(), "Output": ex_tgt.strip()}
                            print("Ideal_target/Test_src_input: %s/%s \t Obtained_example_target/Obtained_example_src: "
                                  "%s/%s \t Example_src_tgt: %s \t %s" % (target_sent_ideal_control_token_value, src_sent_control_token_value,
                                             ex_tgt_line_max_dep, ex_src_line_max_dep, ex_src.strip(), ex_tgt.strip()))
                            examples.append(example)
                return examples


    def pick_examples_for_single_feature_from_ratio_csv(self, config, src_sent_control_token_value,
                                         target_sent_ideal_control_token_value, feature):

        src_feature="abs_src_"+ feature
        tgt_feature="abs_tgt_"+ feature
        examples = []
        if self.number_of_examples <= 0:
            return examples

        self.example_val_dataset_src = config["example_val_dataset_src"]
        self.example_val_dataset_tgt = config["example_val_dataset_tgt"]
        example_val_dataset_features = config["example_val_dataset_feature_values"]

        # Read the CSV file into a DataFrame
        # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
        df = pd.read_csv(example_val_dataset_features)

        # Add the new column with line numbers
        # We add 1 because DataFrame indices start from 0, but you typically want line numbers starting from 1
        df['new_line_no'] = df.index + 1

        # print("print line number 4: ")
        # print(df.iloc[3])

        # Apply filters
        filtered_df = df[(df[src_feature] == src_sent_control_token_value) & (
                    df[tgt_feature] == target_sent_ideal_control_token_value)]

        if filtered_df.empty:
            if (target_sent_ideal_control_token_value / src_sent_control_token_value) < 1:
                filtered_df = df[(df[src_feature] > target_sent_ideal_control_token_value) & (
                        df[tgt_feature] == target_sent_ideal_control_token_value)]
            elif (target_sent_ideal_control_token_value / src_sent_control_token_value) > 1:
                filtered_df = df[(df[src_feature] < target_sent_ideal_control_token_value) & (
                        df[tgt_feature] == target_sent_ideal_control_token_value)]
            elif (target_sent_ideal_control_token_value / src_sent_control_token_value) == 1:
                # this maybe redundant:
                filtered_df = df[(df[src_feature] == target_sent_ideal_control_token_value) & (
                        df[tgt_feature] == target_sent_ideal_control_token_value)]

        # Iterate over the filtered DataFrame
        for i, row in filtered_df.iterrows():
            if len(examples) >= self.number_of_examples:
                break

            # Read the corresponding lines from the source and target files
            with open(self.example_val_dataset_src, 'r') as ex_src_file, open(self.example_val_dataset_tgt,
                                                                              'r') as ex_tgt_file:
                lines_src = ex_src_file.readlines()
                lines_tgt = ex_tgt_file.readlines()

                line_no = int(row['new_line_no']) - 1  # Subtract 1 because list indices start from 0
                ex_src = lines_src[line_no].strip()
                ex_tgt = lines_tgt[line_no].strip()

                example = {"Input": ex_src, "Output": ex_tgt}
                print(
                    f"Ideal_target/Test_src_input: {target_sent_ideal_control_token_value}/{src_sent_control_token_value} \t"
                    f" Obtained_example_target/Obtained_example_src: {row[tgt_feature]}/{row[src_feature]} "
                    f"\t Example_src_tgt: {ex_src} \t {ex_tgt}")
                examples.append(example)

        return examples

    def pick_examples_for_multiple_features_from_ratio_csv(self, config_yaml):

        examples=[]
        if self.number_of_examples <= 0:
            return examples

        print("Going to pick examples for given requested feature values!")
        self.example_val_dataset_src = config_yaml["example_val_dataset_src"]
        self.example_val_dataset_tgt = config_yaml["example_val_dataset_tgt"]
        example_val_dataset_features = config_yaml["example_val_dataset_feature_values"]
        # Read the CSV file into a DataFrame
        # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
        df = pd.read_csv(example_val_dataset_features)
        # Add the new column with line numbers
        # We add 1 because DataFrame indices start from 0, but you typically want line numbers starting from 1
        df['new_line_no'] = df.index + 1

        # first assign all df to filtered_df
        filtered_df = df
        print(f"filtered_df.columns: {filtered_df.columns}")

        # print(filtered_df.columns)
        # Iterating over the top level keys 'MaxDepDepth' and 'MaxDepLength'
        for feature_key in self.given_sentences_feature_dict:
            print(f"Feature key from given_sentences_feature_dict: {feature_key}")
            src_feature="abs_src_"+ feature_key
            tgt_feature="abs_tgt_"+ feature_key
            ratio_feature= feature_key + "_ratio"

            # Experiment 1 - Apply filters exact match feature.
            temp_filtered_df = filtered_df[(filtered_df[src_feature] == self.given_sentences_feature_dict[feature_key]["src"]) & (
                        filtered_df[tgt_feature] == self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]
            print(f"EXACT MATCH for feature: {feature_key}\tsrc: {self.given_sentences_feature_dict[feature_key]['src']},\t"
                f"tgt: {self.given_sentences_feature_dict[feature_key]['tgt_ideal']},\tfound possible examples count: {len(temp_filtered_df)}")
            #
            # # # Experiment 2 - pick same ratio sentences as examples.
            # # temp_df_wo_exact_match = filtered_df[(filtered_df[src_feature] != self.given_sentences_feature_dict[feature_key]["src"]) & (
            # #             filtered_df[tgt_feature] != self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]
            # # temp_filtered_df = temp_df_wo_exact_match[(round(temp_df_wo_exact_match[ratio_feature], 1) == round(
            # #                 self.given_sentences_feature_dict[feature_key]["tgt_ideal"] / self.given_sentences_feature_dict[feature_key]["src"], 1))]
            # # Experiment 2a - for 4-feature combination pick same ratio sentences as examples. here we don't exclude exact match.
            # temp_filtered_df = filtered_df[(round(filtered_df[ratio_feature], 1) == round(
            #     self.given_sentences_feature_dict[feature_key]["tgt_ideal"] /
            #     self.given_sentences_feature_dict[feature_key]["src"], 1))]
            # print(f"RATIO MATCH for feature: {feature_key}\tratio: {ratio_feature},\tfound possible ratio match examples count: {len(temp_filtered_df)}")

            # # # Experiment 3 - pick random examples sentences as examples.
            # # temp_df_wo_exact_match = filtered_df[
            # #     (filtered_df[src_feature] != self.given_sentences_feature_dict[feature_key]["src"]) & (
            # #             filtered_df[tgt_feature] != self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]
            # # temp_filtered_df = temp_df_wo_exact_match.sample(n=self.number_of_examples, random_state=1)
            # # print(f"Random examples selection count is {len(temp_filtered_df)}")

            # Now append ratio match rows.
            temp_df_wo_exact_match = filtered_df[(filtered_df[src_feature] != self.given_sentences_feature_dict[feature_key]["src"]) & (
                                filtered_df[tgt_feature] != self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]
            temp_filtered_df_ratio_match = temp_df_wo_exact_match[(round(temp_df_wo_exact_match[ratio_feature],1) == round(
                    self.given_sentences_feature_dict[feature_key]["tgt_ideal"] / self.given_sentences_feature_dict[feature_key]["src"], 1))]
            print(f"RATIO MATCH for feature: {feature_key}\tratio: {ratio_feature},\tfound possible ratio match examples count: {len(temp_filtered_df_ratio_match)}")

            temp_filtered_df = pd.concat([temp_filtered_df, temp_filtered_df_ratio_match], ignore_index=True)
            print(f"Combined EXACT MATCH and RATIO MATCH for feature: {feature_key}\tratio: {ratio_feature},\tfound possible ratio match examples count: {len(temp_filtered_df)}")

            # # if temp_filtered_df.empty or len(temp_filtered_df) < self.number_of_examples:
            # #     # tgt feature same value, src in range:
            # #     if (self.given_sentences_feature_dict[feature_key]["tgt_ideal"] / self.given_sentences_feature_dict[feature_key]["src"]) < 1:
            # #
            # #         temp_filtered_df_range = filtered_df[(filtered_df[src_feature] > self.given_sentences_feature_dict[feature_key]["tgt_ideal"]) & (
            # #                 filtered_df[tgt_feature] == self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]
            # #
            # #     elif (self.given_sentences_feature_dict[feature_key]["tgt_ideal"] / self.given_sentences_feature_dict[feature_key]["src"]) > 1:
            # #
            # #         temp_filtered_df_range = filtered_df[(filtered_df[src_feature] < self.given_sentences_feature_dict[feature_key]["tgt_ideal"]) & (
            # #                 filtered_df[tgt_feature] == self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]
            # #
            # #     elif (self.given_sentences_feature_dict[feature_key]["tgt_ideal"] / self.given_sentences_feature_dict[feature_key]["src"]) == 1:
            # #         # this maybe redundant:
            # #         temp_filtered_df_range = filtered_df[(filtered_df[src_feature] == self.given_sentences_feature_dict[feature_key]["src"]) & (
            # #                 filtered_df[tgt_feature] == self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]
            # #
            # #     temp_filtered_df = pd.concat([temp_filtered_df_range, temp_filtered_df], ignore_index=True)
            #
            filtered_df = temp_filtered_df
            print(f"For feature {feature_key}: found possible examples total: {len(filtered_df)}")

        # #  4-feature experiment case-3 for random example selection:
        # temp_filtered_df = filtered_df.sample(n=self.number_of_examples, random_state=1)
        # filtered_df =  temp_filtered_df
        # print(f"RANDOM PICK examples count: {len(filtered_df)}")

        examples = self.prepare_examples_and_feature_value_dict(filtered_df)
        examples.reverse() # bcz we first add exact match, then fill the gap with ratio match. in prompt, we want to give exact examples at last to understand better.
        return examples

    def pick_grade_level_examples(self, config_yaml):

        pd.set_option('display.max_rows', None)  # Set to None to display all rows
        pd.set_option('display.max_columns', None)  # Set to None to display all columns
        pd.set_option('display.width', None)  # Set to None to automatically detect the display width
        pd.set_option('display.max_colwidth', None)  # Set to None to show full content of each column

        examples=[]
        if self.number_of_examples <= 0:
            return examples

        print("Going to pick Grade-level examples for given sentence!")
        self.example_val_dataset_src = config_yaml["example_val_dataset_src"]
        self.example_val_dataset_tgt = config_yaml["example_val_dataset_tgt"]
        example_val_dataset_features = config_yaml["example_val_dataset_feature_values"]
        # Read the CSV file into a DataFrame
        # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
        df = pd.read_csv(example_val_dataset_features)
        # Add the new column with line numbers
        # We add 1 because DataFrame indices start from 0, but you typically want line numbers starting from 1
        df['new_line_no'] = df.index + 1

        # first assign all df to filtered_df
        filtered_df = df
        print(f"filtered_df.columns: {filtered_df.columns}")

        src_feature="abs_src_FKGL_Grade"
        # Experiment 1 - Apply filters exact match feature.
        grade_temp_filtered_df = filtered_df[(filtered_df[src_feature] == self.given_sentences_feature_dict["Grade"]["src"])]
        #  & ( filtered_df[tgt_feature] == self.given_sentences_feature_dict["Grade"]["tgt_ideal"])]
        print(f"EXACT Src MATCH for feature: Grade\tsrc: {self.given_sentences_feature_dict['Grade']['src']},\t"
            f"tgt: {self.given_sentences_feature_dict['Grade']['tgt_ideal']},\tfound possible examples count: {len(grade_temp_filtered_df)}")

        filtered_df = grade_temp_filtered_df
        print(f"For feature Grade: found possible examples total: {len(filtered_df)}")

        # Iterating over the top level keys 'MaxDepDepth' and 'MaxDepLength'
        for feature_key in self.given_sentences_feature_dict:
            print(f"Feature key from given_sentences_feature_dict: {feature_key}")
            ratio_feature = feature_key + "_ratio"

            if feature_key == "Grade":
                ratio_feature = "FKGL_Grade_ratio"
                print(filtered_df.head())

            temp_abs_src= self.given_sentences_feature_dict[feature_key]["src"] if self.given_sentences_feature_dict[feature_key]["src"] !=0 else 0.5
            temp_filtered_df_ratio_match = filtered_df[(round(filtered_df[ratio_feature], 1) == round(
                    self.given_sentences_feature_dict[feature_key]["tgt_ideal"] / temp_abs_src
                    , 1))]
            print(f"RATIO MATCH for feature: {feature_key}\tratio: {ratio_feature},\tfound possible ratio match examples count: {len(temp_filtered_df_ratio_match)}")

            filtered_df = temp_filtered_df_ratio_match
            print(f"For feature {feature_key}: found possible examples total: {len(filtered_df)}")

        if filtered_df.empty or len(filtered_df) < self.number_of_examples:
            filtered_df=grade_temp_filtered_df
        examples = self.prepare_examples_and_feature_value_dict(filtered_df)
        examples.reverse()  # we don't need to reverse in grade level example selection.
        return examples


    def exact_src_tgt_examples_selection(self, df_all):
        filtered_df = df_all
        for feature_key in self.given_sentences_feature_dict:
            # print(f"EXACT MATCH: Feature key from given_sentences_feature_dict: {feature_key}")
            src_feature = "abs_src_" + feature_key
            tgt_feature = "abs_tgt_" + feature_key

            # Experiment 1 - Apply filters exact match feature.
            temp_filtered_df = filtered_df[
                (filtered_df[src_feature] == self.given_sentences_feature_dict[feature_key]["src"]) & (
                        filtered_df[tgt_feature] == self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]
            filtered_df = temp_filtered_df
            print(f"EXACT MATCH for feature: {feature_key}\tsrc: {self.given_sentences_feature_dict[feature_key]['src']},\t"
                f"tgt: {self.given_sentences_feature_dict[feature_key]['tgt_ideal']},\tfound possible examples count: {len(filtered_df)}")
        return filtered_df

    def pick_examples_for_multiple_features_from_ratio_csv_for_fuzzy_tgt_range(self, config_yaml):

        examples=[]
        if self.number_of_examples <= 0:
            return examples

        print("Going to pick examples for given requested feature values!")
        self.example_val_dataset_src = config_yaml["example_val_dataset_src"]
        self.example_val_dataset_tgt = config_yaml["example_val_dataset_tgt"]
        example_val_dataset_features = config_yaml["example_val_dataset_feature_values"]
        # Read the CSV file into a DataFrame
        # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
        df = pd.read_csv(example_val_dataset_features)
        # Add the new column with line numbers
        # We add 1 because DataFrame indices start from 0, but you typically want line numbers starting from 1
        df['new_line_no'] = df.index + 1
        print(f"filtered_df.columns: {df.columns}")

        # First get exact match examples:
        exact_src_tgt_match_filtered_df =  self.exact_src_tgt_examples_selection(df)

        fuzzy_match_filtered_df = pd.DataFrame()
        fuzzy_match_filtered_df_round_2 = pd.DataFrame()

        if exact_src_tgt_match_filtered_df.empty or len(exact_src_tgt_match_filtered_df) < self.number_of_examples:
            #     move to pick examples within fuzzy range.
            fuzzy_match_filtered_df = self.fuzzy_match_example_selection(df)
            if fuzzy_match_filtered_df.empty or len(fuzzy_match_filtered_df) < self.number_of_examples:
                fuzzy_match_filtered_df_round_2 = self.fuzzy_match_example_selection(df, additional_diff=1)
                indices_to_remove = fuzzy_match_filtered_df.index
                # Remove these entries from fuzzy_match_filtered_df_round_2
                fuzzy_match_filtered_df_round_2 = fuzzy_match_filtered_df_round_2.drop(indices_to_remove, errors='ignore')

        examples = self.prepare_examples_and_feature_value_dict(exact_src_tgt_match_filtered_df)
        fuzzy_examples_list= self.prepare_examples_and_feature_value_dict(fuzzy_match_filtered_df)
        fuzzy_examples_list_2= self.prepare_examples_and_feature_value_dict(fuzzy_match_filtered_df_round_2)

        # Calculate how many more examples are needed
        examples_needed = self.number_of_examples - len(examples)

        # If more examples are needed, prepend them from fuzzy_examples_dict
        if examples_needed > 0:
            # Take the last 'examples_needed' items from fuzzy_examples_dict since we're prepending
            fuzzy_to_add = fuzzy_examples_list[-examples_needed:]
            # Prepend to the examples list
            examples = fuzzy_to_add + examples

        # re-Calculate how many more examples are needed
        examples_needed = self.number_of_examples - len(examples)
        if examples_needed > 0:
            fuzzy_to_add_list_2 = fuzzy_examples_list_2[-examples_needed:]
            examples = fuzzy_to_add_list_2 + examples

        return examples

    def fuzzy_match_example_selection(self, df_all, additional_diff=0):
        filtered_df = df_all
        for feature_key in self.given_sentences_feature_dict:
            # print(f"FUZZY MATCH: Feature key from given_sentences_feature_dict: {feature_key}")
            src_feature = "abs_src_" + feature_key
            tgt_feature = "abs_tgt_" + feature_key

            # Experiment 1 - Apply filters exact match feature.
            tgt_min = self.given_sentences_feature_dict[feature_key]["tgt_ideal"] - \
                      self.given_sentences_feature_dict[feature_key]["tgt_ideal_range"]
            tgt_max = self.given_sentences_feature_dict[feature_key]["tgt_ideal"] + \
                      self.given_sentences_feature_dict[feature_key]["tgt_ideal_range"]
            # print(f"FUZZY MATCH: target range: tgt_min: {tgt_min} and tgt_max: {tgt_max}")

            src_min = self.given_sentences_feature_dict[feature_key]["src"] - \
                      self.given_sentences_feature_dict[feature_key]["tgt_ideal_range"] - additional_diff
            src_max = self.given_sentences_feature_dict[feature_key]["src"] + \
                      self.given_sentences_feature_dict[feature_key]["tgt_ideal_range"] + additional_diff

            # src_min = self.given_sentences_feature_dict[feature_key]["src"] - (3* self.given_sentences_feature_dict[feature_key]["tgt_ideal_range"])
            # src_max = self.given_sentences_feature_dict[feature_key]["src"] + (3* self.given_sentences_feature_dict[feature_key]["tgt_ideal_range"])
            # print(f"FUZZY MATCH: source range: src_min: {src_min} and src_max: {src_max}")

            temp_df_wo_exact_match = filtered_df[
                (filtered_df[src_feature] != self.given_sentences_feature_dict[feature_key]["src"]) & (
                        filtered_df[tgt_feature] != self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]
            # print(f"FUZZY MATCH: temp_df_wo_exact_match examples without exact match: count is {len(temp_df_wo_exact_match)}")

            temp_filtered_df = temp_df_wo_exact_match[
                ((temp_df_wo_exact_match[src_feature] >= src_min) & (temp_df_wo_exact_match[src_feature] <= src_max)) &
                ((temp_df_wo_exact_match[tgt_feature] >= tgt_min) & (temp_df_wo_exact_match[tgt_feature] <= tgt_max))
                ]

            # print(f"FUZZY MATCH: pick examples within range: {len(temp_filtered_df)}")

            filtered_df = temp_filtered_df
            print(f"FUZZY MATCH for feature: {feature_key}\tsrc_range: {src_min}-{src_max},\ttgt_range: {tgt_min}-{tgt_max},\t"
                  f"found possible fuzzy range of examples total: {len(filtered_df)}")
        return filtered_df

    def prepare_examples_and_feature_value_dict(self, filtered_df):
        examples = []
        # Iterate over the filtered DataFrame
        for i, row in filtered_df.iterrows():
            if len(examples) >= self.number_of_examples:
                break

            # Read the corresponding lines from the source and target files
            with open(self.example_val_dataset_src, 'r') as ex_src_file, open(self.example_val_dataset_tgt,
                                                                              'r') as ex_tgt_file:
                lines_src = ex_src_file.readlines()
                lines_tgt = ex_tgt_file.readlines()

                line_no = int(row['new_line_no']) - 1  # Subtract 1 because list indices start from 0
                ex_src = lines_src[line_no].strip()
                ex_tgt = lines_tgt[line_no].strip()

                example_src = {"src": ex_src, "abs_src_MaxDepDepth": round(int(row["abs_src_MaxDepDepth"])),
                               "abs_src_MaxDepLength": round(int(row["abs_src_MaxDepLength"])),
                               "abs_src_DiffWords": round(int(row["abs_src_DiffWords"])),
                               "abs_src_WordCount": round(int(row["abs_src_WordCount"])),
                               "abs_src_Grade": round(int(row["abs_src_FKGL_Grade"])),
                               "abs_src_Length": round(int(row["abs_src_Length"]))}

                example_tgt = {"tgt": ex_tgt, "abs_tgt_MaxDepDepth": round(int(row["abs_tgt_MaxDepDepth"])),
                               "abs_tgt_MaxDepLength": round(int(row["abs_tgt_MaxDepLength"])),
                               "abs_tgt_DiffWords": round(int(row["abs_tgt_DiffWords"])),
                               "abs_tgt_WordCount": round(int(row["abs_tgt_WordCount"])),
                               "abs_tgt_Grade": round(int(row["abs_tgt_FKGL_Grade"])),
                               "abs_tgt_Length": round(int(row["abs_tgt_Length"]))}
                example = {"Input": example_src, "Output": example_tgt}

                print(f"example_src: {example_src}\n example_tgt:{example_tgt}")
                for key in self.given_sentences_feature_dict:
                    if key == "Grade":
                        print(
                            f"For {key}: Ideal_target_FKGL/Test_src_input_FKGL: {self.given_sentences_feature_dict[key]['tgt_ideal']}/{self.given_sentences_feature_dict[key]['src']} \t"
                            f"Obtained_ex_tgt_FKGL/Obtained_ex_src_FKGL of {key}: {row['abs_tgt_FKGL_' + key]}/{row['abs_src_FKGL_' + key]} \t")
                    else:
                        print(
                        f"For {key}: Ideal_target/Test_src_input: {self.given_sentences_feature_dict[key]['tgt_ideal']}/{self.given_sentences_feature_dict[key]['src']} \t"
                        f"Obtained_ex_tgt/Obtained_ex_src of {key}: {row['abs_tgt_' + key]}/{row['abs_src_' + key]} \t")

                # print(f"\tObtained Example_src_tgt: {ex_src} \t {ex_tgt}")
                examples.append(example)
        return examples

    def prepare_messages_for_chat_model(self, config_yaml, examples, src, control_token, target_sent_ideal_value):
        messages = []
        system_message = {"role": "system",
                          "content": self.system_prompt.replace(control_token, str(target_sent_ideal_value))}
        test_input = {"role": "user", "content": self.user_prompt.replace(
            "{max_dep_depth_src}", str(get_max_dep_depth_of_given_sent(src))).replace(
            "{input_src}", src.strip()).replace(
            control_token, str(target_sent_ideal_value)).replace(
            "{print_dependency_tree_with_depth}", str(dependency_tree_print_with_depth(config_yaml["lang"],src))).replace(
            "{max_dep_length_src}", str(get_max_dep_length_of_given_sent(src))).replace(
            "{print_dependency_tree_with_length}", str(dependency_tree_print_with_length(config_yaml["lang"], src))).strip()
        }
        messages.append(system_message)
        for example in examples:
            example_user = {"role": "user", "content": self.user_prompt.replace(
                "{max_dep_depth_src}", str(get_max_dep_depth_of_given_sent(src))).replace(
                "{input_src}", example.get("Input").strip()).replace(
                control_token, str(target_sent_ideal_value)).replace(
                "{print_dependency_tree_with_depth}", str(dependency_tree_print_with_depth(config_yaml["lang"],src))).replace(
                "{max_dep_length_src}", str(get_max_dep_length_of_given_sent(src))).replace(
                "{print_dependency_tree_with_length}", str(dependency_tree_print_with_length(config_yaml["lang"], src))).strip()
                            }
            example_assistant = {"role": "assistant", "content": example.get("Output").strip()}
            messages.append(example_user)
            messages.append(example_assistant)
        messages.append(test_input)
        return messages

    def prepare_messages_for_chat_model_with_multiple_features(self, config_yaml, examples, src):
        messages = []
        if self.system_prompt != "":
            system_message = {"role": "system","content": self.system_prompt}
            system_message = self.fill_prompt_with_feature_values(system_message, self.given_sentences_feature_dict)
            messages.append(system_message)

        test_input = {"role": "user", "content": self.user_prompt.replace(
            "{input_src}", src.strip()).replace(
            "{print_dependency_tree_with_depth}", str(dependency_tree_print_with_depth(config_yaml["lang"],src))).replace(
            "{print_dependency_tree_with_length}", str(dependency_tree_print_with_length(config_yaml["lang"], src))).replace(
            "{print_difficult_words_list}", str(print_difficult_words(src))).replace(
            "{print_word_list}", str(print_word_count(src))).replace("{no_of_sentences}", str(len(split_sentences(src)))).replace(
            "{print_char_list}", str(print_char_list(src))).strip()}
        test_input = self.fill_prompt_with_feature_values(test_input, self.given_sentences_feature_dict)

        # if not self.examples_messages:
        for example in examples:
            input_src_example= example["Input"]["src"].strip()
            example_user = {"role": "user", "content": self.user_prompt.replace(
                "{input_src}", input_src_example).replace(
                "{print_dependency_tree_with_depth}", str(dependency_tree_print_with_depth(config_yaml["lang"],input_src_example))).replace(
                "{print_dependency_tree_with_length}", str(dependency_tree_print_with_length(config_yaml["lang"], input_src_example))).replace(
                "{print_difficult_words_list}", str(print_difficult_words(input_src_example))).replace(
                "{print_word_list}", str(print_word_count(input_src_example))).replace("{no_of_sentences}", str(len(split_sentences(input_src_example)))).replace(
            "{print_char_list}", str(print_char_list(src))).strip()
                            }

            # example_assistant = {"role": "assistant", "content": example.get("Output").strip()}
            output_tgt_example = example["Output"]["tgt"].strip()
            # output_prompt =  "" if "output_prompt" not in config_yaml else read_content(config_yaml["output_prompt"])
            example_assistant = {"role": "assistant", "content":  self.output_prompt.replace(
            "{output_text}", output_tgt_example).replace(
            "{print_dependency_tree_with_depth}", str(dependency_tree_print_with_depth(config_yaml["lang"],output_tgt_example))).replace(
            "{print_dependency_tree_with_length}", str(dependency_tree_print_with_length(config_yaml["lang"], output_tgt_example))).replace(
            "{print_difficult_words_list}", str(print_difficult_words(output_tgt_example))).replace(
            "{print_word_list}", str(print_word_count(output_tgt_example))).replace("{no_of_sentences}", str(len(split_sentences(output_tgt_example)))).replace(
            "{print_char_list}", str(print_char_list(src))).strip()}

            example_pair_feature_dict=self.cal_example_text_feature_values(config_yaml, example["Input"], example["Output"])
            example_user = self.fill_prompt_with_feature_values(example_user, example_pair_feature_dict)
            example_assistant = self.fill_prompt_with_feature_values(example_assistant, example_pair_feature_dict)

            messages.append(example_user)
            messages.append(example_assistant)

                # self.examples_messages.append(example_user)
                # self.examples_messages.append(example_assistant)

        # messages.extend(self.examples_messages)
        messages.append(test_input)
        return messages

    def fill_prompt_with_feature_values(self, message, feature_dict):
        for feature_key in feature_dict:
                # range is always set according to test src and requested conditions. so FS examples also carry the same.
                tgt_min = self.given_sentences_feature_dict[feature_key]["tgt_ideal"] - self.given_sentences_feature_dict[feature_key]["tgt_ideal_range"]
                tgt_max = self.given_sentences_feature_dict[feature_key]["tgt_ideal"] + self.given_sentences_feature_dict[feature_key]["tgt_ideal_range"]
                tgt_range = f"{tgt_min} and {tgt_max}"
                message["content"] = message["content"].replace(
                    "{src_" + feature_key + "}", str(feature_dict[feature_key]["src"])).replace(
                    "{tgt_ideal_" + feature_key + "}", str(feature_dict[feature_key]["tgt_ideal"])).replace(
                    "{tgt_ideal_" + feature_key + "_range}", str(tgt_range)).strip()

        return message

    def cal_example_text_feature_values(self, config_yaml, example_src, example_tgt):

        feature_dict = {}
        for feature in config_yaml["feature_names"].split(','):
            pattern = rf"\{{[^{{}}]*_{re.escape(feature)}\}}"

            if feature == "MaxDepDepth" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
                feature_dict.setdefault("MaxDepDepth", {})
                # feature_dict["MaxDepDepth"] ["src"] = get_max_dep_depth_of_given_sent(example_src)
                # feature_dict["MaxDepDepth"] ["tgt_ideal"] = get_max_dep_depth_of_given_sent(example_tgt)
                feature_dict["MaxDepDepth"] ["src"] = example_src["abs_src_MaxDepDepth"]
                feature_dict["MaxDepDepth"] ["tgt_ideal"] =  example_tgt["abs_tgt_MaxDepDepth"]

            if feature == "MaxDepLength" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
                feature_dict.setdefault("MaxDepLength", {})
                # feature_dict["MaxDepLength"] ["src"] = get_max_dep_length_of_given_sent(example_src)
                # feature_dict["MaxDepLength"] ["tgt_ideal"] = get_max_dep_length_of_given_sent(example_tgt)
                feature_dict["MaxDepLength"]["src"] = example_src["abs_src_MaxDepLength"]
                feature_dict["MaxDepLength"]["tgt_ideal"] = example_tgt["abs_tgt_MaxDepLength"]

            if feature == "DiffWords" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
                feature_dict.setdefault("DiffWords", {})
                # feature_dict["DiffWords"] ["src"] = get_no_of_difficult_words_of_given_sent(example_src)
                # feature_dict["DiffWords"] ["tgt_ideal"] = get_no_of_difficult_words_of_given_sent(example_tgt)
                feature_dict["DiffWords"]["src"] = example_src["abs_src_DiffWords"]
                feature_dict["DiffWords"]["tgt_ideal"] = example_tgt["abs_tgt_DiffWords"]

            if feature == "WordCount" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
                feature_dict.setdefault("WordCount", {})
                # feature_dict["WordCount"] ["src"]= get_word_count_of_given_sent(example_src)
                # feature_dict["WordCount"] ["tgt_ideal"]= get_word_count_of_given_sent(example_tgt)
                feature_dict["WordCount"]["src"] = example_src["abs_src_WordCount"]
                feature_dict["WordCount"]["tgt_ideal"] = example_tgt["abs_tgt_WordCount"]

            if feature == "Length" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
                feature_dict.setdefault("Length", {})
                # feature_dict["Length"] ["src"]= get_legnth_given_sent(example_src)
                # feature_dict["Length"] ["tgt_ideal"]= get_length_of_given_sent(example_tgt)
                feature_dict["Length"]["src"] = example_src["abs_src_Length"]
                feature_dict["Length"]["tgt_ideal"] = example_tgt["abs_tgt_Length"]

        # if feature == "Grade" and (re.search(pattern, self.system_prompt) or re.search(pattern, self.user_prompt)):
        feature_dict.setdefault("Grade", {})
        # feature_dict["Grade"] ["src"]= get_word_count_of_given_sent(example_src)
        # feature_dict["Grade"] ["tgt_ideal"]= get_word_count_of_given_sent(example_tgt)
        feature_dict["Grade"]["src"] = example_src["abs_src_Grade"]
        feature_dict["Grade"]["tgt_ideal"] = example_tgt["abs_tgt_Grade"]

        # print(feature_dict)
        return feature_dict


    def freq_rank_to_freq_catagory(self, freq_rank):
        if 0 < freq_rank and freq_rank <= 5:
            return "high frequency words"
        elif  5 < freq_rank and freq_rank <= 10:
            return "moderate frequency words"
        elif 10 < freq_rank:
            return "low frequency words or rare Words"
        return ""

    def fine_tune(self):
        pass

    def write_test_prompt_details(self, found_examples_count, requested_features,
                                  src_feature_value, ideal_tgt_feature_value, output_generation_path,
                                  line_number, calibration_round, feature_value_fussy_range):
        path_to_write_word_ratio_values = output_generation_path + "/gpt_request_details.csv"
        with open(path_to_write_word_ratio_values, 'a') as fp:
            final_line_to_write = "line_number, " + str(line_number) + ", " + \
                                  "calibration_round, " + str(calibration_round) + ", " + \
                                  "number_of_examples_found, " + str(found_examples_count) + ", " + \
                                  "requested_dependency_depth, " + str(self.dependency_depth) + ", " + \
                                  "requested_dependency_length, " + str(self.dependency_length) + ", " + \
                                  "requested_difficult_words, " + str(self.difficult_words) + ", " + \
                                  "requested_word_count, " + str(self.word_count) + ", " + \
                                  "requested_frequency, " + str(self.frequency) + ", " + \
                                  "requested_length, " + str(self.length) + ", " + \
                                  "requested_levenshtein, " + str(self.levenshtein) + ", " + \
                                  "requested_features, " + str(requested_features) + ", " + \
                                  "src_feature_value, " + str(src_feature_value) + ", " + \
                                  "ideal_tgt_feature_value, " + str(ideal_tgt_feature_value) + ", " \
                                  "tgt_ideal_fussy_range_diff, " + str(feature_value_fussy_range) + ",\n"
                                  # "prompt, " + str(final_prompt_message) + "\n"
            fp.write(final_line_to_write)


    def pick_match_examples_or_else_radom_examples_for_multiple_features_from_ratio_csv(self, config_yaml):

        examples=[]
        if self.number_of_examples <= 0:
            return examples

        print("Going to pick examples for given requested feature values!")
        self.example_val_dataset_src = config_yaml["example_val_dataset_src"]
        self.example_val_dataset_tgt = config_yaml["example_val_dataset_tgt"]
        example_val_dataset_features = config_yaml["example_val_dataset_feature_values"]
        # Read the CSV file into a DataFrame
        # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
        df = pd.read_csv(example_val_dataset_features)
        # Add the new column with line numbers
        # We add 1 because DataFrame indices start from 0, but you typically want line numbers starting from 1
        df['new_line_no'] = df.index + 1

        # first assign all df to filtered_df
        filtered_df = df

        # print(filtered_df.columns)
        # Iterating over the top level keys 'MaxDepDepth' and 'MaxDepLength'
        for feature_key in self.given_sentences_feature_dict:
            print(f"Feature key from given_sentences_feature_dict: {feature_key}")
            src_feature="abs_src_"+ feature_key
            tgt_feature="abs_tgt_"+ feature_key

            # Apply filters: exact value match examples
            temp_filtered_df = filtered_df[(filtered_df[src_feature] == self.given_sentences_feature_dict[feature_key]["src"]) & (
                        filtered_df[tgt_feature] == self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]

            if temp_filtered_df.empty:
                # target value match examples.
                if (self.given_sentences_feature_dict[feature_key]["tgt_ideal"] / self.given_sentences_feature_dict[feature_key]["src"]) < 1:

                    temp_filtered_df = filtered_df[(filtered_df[src_feature] > self.given_sentences_feature_dict[feature_key]["tgt_ideal"]) & (
                            filtered_df[tgt_feature] == self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]

                elif (self.given_sentences_feature_dict[feature_key]["tgt_ideal"] / self.given_sentences_feature_dict[feature_key]["src"]) > 1:

                    temp_filtered_df = filtered_df[(filtered_df[src_feature] < self.given_sentences_feature_dict[feature_key]["tgt_ideal"]) & (
                            filtered_df[tgt_feature] == self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]

                elif (self.given_sentences_feature_dict[feature_key]["tgt_ideal"] / self.given_sentences_feature_dict[feature_key]["src"]) == 1:
                    # this maybe redundant:
                    temp_filtered_df = filtered_df[(filtered_df[src_feature] == self.given_sentences_feature_dict[feature_key]["src"]) & (
                            filtered_df[tgt_feature] == self.given_sentences_feature_dict[feature_key]["tgt_ideal"])]

            filtered_df = temp_filtered_df
            print(f"For feature {feature_key}: found possible examples total: {len(filtered_df)}")

        if filtered_df.empty or len(filtered_df) < 10:
            # ****************************
            # Step 1: Get the list of 'New Line' values from 'filtered_df'
            newline_values_to_remove = filtered_df['New Line'].unique()

            # Step 2: Create a mask for rows in 'df' whose 'New Line' value is in the list of values to remove
            mask = df['New Line'].isin(newline_values_to_remove)

            # Step 3: Invert the mask and use it to filter 'df', keeping only the rows not present in 'filtered_df'
            diff_df = df[~mask]

            # Select 10 random rows from the difference DataFrame
            if len(diff_df) >= 10:
                selected_rows = diff_df.sample(n=10, random_state=np.random.RandomState())
            else:
                print("Not enough unique rows in df after filtering. Found only {} rows.".format(len(diff_df)))
                selected_rows = diff_df
            # ****************************

            # Assuming filtered_df and selected_rows are already defined DataFrames
            # filtered_df = selected_rows.append(filtered_df, ignore_index=True)
            filtered_df = pd.concat([selected_rows, filtered_df], ignore_index=True)

        # Iterate over the filtered DataFrame
        for i, row in filtered_df.iterrows():
            if len(examples) >= self.number_of_examples:
                break

            # Read the corresponding lines from the source and target files
            with open(self.example_val_dataset_src, 'r') as ex_src_file, open(self.example_val_dataset_tgt,
                                                                              'r') as ex_tgt_file:
                lines_src = ex_src_file.readlines()
                lines_tgt = ex_tgt_file.readlines()

                line_no = int(row['new_line_no']) - 1  # Subtract 1 because list indices start from 0
                ex_src = lines_src[line_no].strip()
                ex_tgt = lines_tgt[line_no].strip()

                example = {"Input": ex_src, "Output": ex_tgt}

                for key in self.given_sentences_feature_dict:
                    print(f"For {key}: Ideal_target/Test_src_input: {self.given_sentences_feature_dict[key]['tgt_ideal']}/{self.given_sentences_feature_dict[key]['src']} \t"
                          f"Obtained_ex_tgt/Obtained_ex_src of {key}: {row['abs_tgt_'+ key]}/{row['abs_src_'+key]} \t")
                print(f"\tObtained Example_src_tgt: {ex_src} \t {ex_tgt}")
                examples.append(example)

        return examples

    def pick_default_common_examples_for_multiple_features_from_ratio_csv(self, config_yaml):

        examples = []
        if self.number_of_examples <= 0:
            return examples

        print("Going to pick random common examples set for given requested feature values!")
        self.example_val_dataset_src = config_yaml["example_val_dataset_src"]
        self.example_val_dataset_tgt = config_yaml["example_val_dataset_tgt"]
        example_val_dataset_features = config_yaml["example_val_dataset_feature_values"]
        # Read the CSV file into a DataFrame
        # Line,abs_src_Length,abs_src_MaxDepDepth,abs_src_MaxDepLength,abs_src_FreqRank,abs_src_Leven,abs_src_WordCount,abs_tgt_Length,abs_tgt_MaxDepDepth,abs_tgt_MaxDepLength,abs_tgt_FreqRank,abs_tgt_Leven,abs_tgt_WordCount,Length_ratio,MaxDepDepth_ratio,MaxDepLength_ratio,FreqRank_ratio,Leven_ratio,WordCount_ratio
        df = pd.read_csv(example_val_dataset_features)
        # Add the new column with line numbers
        # We add 1 because DataFrame indices start from 0, but you typically want line numbers starting from 1
        df['new_line_no'] = df.index + 1

        # Select 30 random rows from the difference DataFrame
        if len(df) >= self.number_of_examples:
            selected_rows = df.sample(n=self.number_of_examples, random_state=np.random.RandomState(26)) #seed=26
        else:
            print("Not enough unique rows in df after filtering. Found only {} rows.".format(len(df)))
            selected_rows = df
        # ****************************


        # Iterate over the filtered DataFrame
        for i, row in selected_rows.iterrows():
            if len(examples) >= self.number_of_examples:
                break

            # Read the corresponding lines from the source and target files
            with open(self.example_val_dataset_src, 'r') as ex_src_file, open(self.example_val_dataset_tgt,
                                                                              'r') as ex_tgt_file:
                lines_src = ex_src_file.readlines()
                lines_tgt = ex_tgt_file.readlines()

                line_no = int(row['new_line_no']) - 1  # Subtract 1 because list indices start from 0
                ex_src = lines_src[line_no].strip()
                ex_tgt = lines_tgt[line_no].strip()

                example = {"Input": ex_src, "Output": ex_tgt}

                for key in self.given_sentences_feature_dict:
                    print(
                        f"For {key}: Ideal_target/Test_src_input: {self.given_sentences_feature_dict[key]['tgt_ideal']}/{self.given_sentences_feature_dict[key]['src']} \t"
                        f"Obtained_ex_tgt/Obtained_ex_src of {key}: {row['abs_tgt_' + key]}/{row['abs_src_' + key]} \t")
                print(f"\tObtained Example_src_tgt: {ex_src} \t {ex_tgt}")
                examples.append(example)

        return examples



