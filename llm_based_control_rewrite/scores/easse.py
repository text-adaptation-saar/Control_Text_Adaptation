import subprocess
import json
import ast
import re
from llm_based_control_rewrite.utils.helpers import write_easse_stats
# import lens
# from lens.lens_score import LENS
import argparse


def calculate_easse(path_to_orig_sents, path_to_refs_sents, path_to_sys_sents):

    # command = "easse evaluate -m bleu,sent_bleu,sari,fkgl,bertscore,sari_legacy,sari_by_operation,f1_token -t custom -tok moses --orig_sents_path $1 --refs_sents_paths $2  --sys_sents_path $3"
    # default tok="13a": Tokenizes an input line using a relatively minimal tokenization
    #         that is however equivalent to mteval-v13a, used by WMT https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_13a.py
    command = "easse evaluate -m bleu,sent_bleu,sari,fkgl,bertscore,sari_legacy,sari_by_operation,f1_token -t custom -tok 13a --orig_sents_path $1 --refs_sents_paths $2  --sys_sents_path $3"
    command = command.replace("$1", path_to_orig_sents).replace("$2", path_to_refs_sents).replace("$3", path_to_sys_sents)

    # Bertscore between orig_test sent vs. system_outputs
    # command_2 = "easse evaluate -m bertscore -t custom -tok moses --orig_sents_path $1 --refs_sents_paths $2  --sys_sents_path $3"
    command_2 = "easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path $1 --refs_sents_paths $2  --sys_sents_path $3"
    command_2 = command_2.replace("$1", path_to_orig_sents).replace("$2", path_to_orig_sents).replace("$3", path_to_sys_sents)

    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stdout

    process_2 = subprocess.run(command_2, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output_2 = process_2.stdout
    try:
        # bertscore_precision, 0.317,bertscore_recall, 0.478,bertscore_f1
        print(output)
        output = output.replace("bertscore_precision", "bertscore_precision_hyp_vs_ref").replace(
            "bertscore_recall", "bertscore_recall_hyp_vs_ref").replace("bertscore_f1", "bertscore_f1_hyp_vs_ref")
        print(output)

        print(output_2)
        output_2 = output_2.replace("bertscore_precision", "bertscore_precision_hyp_vs_src").replace(
            "bertscore_recall", "bertscore_recall_hyp_vs_src").replace("bertscore_f1", "bertscore_f1_hyp_vs_src")

        output_dict = json.loads(output.replace("'", '"'))  # JSON requires double quotes
        output_dict_2 = json.loads(output_2.replace("'", '"'))  # JSON requires double quotes

        output_dict.update(output_dict_2)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        output_dict = {}  # Use an empty dictionary as a fallback

    final_line=""
    for metric, value in output_dict.items():
        final_line += f"{metric}, {value},"

    return final_line.strip(',')


def calculate_easse_on_mac(path_to_orig_sents, path_to_refs_sents, path_to_sys_sents):
    print("hit calculate_easse_on_mac() method")

    command = "easse evaluate -m bleu,sent_bleu,sari,fkgl,sari_legacy,sari_by_operation,f1_token -t custom -tok 13a --orig_sents_path $1 --refs_sents_paths $2 --sys_sents_path $3"
    command = command.replace("$1", path_to_orig_sents).replace("$2", path_to_refs_sents).replace("$3",path_to_sys_sents)

    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stdout

    try:
        print("process the output")
        print(f"output: {output}")

        # Use regular expression to extract the dictionary part from the output
        dict_str = re.search(r"\{.*\}", output).group(0)
        print(f"dict_str: {dict_str}")

        output_dict = json.loads(dict_str.replace("'", '"'))  # JSON requires double quotes

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        output_dict = {}  # Use an empty dictionary as a fallback

    # try:
    #     # Safely evaluate the string as a Python dictionary
    #     output_dict = ast.literal_eval(dict_str)
    #     print(f"output_dict: {output_dict}")
    # except (SyntaxError, ValueError, AttributeError) as e:
    #     print(f"Error evaluating output as dictionary: {e}")
    #     output_dict = {}  # Use an empty dictionary as a fallback

    final_line = ""
    for metric, value in output_dict.items():
        final_line += f"{metric}, {value},"

    return final_line.strip(',')


def calculate_easse_fkgl(path_to_orig_sents, path_to_refs_sents, path_to_sys_sents):
    print("hit calculate_easse_on_mac() method")

    command = "easse evaluate -m fkgl -t custom -tok 13a --orig_sents_path $1 --refs_sents_paths $2 --sys_sents_path $3"
    command = command.replace("$1", path_to_orig_sents).replace("$2", path_to_refs_sents).replace("$3",path_to_sys_sents)

    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stdout

    try:
        print("process the output")
        print(f"output: {output}")

        # Use regular expression to extract the dictionary part from the output
        dict_str = re.search(r"\{.*\}", output).group(0)
        print(f"dict_str: {dict_str}")

        # Safely evaluate the string as a Python dictionary
        output_dict = ast.literal_eval(dict_str)
        print(f"output_dict: {output_dict}")
    except (SyntaxError, ValueError, AttributeError) as e:
            print(f"Error evaluating output as dictionary: {e}")
            output_dict = {}  # Use an empty dictionary as a fallback

    return output_dict

# def lens_score(complex_file_path, simple_file_path, ref_file_path,
#                model_path="data_auxiliary/en/lens/LENS/checkpoints/epoch=5-step=6102.ckpt"):
#
#     # Original LENS is a real-valued number.
#     # Rescaled version (rescale=True) rescales LENS between 0 and 100 for better interpretability.
#     # You can also use the original version using rescale=False
#     metric = LENS(model_path, rescale=True)
#
#     # complex = ["They are culturally akin to the coastal peoples of Papua New Guinea."]
#     # simple = ["They are culturally similar to the people of Papua New Guinea."]
#     # references = [
#     #     [
#     #         "They are culturally similar to the coastal peoples of Papua New Guinea.",
#     #         "They are similar to the Papua New Guinea people living on the coast."
#     #     ]
#     # ]
#
#     with open(complex_file_path, 'r') as test_input, open(simple_file_path, 'r') as sys_output_hyp, \
#             open(ref_file_path, 'r') as test_gold_ref:
#         test_input_dataset = test_input.readlines()
#         test_sys_output_hyp = sys_output_hyp.readlines()
#         test_gold_ref_dataset = test_gold_ref.readlines()
#
#         for line_number, (src_sentence, output_hyp, gold_ref) in enumerate(zip(test_input_dataset,test_sys_output_hyp, test_gold_ref_dataset), start=1):
#             scores = metric.score(complex, output_hyp, gold_ref, batch_size=8, gpus=1)
#             print(scores)


def calculate_bert_score(path_to_orig_sents, path_to_refs_sents, path_to_sys_sents, save_path):

    # command = "easse evaluate -m bleu,sent_bleu,sari,fkgl,bertscore,sari_legacy,sari_by_operation,f1_token -t custom -tok moses --orig_sents_path $1 --refs_sents_paths $2  --sys_sents_path $3"
    # default tok="13a": Tokenizes an input line using a relatively minimal tokenization
    #         that is however equivalent to mteval-v13a, used by WMT https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_13a.py
    command = "easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path $1 --refs_sents_paths $2  --sys_sents_path $3"
    command = command.replace("$1", path_to_orig_sents).replace("$2", path_to_refs_sents).replace("$3", path_to_sys_sents)

    # Bertscore between orig_test sent vs. system_outputs
    # command_2 = "easse evaluate -m bertscore -t custom -tok moses --orig_sents_path $1 --refs_sents_paths $2  --sys_sents_path $3"
    command_2 = "easse evaluate -m bertscore -t custom -tok 13a --orig_sents_path $1 --refs_sents_paths $2  --sys_sents_path $3"
    command_2 = command_2.replace("$1", path_to_orig_sents).replace("$2", path_to_orig_sents).replace("$3", path_to_sys_sents)

    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stdout

    process_2 = subprocess.run(command_2, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output_2 = process_2.stdout
    try:
        # bertscore_precision, 0.317,bertscore_recall, 0.478,bertscore_f1
        print(output)
        output = output.replace("bertscore_precision", "bertscore_precision_hyp_vs_ref").replace(
            "bertscore_recall", "bertscore_recall_hyp_vs_ref").replace("bertscore_f1", "bertscore_f1_hyp_vs_ref")
        print(output)

        print(output_2)
        output_2 = output_2.replace("bertscore_precision", "bertscore_precision_hyp_vs_src").replace(
            "bertscore_recall", "bertscore_recall_hyp_vs_src").replace("bertscore_f1", "bertscore_f1_hyp_vs_src")

        output_dict = json.loads(output.replace("'", '"'))  # JSON requires double quotes
        output_dict_2 = json.loads(output_2.replace("'", '"'))  # JSON requires double quotes

        output_dict.update(output_dict_2)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        output_dict = {}  # Use an empty dictionary as a fallback

    final_line=""
    for metric, value in output_dict.items():
        final_line += f"{metric}, {value},"

    output_generation_path = path_to_orig_sents.replace("/input.txt", "").replace("/maxdepdepth_-1_maxdeplength_-1_diffwordscount_-1_avgwordcount_-1_length_-1_leven_-1_grade_-1","")
    write_easse_stats(f"{save_path}/easse_stats.csv", final_line, output_generation_path)
    # return final_line.strip(',')


if __name__=="__main__":
    # use default textstat setting where lang="en". I don't explicitly set the language using set_lang.
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_sents_path", required=True, help="yaml config file for feature based evaluation")
    parser.add_argument("--refs_sents_paths", required=True, help="yaml config file for feature based evaluation")
    parser.add_argument("--sys_sents_path", required=True, help="yaml config file for feature based evaluation")
    parser.add_argument("--save_path", required=True, help="yaml config file for feature based evaluation")
    args = vars(parser.parse_args())

    calculate_bert_score(args["orig_sents_path"], args["refs_sents_paths"], args["sys_sents_path"], args["save_path"])
