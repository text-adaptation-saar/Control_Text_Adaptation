import argparse
import os

# from ControlTS_T5.scripts.generate_code_by_sarubi import generate_for_inference
# from ControlTS_T5.source.evaluate import simplify_file, evaluate_all_metrics
from llm_based_control_rewrite.t5_ft_scripts.preprocssed_data_creation import preprocess_dataset, \
    prepend_ratios_to_source_text, prepend_ratios_to_source_text_ACCESS, prepend_ratios_to_source_text_all_7f
from llm_based_control_rewrite.utils.helpers import load_yaml, map_control_token_values
from only_calculate_scores import generate_scores


def t5_infer(**kwargs):
    config_yaml = load_yaml(kwargs["config"])
    requested_feature_dict_vals, requested_absolute_value, output_folder_name = map_control_token_values(kwargs)
    print("Start T5 infer!")
    output_generation_path = config_yaml["path_to_output_generation"] + "/" + output_folder_name
    if not os.path.exists(output_generation_path):
        os.makedirs(output_generation_path)
    generate_with_T5_ft(config_yaml, output_generation_path,
                        kwargs["predicted_ratio_file_given"] if kwargs["predicted_ratio_file_given"] is not None else False)
    # generate_scores(config_yaml, requested_feature_dict_vals, output_generation_path,
    #                      kwargs["predicted_ratio_file_given"] if kwargs["predicted_ratio_file_given"] is not None else False)


def generate_with_T5_ft(config_yaml, output_generation_path,
                         predicted_ratio_file_given=False):

    path_to_write_input_lines_of_outputs = output_generation_path + "/input.txt"
    path_to_gold_ref = output_generation_path + "/gold_ref.txt"
    path_to_output = output_generation_path + "/output.txt"

    print("T5 FT request process enabled!")
    t5_ft_processed_file = f"{output_generation_path}/input_preprocessed.txt"

    print(f"predicted_ratio_file_given is: {predicted_ratio_file_given}")
    def copy_first_200_lines(source_path, dest_path):
        with open(source_path, 'r', encoding='utf-8') as source_file:
            with open(dest_path, 'w', encoding='utf-8') as dest_file:
                for _ in range(config_yaml["number_of_lines_to_test"]):
                    line = source_file.readline()
                    if not line:
                        break
                    dest_file.write(line)

    copy_first_200_lines(config_yaml["path_to_input_test_data"], path_to_write_input_lines_of_outputs )
    copy_first_200_lines(config_yaml["path_to_test_gold_ref_data"], path_to_gold_ref )

    print(f"predicted_ratio_file_given: {predicted_ratio_file_given}")
    if predicted_ratio_file_given:
        # prepend_ratios_to_source_text(source_file_path=path_to_write_input_lines_of_outputs,
        #                               ratios_file_paths=config_yaml["predicted_ratio_file_path"].split(','),
        #                               processed_source_file_path=t5_ft_processed_file)

        # prepend_ratios_to_source_text_ACCESS(source_file_path=path_to_write_input_lines_of_outputs,
        #                               ratios_file_paths=config_yaml["predicted_ratio_file_path"].split(','),
        #                               processed_source_file_path=t5_ft_processed_file)

        prepend_ratios_to_source_text_all_7f(source_file_path=path_to_write_input_lines_of_outputs,
                                             ratios_file_paths=config_yaml["predicted_ratio_file_path"].split(','),
                                             processed_source_file_path=t5_ft_processed_file)

    # simplify_file(t5_ft_processed_file, path_to_output, features_kwargs=None, model_dirname=config_yaml["models"])
    # generate_for_inference(config_yaml["models"], t5_ft_processed_file, path_to_output, path_to_gold_ref)
    # print(evaluate_all_metrics(path_to_write_input_lines_of_outputs, path_to_output, [path_to_gold_ref]))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml config file for prompt based generation")
    parser.add_argument("--requested_dependency_depth", required=False, help="Target sentence's dependency depth")
    parser.add_argument("--requested_dependency_length", required=False, help="Target sentence's dependency length")
    parser.add_argument("--requested_difficult_words", required=False, help="Target sentence's no of difficult words")
    parser.add_argument("--requested_frequency", required=False, help="Target sentence's frequency")
    parser.add_argument("--requested_length", required=False, help="Target sentence's character length")
    parser.add_argument("--requested_levenshtein", required=False, help="Target sentence's levenshtein")
    parser.add_argument("--requested_word_count", required=False, help="Target sentence's word count")
    parser.add_argument("--requested_grade_level", required=False, help="Target sentence's word count")
    parser.add_argument("--requested_absolute_value", required=False, help="Default requested feature values are in "
                                                                           "ratio, make this param true, if you want to"
                                                                           "treat feature value as it is")
    parser.add_argument("--predicted_ratio_file_given", action="store_true", required=False, help="Predicted ratio using linear regression file is given")
    # parser.add_argument("--requested_frequency_category", required=False, help="Target sentence's frequency")
    args = vars(parser.parse_args())
    t5_infer(**args)