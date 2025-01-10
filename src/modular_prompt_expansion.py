import csv
import copy
import argparse
import yaml
import pandas as pd

from utils import product_of_dict_grid, independent_variations_of_dict_grid, multilingual

"""
Script to expand the modular prompt .yaml at MODULAR_PROMPT_PATH, filled in with topics from a .txt at TOPICS_FILE_PATH, 
to a .csv file at QUESTIONS_FILE_PATH with full questions (including settings).
"""

# EXPANSION_FN = product_of_dict_grid
# EXPANSION_FN = independent_variations_of_dict_grid
EXPANSION_FN = multilingual


def main(args):
    # Read files to combine.
    with open(args.modular_prompt_path, 'r') as file:
        modular_prompt = yaml.safe_load(file)
    stage_keys = [key for key in modular_prompt.keys() if key.startswith('stage_')]

    topic_df = []
    for topics_path in args.topics_paths:
        with open(topics_path, 'r') as file:
            topic_df.append(pd.read_csv(
                    file,
                    index_col=0,
                    na_filter=False,
                    quoting=csv.QUOTE_ALL,
                    escapechar='\\',
                    encoding='utf-8'
                )
            )
    topic_df = pd.concat(topic_df)
    print(topic_df.head())
    # Expand the modular prompt.
    prompt_template_settings = EXPANSION_FN(modular_prompt)
    settings_df = pd.DataFrame(prompt_template_settings)
    settings_df.to_csv(args.output_path.replace('.csv', "_settings.csv"), index=False)

    prompts = []
    for p, prompt_template_setting in enumerate(prompt_template_settings):
        answer_scale = prompt_template_setting['answer_scale']
        scale_string = ', '.join(f"'{w}'" for w in answer_scale)
        # scale_string = ', '.join(f"'{w}'" for w in answer_scale[:-1])
        # scale_string += f" or '{answer_scale[-1]}'"

        for t, topic_info in topic_df.iterrows():
            try:
                topic = topic_info[f'name-{prompt_template_setting["language_code"]}']
            except KeyError:
                topic = topic_info['name-en']

            prompt = {'prompt_template_idx': p,
                      'topic_idx': t,
                      'topic': topic,  # not strictly necessary to include, but helpful.
                      } | copy.deepcopy(prompt_template_setting)
            
            # Add wikidata_id only if it exists in the data
            if 'wikidata_id' in topic_info:
                prompt['wikidata_id'] = topic_info['wikidata_id']

            prompt[stage_keys[-1]] = f"{prompt[stage_keys[-1]]} {prompt['assurances']}"

            for stage_key in stage_keys:
                # Replace placeholders in each stage.
                prompt[stage_key] = (prompt[stage_key]
                                     .replace('<VAR>', topic)
                                     .replace('<SCALE>', scale_string))
            prompts.append(prompt)
    df = pd.DataFrame(prompts)

    # Write prompts to a .csv file.
    with open(args.output_path, 'w') as file:
        df.to_csv(file, index_label='question_idx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Expand a modular prompt .yaml to a .csv file where topic placeholders"
                                                 "are filled in.")
    parser.add_argument('--modular_prompt_path', type=str, help="Path to the modular prompt .yaml file.")
    parser.add_argument('--output_path', type=str, help="Path to the output .csv file.")
    parser.add_argument('--topics_paths', nargs='+', help="Path to the topics .csv file. If multiple paths are "
                                                          "provided, all topics in all files are used.")
    main(parser.parse_args())
