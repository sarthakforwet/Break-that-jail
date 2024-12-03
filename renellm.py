import time
import json
import os
import argparse
import random

from utils.data_utils import data_reader
from utils.llm_completion_utils import claudeCompletion
from utils.prompt_rewrite_utils import shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange
from utils.harmful_classification_utils import harmful_classification


def main(args):
    data = data_reader(args.data_path)

    # single prompt jailbreak
    if args.prompt is not None:
        data = [args.prompt]

    operations = [shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange]

    # save the generated data to data_list
    data_list = []

    for idx, harm_behavior in enumerate(data):

        start_time = time.time()
        temp_harm_behavior = harm_behavior

        if not os.path.exists('./results/renellm/temp'):
            os.makedirs('./results/renellm/temp')

        # save the results for every 10 samples
        if idx != 0 and idx % 10 == 0:
            file_name = f"./results/renellm/temp/{args.save_suffix}_{idx}.json"

            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4)

            file_path = os.path.abspath(file_name)
            print(f"\nThe temporary file has been saved to: {file_path}\n")

        loop_count = 0

        while True:
            print(
            "\n################################\n"
            f"Current Data: {idx+1}/{len(data)}, {harm_behavior}\n"
            f"Current Iteration Round: {loop_count+1}/{args.iter_max}\n"
            "################################\n")
            # prompt rewriting
            while True:
                print(f"******* Start idx {idx} Prompt Rewriting! *******")
                # randomly select the number of rewriting operations.
                # n = random.randint(1, 6)
                n = 1 # Revert to the above line after debugging
                # arrange the execution order randomly
                operation_indexes = random.sample(range(1), n)
                print(f"The number of rewriting functions is: {n}")
                print(f"The order of the rewriting operations is: {operation_indexes}\n")

                temp_rewrite_results = [['original prompt', temp_harm_behavior]]
                for index in operation_indexes:
                    print(f"Excute function {index}: {operations[index].__name__}")
                    harm_behavior = operations[index](args, harm_behavior)
                    print(f"Rewritten prompt: {harm_behavior}\n")
                    temp_rewrite_results.append([operations[index].__name__, harm_behavior])

                # check if the prompt meets the requirements; if it does, break out of the loop.
                print("******* Check if the prompt has been modified to be harmless *******")
                label = harmful_classification(args, harm_behavior)
                print(f"\nThe label given by {args.judge_model} is: {label}\n")
                # if label == "1":
                if label == "LABEL_1":
                    print(f"******* Prompt idx {idx} Rewriting Done! *******\n")
                    break
                else: # LABEL_0
                    print(f"******* Prompt idx {idx} Rewriting Failed! *******\n")
                    harm_behavior = temp_harm_behavior
                    continue

            # scenario nesting
            print(f"******* Start idx {idx} Prompt Scenario Nesting! *******")
            print(f"harm_behavior:\n{harm_behavior}\n")

            print(f"******* Start idx {idx} Prompt Jailbreaking on {args.attack_model}! *******")
            checkpoint = "ARahul2003/lamini_flan_t5_detoxify_rlaif"

            from transformers import AutoTokenizer
            from trl import AutoModelForSeq2SeqLMWithValueHead
            import torch

            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            base_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(checkpoint,
                                                        torch_dtype=torch.float32)

            base_model = base_model.to('cpu')

            # Tokenize input
            harm_behavior += "Output should be less than 10 words."
            inputs = tokenizer(harm_behavior, return_tensors="pt", truncation=True, padding=True, max_length=512)

            # Generate text
            outputs = base_model.generate(
                input_ids=inputs["input_ids"].to('cpu'),
                attention_mask=inputs["attention_mask"].to('cpu'),
                max_length=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
            )

            # Decode generated text
            attack_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"The output of the attacked model {args.attack_model} is:\n\n{attack_output}\n")

            loop_count += 1

            if "apologize" not in attack_output or "sorry" not in attack_output.lower():

                print(f"******* Determine if the attacked model's response to the nested idx {idx} prompt is harmful *******")
                jailbreak_label = harmful_classification(args, attack_output)
                print(f'Jailbreak Label is {jailbreak_label}')

            else:
                if loop_count < args.iter_max:
                    print(f"\nJailbreaking idx {idx} Prompt Failed!\n")
                    harm_behavior = temp_harm_behavior
                    continue
                else:
                    print(f"\n******* Exceeded the maximum number of iterations {args.iter_max}, adopt the current round results and end the loop.*******\n")
                    # save the data
                    item = {}
                    item['idx']  = idx
                    item['original_harm_behavior']  = temp_harm_behavior
                    item['rewrite_model']  = args.rewrite_model
                    item['operation_functions'] = [operations[i].__name__ for i in operation_indexes]
                    item['rewrite_track'] = temp_rewrite_results
                    item['rewritten_prompt']  = harm_behavior
                    item['judge_model']  = args.judge_model
                    item['attack_model']  = args.attack_model
                    item['attack_output']  = attack_output
                    item['model_label'] = "0"
                    item['iteration_count']  = loop_count
                    data_list.append(item)

                    break

            if jailbreak_label == "LABEL_1":
                print(f"\n******* Jailbreaking idx {idx} Prompt Successful! *******\n")
                # save the data
                item = {}
                item['idx']  = idx
                item['original_harm_behavior']  = temp_harm_behavior
                item['rewrite_model']  = args.rewrite_model
                item['operation_functions'] = [operations[i].__name__ for i in operation_indexes]
                item['rewrite_track'] = temp_rewrite_results
                item['rewritten_prompt']  = harm_behavior
                item['judge_model']  = args.judge_model
                item['attack_model']  = args.attack_model
                item['attack_output']  = attack_output
                item['model_label'] = "1"
                item['iteration_count']  = loop_count


                end_time = time.time()
                elapsed_time = end_time - start_time
                item['time_cost'] = elapsed_time

                data_list.append(item)

                break
            else:
                if loop_count < args.iter_max:
                    print(f"\nJailbreaking idx {idx} Prompt Failed!\n")
                    harm_behavior = temp_harm_behavior
                    continue
                else:
                    print(f"\n******* Exceeded the maximum number of iterations {args.iter_max}, adopt the current round results and end the loop.*******\n")
                    # save the data
                    item = {}
                    item['idx']  = idx
                    item['original_harm_behavior']  = temp_harm_behavior
                    item['rewrite_model']  = args.rewrite_model
                    item['rewrite_functions'] = [operations[i].__name__ for i in operation_indexes]
                    item['rewrite_track'] = temp_rewrite_results
                    item['rewritten_prompt']  = harm_behavior
                    item['judge_model']  = args.judge_model
                    item['attack_model']  = args.attack_model
                    item['attack_output']  = attack_output
                    item['model_label'] = "0"
                    item['iteration_count']  = loop_count
                    data_list.append(item)

                    break

    # save all data after jailbreaking.
    file_name = f"logs.json"

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    file_path = os.path.abspath(file_name)
    print(f"\nThe final file has been saved to:\n{file_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # advbench
    parser.add_argument('--data_path', type=str, default='data/advbench/harmful_behaviors.csv')
    parser.add_argument('--prompt', type=str, default=None)     # your single prompt

    args = parser.parse_args()
    main(args)
