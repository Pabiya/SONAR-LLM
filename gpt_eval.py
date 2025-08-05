import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from time import sleep

import httpx
from tqdm import tqdm

client = None
NUM_SECONDS_TO_SLEEP = 0.5


def get_final_marks(scores):
    pattern = re.compile(r"\d*/\d*")
    subj_scores = scores.split("\n")
    subj_scores_dict = {}
    for subj_score in subj_scores:
        try:
            if "grammar" in subj_score.lower():
                label = pattern.findall(subj_score)
                subj_scores_dict["grammar"] = float(label[0].split("/")[0])
            elif "creativity" in subj_score.lower():
                label = pattern.findall(subj_score)
                subj_scores_dict["creativity"] = float(label[0].split("/")[0])
            elif "consistency" in subj_score.lower():
                label = pattern.findall(subj_score)
                subj_scores_dict["consistency"] = float(label[0].split("/")[0])
            elif "plot" in subj_score.lower():
                label = pattern.findall(subj_score)
                subj_scores_dict["plot"] = float(label[0].split("/")[0])
        except Exception as e:
            continue
    return subj_scores_dict


def get_message_1(story_prompt, generation):
    system_prompt = """Please act as an impartial judge and grade the student's completion in terms of grammar, creativity, consistency with the story's beginning and
whether the plot makes sense."""

    task_prompt_1 = f"""There is the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story.
The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the
prescribed beginning and the student's completion:

{story_prompt}***{generation}

Please provide your short general assessment (3-4 sentences) about the part written by the student (the one after the *** symbol).
Is it gramatically correct? Is it consistent with the beginning of the story? Pay special attention to whether the
student manages to complete the sentence which is split in the middle by the separator ***."""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task_prompt_1,
        },
    ]
    return messages


def get_message_2(gpt_eval, messages):
    task_prompt_2 = f"""Now, grade the student's story in terms of grammar, creativity, consistency and
whether the plot makes sense. Moreover, please provide your best guess of what the age of the student might be,
as reflected from the completion. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E:
10-12. F: 13-16.
The output should be in the following format:
Grammar: 8/10
Creativity: 7/10
Consistency: 7/10
Plot: 6/10
Age group: E (10-12)
It should not be any extra explanation included."""
    
    messages_part_2 = [
        {
            "role": "assistant",
            "content": gpt_eval,
        },
        {
            "role": "user",
            "content": task_prompt_2,
        },
    ]

    messages.extend(messages_part_2)
    return messages


def get_eval_prompt_1(story_prompt, generated_answer):
    messages = get_message_1(story_prompt, generated_answer)
    chat_completion = client.chat.completions.create(messages=messages, model="gpt-4o-2024-08-06", max_tokens=512)
    return messages, chat_completion.choices[0].message.content


def get_eval_prompt_2(evaluation, messages):
    messages = get_message_2(evaluation, messages)
    chat_completion = client.chat.completions.create(messages=messages, model="gpt-4o-2024-08-06", max_tokens=512)
    return chat_completion.choices[0].message.content


def get_gpt_eval(story_prompts, generated_answers):
    mark_dict = {}
    all_labels = []

    i = 0
    for story_prompt, generated_answer in tqdm(zip(story_prompts, generated_answers)):
        messages, gpt_eval = get_eval_prompt_1(story_prompt, generated_answer)
        answer = get_eval_prompt_2(gpt_eval, messages)
        all_labels.append({"review": gpt_eval, "answer": answer})
        mark_dict[i] = get_final_marks(answer)
        i += 1
    return all_labels, mark_dict


def print_mean_scores(score_dict):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for rec in score_dict.values():
        for metric, val in rec.items():
            totals[metric] += val
            counts[metric] += 1
    print("\nMean scores:")
    for metric in ["grammar", "creativity", "consistency", "plot"]:
        if counts[metric]:
            print(f"{metric.capitalize():<12}: {totals[metric]/counts[metric]:.2f} (n={counts[metric]})")
        else:
            print(f"{metric.capitalize():<12}: —")


def main():
    from openai import OpenAI

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--api_key", required=True)
    args = parser.parse_args()

    global client
    client = OpenAI(api_key=args.api_key)

    json_path = Path(args.json)
    if not json_path.exists():
        sys.exit(f"File {json_path} not found.")

    with json_path.open(encoding="utf-8") as f:
        records = json.load(f)

    story_prompts = []
    generated_answers = []
    for rec in records:
        prefix = rec["prefix"]
        full_text = rec["full_text"]
        assert full_text.startswith(prefix), "full_text does not start with prefix"
        story_prompts.append(prefix)
        generated_answers.append(full_text[len(prefix):])

    scores = get_gpt_eval(story_prompts, generated_answers)
    print_mean_scores(scores)


if __name__ == "__main__":
    main()