import anyconfig
import click
import csv
import random

from faker import Faker
from generate_users import User, Tool


@click.command()
@click.option("-i", "--users-tools-file", type=str, required=True)
@click.option("-r", "--ratings-csv-out", type=str, required=True)
@click.option("-p", "--property-csv-out", type=str, required=True)
@click.option("-d", "--demographics-out", type=str, required=True)
def main(users_tools_file, ratings_csv_out, property_csv_out, demographics_out):
    f = Faker()
    input_data = anyconfig.load(users_tools_file)
    users = input_data["users"]
    tools = input_data["tools"]
    capability_rank_header = [
        "CapabilityLabel",
        "CapabilityValue",
        "Survey",
        "SurveyUser",
        "CapabilityRank",
    ]
    capability_rename = {
        "auto_task": "Ability to automate common tasks such as responding to phishing attacks and failed user logins",
        "playbooks": "Playbooks or workflows that are easy to create, configure, share, and combine",
        "rank_alert": "Ability to rank / score alerts so that analysts can easily prioritize alerts from most to least significant",
        "colab": "Ability for analysts in different geographic locations to work simultaneously on an investigation",
        "ticketing": "Ability to prepopulate alert and logging data into tickets",
        "ingest": "Ability to ingest custom logging / alert formats",
    }
    with open(property_csv_out, "w+", newline="") as prop_csv:
        rating_lines = []
        writer = csv.DictWriter(prop_csv, fieldnames=capability_rank_header)
        writer.writeheader()
        for user in users:
            user_tools = []
            user = User(from_dict=user)
            capability_rank = sorted(
                list(user.property_weights.items()), key=lambda x: x[1]
            )
            for i, name in enumerate(capability_rank):
                d = {
                    "CapabilityLabel": capability_rename[name[0]],
                    "CapabilityValue": capability_rename[name[0]],
                    "Survey": "Tool Survey",
                    "SurveyUser": user.name,
                    "CapabilityRank": 6 - i,
                }
                writer.writerow(d)
            for tool in user.tool_assignments:
                user_tools.append(Tool(from_dict=tools[tool - 1]))
            ratings = user.rate_tools(user_tools)
            for rating in ratings:
                rating = rating[1]
                rating["SurveyUser"] = user.name
                rating["SurveyStatus"] = "Submitted"
                rating["Q10Answer"] = f.paragraph()
                rating["Survey"] = "Tool Survey"
                rating["Title"] = "Added by selecting."
                rating["Q1Answer"] = random.choice(
                    [
                        "Never heard of it",
                        "Heard of it",
                        "Used it at least once",
                        "Used to use it frequently",
                        "Currently use it frequently",
                    ]
                )
                rating["video_qual"] = ["terrible", "okay", "great"][
                    rating["video_qual"] - 1
                ]
            rating_lines += ratings
    rename_columns = [
        ["video_qual", "Q2Answer"],
        ["auto_task", "Q7Answer"],
        ["playbooks", "Q6Answer"],
        ["ticketing", "Q9Answer"],
        ["colab", "Q8Answer"],
        ["ingest", "Q5Answer"],
        ["rank_alert", "Q4Answer"],
        ["tool", "Tool"],
        ["rank", "ToolRank"],
        ["overall", "Q3Answer"],
    ]
    column_order = [
        "Title",
        "Survey",
        "Tool",
        "Q1Answer",
        "Q2Answer",
        "Q3Answer",
        "Q4Answer",
        "Q5Answer",
        "Q6Answer",
        "Q7Answer",
        "Q8Answer",
        "ToolRank",
        "Q9Answer",
        "Q10Answer",
        "SurveyUser",
        "SurveyStatus",
    ]
    for rating_line in rating_lines:
        rating_line = rating_line[1]
        for rename in rename_columns:
            rating_line[rename[1]] = rating_line.pop(rename[0])
        for column in column_order[5:11] + [column_order[12]]:
            if random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]):
                rating_line[column] = "can't tell"
            if random.randint(0, 1):
                rating_line["Q10Answer"] = ""
    with open(ratings_csv_out, "w+", newline="") as csvout:
        writer = csv.DictWriter(csvout, fieldnames=column_order)
        writer.writeheader()
        for rating in rating_lines:
            writer.writerow(rating[1])
    column_order = [
        "Title",
        "Survey",
        "Q1Answer",
        "Q2Answer",
        "Q3Answer",
        "SurveyUser",
        "SurveyStatus",
    ]
    with open(demographics_out, "w+", newline="") as csvout:
        writer = csv.DictWriter(csvout, fieldnames=column_order)
        writer.writeheader()
        for user in users:
            line = {
                "Title": "Added by application.",
                "Survey": "Tool Survey",
                "Q1Answer": user['familiarity'],
                "Q2Answer": user['occupation'],
                "Q3Answer": user['experience'],
                "SurveyUser": user['name'],
                "SurveyStatus": "Submitted",
            }
            writer.writerow(line)



    print(property_csv_out)


if __name__ == "__main__":
    main()
