import click
import json
import random
from faker import Faker
from numpy.random import normal

def uniform(a, b):
    def _uniform():
        return random.uniform(a, b)
    return _uniform

def generate_users_tools(tool_assignments, capability_names, property_names, custom_ranges=[], clear_winners=False): 
    tool_set = set()
    users = []
    tools = []
    for tool_assignment in tool_assignments:
        users.append(User(tool_assignment, capability_names=capability_names))
        tool_set = tool_set.union(set(tool_assignment))
    for i in range(len(tool_set)):
        if not clear_winners:
            tools.append(Tool(custom_ranges=custom_ranges, property_names=property_names))
        else:
            if i > 2:
                place = 5
            elif i == 3:
                place = 2
            else:
                place = 1
            tools.append(Tool(custom_ranges=custom_ranges, property_names=property_names, place=place))
    return users, tools
        
class User:
    f = Faker() 
    jitter = normal
    def __init__(self, tool_assignments=[], capability_names="abcdefghijklmnopqrstuvwxyz", from_dict=None):
        if from_dict != None:
            self.__dict__ = from_dict
        else:
            self.niceness = normal(loc=1.0, scale=0.3)
            if self.niceness < 0:
                self.niceness = 0
            self.name = User.f.name()
            self.tool_assignments = tool_assignments
            self.jitter_size = random.uniform(0.3, 1.5)
            self.tool_scores = {}
            self.property_weights = self._generate_capability_ranking(capability_names)
            self.occupation = random.choice(['security operator', 'network operator', 'systems administrator', 'other'])
            self.experience = random.randint(1, 25)
            self.familiarity = random.randint(1, 3)

    @staticmethod
    def _generate_capability_ranking(capability_names):
        # generate n random variables on a normal centered on 1
        capability_importance = {}
        for capability in capability_names:
            capability_importance[capability] = normal(loc=1.0, scale=0.3)
            if capability_importance[capability] < 0:
                capability_importance[capability] = 0
        return capability_importance

    def _rate_tool(self, tool_properties, property_ranges, name):
        ratings = {}
        overall = 0
        for property_name in tool_properties:
            rating = tool_properties[property_name]*self.niceness
            if property_name in self.property_weights:
                property_weight = self.property_weights[property_name]
                overall += property_weight * rating / len(self.property_weights)
            ratings[property_name] = User.jitter(loc=rating, scale=self.jitter_size)
        ratings['overall'] = User.jitter(loc=overall, scale=self.jitter_size)
        for property_name in ratings:
            if property_name in property_ranges:
                r = property_ranges[property_name]
            else:
                r = [1, 5]
            if ratings[property_name] < r[0]:
                ratings[property_name] = r[0]
            elif ratings[property_name] > r[1]:
                ratings[property_name] = r[1]
            else:
                ratings[property_name] = round(ratings[property_name])
        ratings['tool'] = name
        return overall, ratings
    
    def rate_tools(self, tools):
        ratings = []
        for tool in tools:
            ratings.append(self._rate_tool(tool.properties, tool.property_ranges, tool.name))
        ratings.sort(key=lambda x: x[0])

        swap = random.randint(0, 1)
        if swap:
            swap_index = random.randint(0, len(ratings) - 2)
            tmp = ratings[swap_index]
            ratings[swap_index] = ratings[swap_index + 1]
            ratings[swap_index + 1] = tmp
        i = len(ratings)
        for rating in ratings:
            rating[1]["rank"] = i
            i -= 1
        return ratings

class Tool:
    f = Faker()
    def __init__(self, custom_ranges=[], property_names='abcdefghijklmnopqrstuvwxyz', from_dict=None, place=None):
        if from_dict != None:
            self.__dict__ = from_dict
        else:
            self.properties = {}
            self.property_ranges = {}
            self.name = self.f.word()
            property_generators = {}
            if place is None:
                for name in property_names:
                    property_generators[name] = uniform(1, 5)
                    self.property_ranges[name] = [1, 5]
                for name in custom_ranges:
                    self.property_ranges[name] = custom_ranges[name]
                    property_generators[name] = uniform(*custom_ranges[name]) 
                for name in property_generators:
                    self.properties[name] = property_generators[name]()
            else:

                def winner_generator(max_score, place):
                    def _winner_generator():
                        return max_score - place + 1
                    return _winner_generator

                for name in property_names:
                    property_generators[name] = winner_generator(5, place)
                    self.property_ranges[name] = [1, 5]
                for name in custom_ranges:
                    self.property_ranges[name] = custom_ranges[name]
                    property_generators[name] = uniform(*custom_ranges[name]) 
                for name in property_generators:
                    self.properties[name] = property_generators[name]()


output_help = "Output json file to store generated tool and user set in."
tool_assignments_file_help = "Input tool assignments file. This file should have a json list of integers on each line. Each array corresponds to the list of tools a user will rate. For example, a line of [0, 3, 4] will generate a user which rates tools 0, 3 and 4."
custom_ranges_help = "Some tools aspects may not be rated on a range from 1 to 5. This argument takes a json object which maps tool property names to the rating range of that property. IE {\"video_qual\": [1, 3]} will make the video quality tool property be rated from 1 to 3."
property_names_help = "A comma separated string of tool properties to generate."

@click.command()
@click.option('-o', '--output', type=str, required=True)
@click.option('-i', '--tool-assignments-file', type=str, required=True)
@click.option('-r', '--custom-ranges', type=str, default='{}')
@click.option('-p', '--property_names', type=str, required=True)
@click.option('-c', '--capability-names', type=str, default="")
@click.option('--clear-winners', is_flag=True, help="Makes some tools clearly better than the rest")
def main(output, tool_assignments_file, property_names, custom_ranges, capability_names, clear_winners):
    """
    Generate a set of users and tools with randomly generated properties.
    """

    tool_assignment_list = []
    with open(tool_assignments_file) as f:
        for line in f.readlines():
            tool_assignment_list.append(json.loads(line))
    users, tools = generate_users_tools(tool_assignment_list, capability_names.split(','), property_names.split(','), json.loads(custom_ranges), clear_winners)
    output_dict = {"users": [], "tools": []}
    for user in users:
        output_dict["users"].append(user.__dict__)
    for tool in tools:
        output_dict["tools"].append(tool.__dict__)
    with open(output, "w+") as o:
        json.dump(output_dict, o, indent=2)


if __name__ == "__main__":
    #t = Tool(property_names=["prop1", "prop2", "prop3"])
    #u = User([1, 2, 3], capability_names=["prop1", "prop3"]) 
    #print("User:\n" + json.dumps(u.__dict__, indent=2))
    #print("Tool: " + str(t.properties))
    #print("Ratings: " + str(u._rate_tool(t.properties, t.property_ranges)))
    main()
    #print(User._generate_capability_ranking())
