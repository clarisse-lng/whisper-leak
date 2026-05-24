# Generate a docker compose file dynamically
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', type=int, default=None, help='Number of containers')
parser.add_argument('-c', '--chatbot', default='GPT4oMiniObfuscation', help='Chatbot class name')
args = parser.parse_args()

if args.num is None:
    args.num = int(input("no of containers to be created?: "))
    print()

base = {
    "build": ".",
    "restart": "no",
    "cap_add": ["NET_ADMIN", "NET_RAW"],
    "user": "root",
    "env_file": [".env"],
}

services = {}
for n in range(1, args.num + 1):
    services[f"collector-{n}"] = {
        **base,
        "container_name": f"collector-{n}",
        "volumes": [
            "./data:/app/data",
            f"./prompts/collector-{n}.json:/app/prompts/collector.json:ro",
        ],
        "command": [
            "-c", args.chatbot,
            "-s", "/app/prompts/collector.json",
            "-o", f"data/collector-{n}",
        ],
    }


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


with open("compose.yml", "w") as f:
    yaml.dump(
        {"services": services},
        f,
        Dumper=NoAliasDumper,
        sort_keys=False,
        default_flow_style=False,
    )

print("done")
