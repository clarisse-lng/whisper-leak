# Generate a docker compose file dynamically
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', type=int, default=None, help='Number of containers')
parser.add_argument('-c', '--chatbot', default='GPT4oMini', help='Chatbot class name')
parser.add_argument('-e', '--env-file', default='.env', help='Env file to use for API keys (default: .env)')
parser.add_argument('-p', '--prefix', default='copy-collector', help='Container name prefix (default: copy-collector)')
args = parser.parse_args()

if args.num is None:
    args.num = int(input("no of containers to be created?: "))
    print()

base = {
    "build": ".",
    "restart": "no",
    "cap_add": ["NET_ADMIN", "NET_RAW"],
    "user": "root",
    "env_file": [args.env_file],
}

services = {}
for n in range(1, args.num + 1):
    name = f"{args.prefix}-{n}"
    services[name] = {
        **base,
        "container_name": name,
        "volumes": [
            "./data:/app/data",
            f"./prompts/collector-{n}.json:/app/prompts/collector.json:ro",
        ],
        "command": [
            "-c", args.chatbot,
            "-p", "/app/prompts/collector.json",
            "-o", f"data/{name}",
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
