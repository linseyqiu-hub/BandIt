import json

with open("test_questions.json", encoding="utf-8") as f:
    d = json.load(f)

print("Number of cases:", len(d))
print("Keys in case 0:", list(d[0].keys()))
print()
print(json.dumps(d[0], indent=2)[:600])