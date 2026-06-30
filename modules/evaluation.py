'''
this
'''



import mlflow

TEST_QUERIES = [
    {"query": "what is the battery capacity?",                "expected": "factual"},
    {"query": "what is the range of the byd seal?",           "expected": "factual"},
    {"query": "how fast does it accelerate?",                 "expected": "factual"},
    {"query": "what screen size does it have?",               "expected": "factual"},
    {"query": "does it have apple carplay?",                  "expected": "factual"},
    {"query": "is tesla better than byd?",                    "expected": "opinion"},
    {"query": "what do people think of the interior?",        "expected": "opinion"},
    {"query": "would you recommend this car?",                "expected": "opinion"},
    {"query": "do viewers like the design?",                  "expected": "opinion"},
    {"query": "is it worth the price according to comments?", "expected": "opinion"},
]

def evaluate_routing(graph, test_queries=TEST_QUERIES, model_name = 'qwen3:8b', run_name='routing_eval'):
  mlflow.set_experiment('RAG_EVALUATION')

  with mlflow.start_run(run_name=run_name):
    mlflow.log_param('model', model_name)
    mlflow.log_param('n_queries', len(test_queries))

    correct = 0
    details = []

    for item in test_queries:
      result = graph.invoke({'query': item['query']})
      predicted = result['category']
      expected = item['expected']

      is_correct = expected in predicted.lower()
      correct += int(is_correct)

      details.append({
        'query': item['query'],
        'expected': expected,
        'predicted': predicted,
        'correct': is_correct
      })

    accuracy = correct / len(test_queries)

    mlflow.log_metric('routing accuracy', accuracy)
    mlflow.log_dict({'results':details}, 'routing_details.json')

    print(f"routing accuracy ({model_name}): {accuracy:.2f} ")
    print(f"({correct}/{len(test_queries)})")

    for d in details:
      mark = 'good' if d['correct'] else 'not good'
      print(f"  {mark}[{d['expected']} vs {d['predicted']}] {d['query']}")
  return accuracy